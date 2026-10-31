"""Source-independent telemetry calculations. Distances in m, time in s, speed in m/s."""
from dataclasses import dataclass, field
import numpy as np
from scipy.io import loadmat
from pathlib import Path

START_LON_LAT = np.array([[-1.0111, 52.07812], [-1.01113, 52.07800]])

@dataclass
class Channel:
    time: np.ndarray
    value: np.ndarray
    unit: str
    kind: str = "recorded"

@dataclass
class Session:
    time: np.ndarray
    xy: np.ndarray
    line: np.ndarray
    channels: dict[str, Channel] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

@dataclass
class Crossing:
    time: float
    xy: np.ndarray
    index: int
    fraction: float
    line_fraction: float

@dataclass
class Lap:
    number: int
    time: np.ndarray  # absolute logger time
    xy: np.ndarray
    distance: np.ndarray
    speed: np.ndarray
    channels: dict[str, np.ndarray]
    @property
    def duration(self):
        return float(self.time[-1] - self.time[0])
    @property
    def progress(self):
        return self.distance / self.distance[-1]

def validate_series(time, values, name):
    if len(time) < 2 or len(time) != len(values):
        raise ValueError(f"{name}: inconsistent/insufficient samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: non-finite samples")
    if np.any(np.diff(time) <= 0):
        raise ValueError(f"{name}: timestamps must be strictly increasing")

def local_xy(lon_lat, origin):
    """WGS84 tangent-plane linearisation at origin [lon, lat] in degrees.
    Uses ellipsoid meridional/prime-vertical radii; suitable for this sub-km site.
    Horizontal path only: altitude is deliberately excluded.
    """
    lon_lat, origin = np.asarray(lon_lat), np.asarray(origin)
    a, e2 = 6378137.0, 6.69437999014e-3
    phi = np.deg2rad(origin[1])
    denom = 1 - e2 * np.sin(phi)**2
    north_radius = a * (1-e2) / denom**1.5
    east_radius = a / np.sqrt(denom)
    radians = np.deg2rad(lon_lat - origin)
    return radians * [east_radius * np.cos(phi), north_radius]

def load_onboarding(folder):
    """Only this adapter knows the MAT filenames/variable names."""
    folder = Path(folder)
    data = loadmat(folder / "Onboarding_GPS.mat", squeeze_me=True)
    t = np.asarray(data["GPStime"], dtype=float).ravel()
    ll = np.column_stack([data["GPSlon"], data["GPSlat"]])
    validate_series(t, ll, "GPS")
    origin = START_LON_LAT.mean(axis=0)
    channels = {}
    specs = [
        ("throttle", "Onboarding_rThrottle.mat", "ThrottleTime", "Throttle", "%"),
        ("brake", "Onboarding_pBrakeF.mat", "BrakeTime", "BrakeFront", "sensor units (unknown)"),
        ("power", "Onboarding_Power.mat", "PowerTime", "Power", "kW"),
    ]
    for name, filename, time_key, value_key, unit in specs:
        if not (folder / filename).exists():
            continue
        mat = loadmat(folder / filename, squeeze_me=True)
        channels[name] = Channel(np.asarray(mat[time_key], float).ravel(),
                                 np.asarray(mat[value_key], float).ravel(), unit)
        validate_series(channels[name].time, channels[name].value, name)
    return Session(t, local_xy(ll, origin), local_xy(START_LON_LAT, origin),
                   channels, {"origin_lon_lat": origin.tolist(),
                   "source": "FSUK 2025 endurance stint, supplied TBRe onboarding files",
                   "units_source": "Extension_Task_Base_Script.m section 10; MAT arrays have no unit labels"})

def cross2(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

def find_crossings(time, xy, line):
    """Negative-to-positive crossings, linear time/position interpolation.
    Exact-on-line samples are bridged using their surrounding nonzero sides;
    a touch returning to the same side is not a crossing.
    """
    validate_series(time, xy, "crossing input")
    line = np.asarray(line, float)
    direction = line[1] - line[0]
    norm2 = direction @ direction
    if norm2 == 0:
        raise ValueError("Start/finish line has zero length")
    side = cross2(direction, xy - line[0])
    nonzero = np.flatnonzero(np.abs(side) > 1e-10)
    crossings, candidates = [], []
    for i, j in zip(nonzero[:-1], nonzero[1:]):
        if side[i] < 0 < side[j]:
            fraction = float(-side[i] / (side[j] - side[i]))
            # If a sample is exactly on the line, use its actual timestamp.
            if j > i+1:
                j = i+1
                fraction = 1.0
            point = xy[i] + fraction*(xy[j]-xy[i])
            line_fraction = float((point-line[0]) @ direction / norm2)
            crossing = Crossing(float(time[i]+fraction*(time[j]-time[i])),
                                point, int(i), fraction, line_fraction)
            candidates.append(crossing)
            if -1e-10 <= line_fraction <= 1+1e-10:
                crossings.append(crossing)
    return crossings, {"direction_candidates": len(candidates),
                       "outside_segment": len(candidates)-len(crossings),
                       "line_fractions": [c.line_fraction for c in crossings]}

def interpolate_channel(channel, time):
    """Linear in absolute time, NaN outside coverage; never extrapolate."""
    return np.interp(time, channel.time, channel.value, left=np.nan, right=np.nan)

def split_laps(session, crossings):
    run_distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(session.xy, axis=0), axis=1))]
    # Central finite difference of horizontal travelled distance, no extra filter.
    speed = np.gradient(run_distance, session.time)
    laps = []
    for number, (start, end) in enumerate(zip(crossings[:-1], crossings[1:]), 1):
        inside = (session.time > start.time) & (session.time < end.time)
        time = np.r_[start.time, session.time[inside], end.time]
        xy = np.vstack([start.xy, session.xy[inside], end.xy])
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if distance[-1] <= 0:
            raise ValueError("Lap has no travelled distance")
        laps.append(Lap(number, time, xy, distance, np.interp(time, session.time, speed),
                        {k: interpolate_channel(v, time) for k,v in session.channels.items()}))
    return laps

def interp_distance(distance, values, target):
    """Reject decreasing axes. Keep last time at a plateau (departure).
    A stationary period is therefore retained as a jump in elapsed time,
    except at the start where elapsed time must remain zero.
    """
    distance, values, target = map(np.asarray, (distance, values, target))
    if np.any(np.diff(distance) < 0):
        raise ValueError("Distance is decreasing")
    if not np.all(np.isfinite(distance)) or len(distance) != len(values):
        raise ValueError("Invalid distance/value arrays")
    keep = np.r_[np.diff(distance) > 0, True]
    x, y = distance[keep], values[keep]
    if len(x) < 2:
        raise ValueError("Need two distinct distances")
    result = np.interp(target, x, y)
    result[target == distance[0]] = values[0]
    return result

def align_laps(laps, points=1001):
    u = np.linspace(0, 1, points)
    aligned = {}
    for lap in laps:
        target = u * lap.distance[-1]
        aligned[lap.number] = {
            "elapsed": interp_distance(lap.distance, lap.time-lap.time[0], target),
            "speed": interp_distance(lap.distance, lap.speed, target),
            "xy": np.column_stack([interp_distance(lap.distance, lap.xy[:,j], target) for j in range(2)]),
            **{k: interp_distance(lap.distance, v, target) for k,v in lap.channels.items()}
        }
    fastest = min(laps, key=lambda lap: lap.duration)
    for lap in laps:
        delta = aligned[lap.number]["elapsed"] - aligned[fastest.number]["elapsed"]
        aligned[lap.number]["delta"] = delta
        if abs(delta[0]) > 1e-10 or abs(delta[-1]-(lap.duration-fastest.duration)) > 1e-8:
            raise AssertionError("Time-gap endpoints do not match lap times")
    return u, aligned, fastest

STANDARD_GRAVITY_MPS2 = 9.80665
GPS_ACCEL_WINDOW_SAMPLES = 11
GPS_ACCEL_MIN_SPEED_MPS = 2.0

def path_acceleration_g(velocity, acceleration, min_speed_mps=GPS_ACCEL_MIN_SPEED_MPS):
    """Project horizontal acceleration along travel and to its left, in g.
    East/north coordinates form a right-handed horizontal plane: a left turn is
    positive Glat. Positive Glong means increasing horizontal speed.
    These are path axes, not measured vehicle/body axes or accelerometer loads.
    """
    speed = np.linalg.norm(velocity, axis=1)
    valid = speed >= min_speed_mps
    glat = np.full(len(speed), np.nan)
    glong = np.full(len(speed), np.nan)
    glat[valid] = cross2(velocity[valid], acceleration[valid]) / speed[valid] / STANDARD_GRAVITY_MPS2
    glong[valid] = np.sum(velocity[valid]*acceleration[valid], axis=1) / speed[valid] / STANDARD_GRAVITY_MPS2
    return glat, glong

def gps_acceleration(time, xy, window_samples=GPS_ACCEL_WINDOW_SAMPLES,
                     min_speed_mps=GPS_ACCEL_MIN_SPEED_MPS):
    """Offline GPS-derived acceleration with centred Savitzky-Golay derivatives.

    Fit a cubic independently to x(t), y(t) in an odd, symmetric sample window.
    Evaluate its first and second derivatives at the central sample. This
    smooths/differentiates in time BEFORE lap splitting/distance interpolation.
    Centring introduces no fixed time delay in the interior, but it can broaden
    events and reduce/alter peaks; it is not a causal real-time estimator.

    Reject irregular GPS sampling rather than silently assuming constant dt.
    Mask half a window at session edges and speeds below 2 m/s by default.
    NaN marks unavailable estimates, never zero acceleration.
    """
    from scipy.signal import savgol_filter
    time, xy = np.asarray(time, float), np.asarray(xy, float)
    validate_series(time, xy, "GPS acceleration input")
    if xy.shape != (len(time), 2):
        raise ValueError("GPS acceleration requires horizontal x/y pairs")
    if (not isinstance(window_samples, (int, np.integer)) or
            window_samples < 5 or window_samples % 2 != 1 or window_samples > len(time)):
        raise ValueError("Acceleration window must be odd, >=5, and fit within the session")
    if not np.isfinite(min_speed_mps) or min_speed_mps <= 0:
        raise ValueError("Minimum travel speed must be positive")
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-4, atol=1e-6):
        raise ValueError("GPS acceleration requires uniformly sampled time; handle gaps/resample explicitly")
    velocity = savgol_filter(xy, window_samples, 3, deriv=1, delta=dt, axis=0, mode="interp")
    acceleration = savgol_filter(xy, window_samples, 3, deriv=2, delta=dt, axis=0, mode="interp")
    glat, glong = path_acceleration_g(velocity, acceleration, min_speed_mps)
    half = window_samples//2
    for values in (glat, glong):
        values[:half] = np.nan
        values[-half:] = np.nan
    metadata = {
        "source": "derived from horizontal GPS position, not recorded IMU",
        "method": "centred cubic Savitzky-Golay first/second position derivatives, path-frame projection",
        "window_samples": window_samples,
        "window_span_s": (window_samples-1)*dt,
        "sample_interval_s": dt,
        "polynomial_order": 3,
        "min_speed_mps": min_speed_mps,
        "masked_session_edge_samples_each_end": half,
        "gravity_mps2": STANDARD_GRAVITY_MPS2,
        "glat_sign": "positive left of GPS travel direction",
        "glong_sign": "positive speeding up; negative slowing down",
        "limitations": "horizontal path-frame kinematic estimates; no body heading, sideslip, banking, roll or gravity-load correction; filter-dependent peaks",
        "reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html",
    }
    return glat, glong, metadata

def add_gps_acceleration(session):
    """Attach derived channels without modifying recorded GPS or existing speed."""
    if any(key in session.channels for key in ("glat", "glong")):
        raise ValueError("Acceleration channels already exist; refusing to overwrite")
    glat, glong, metadata = gps_acceleration(session.time, session.xy)
    session.channels["glat"] = Channel(session.time, glat, "g", kind="derived")
    session.channels["glong"] = Channel(session.time, glong, "g", kind="derived")
    session.metadata["gps_acceleration"] = metadata
