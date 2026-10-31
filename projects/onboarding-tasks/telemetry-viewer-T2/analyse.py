"""Run the reproducible onboarding analysis; exports stay beside this script."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "tmp/cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from telemetry import (load_onboarding, find_crossings, split_laps, align_laps,
                       interp_distance, add_gps_acceleration, gps_acceleration,
                       path_acceleration_g)
OUT = ROOT / "output"
DATA = ROOT / "data/original/VD Onboarding 22-10 MATLAB"
SECTIONS = [("A", 0.0, 0.3), ("B", 0.3, 0.6), ("C", 0.6, 1.0)]
COLORS = {12: "#007f72", 11: "#bd4c00", 13: "#6945a0"}

def load_analysis(data=DATA):
    session = load_onboarding(data)
    add_gps_acceleration(session)
    crossings, detection = find_crossings(session.time, session.xy, session.line)
    if len(crossings) < 2:
        raise ValueError("Fewer than two accepted crossings")
    laps = split_laps(session, crossings)
    u, aligned, fastest = align_laps(laps)
    return session, crossings, detection, laps, u, aligned, fastest

def section_rows(laps, u, aligned, fastest):
    rows = []
    for lap in laps:
        values = aligned[lap.number]
        for name, a, b in SECTIONS:
            mask = (u >= a) & (u <= b)
            speed = values["speed"]*3.6
            rows.append({
                "lap": lap.number, "section": name, "start_progress": a, "end_progress": b,
                "delta_change_s": float(np.interp(b,u,values["delta"])-np.interp(a,u,values["delta"])),
                "entry_kmh": float(np.interp(a,u,speed)), "minimum_kmh": float(speed[mask].min()),
                "exit_kmh": float(np.interp(b,u,speed)),
                "mean_throttle_pct": float(np.mean(values["throttle"][mask])) if "throttle" in values else None,
                "max_brake_sensor": float(np.max(values["brake"][mask])) if "brake" in values else None,
            })
    return rows

def gate_check(laps, u, aligned, fastest):
    """Two 20 m transverse physical gates at reference progress 30%, 60%.
    Use only nearby-progress crossings to disambiguate neighbouring track.
    These are a sensitivity check, not official timing sectors.
    """
    ref = aligned[fastest.number]
    times = {lap.number: [float(lap.time[0])] for lap in laps}
    gate_lines, diagnostics = [], []
    for p in [0.3, 0.6]:
        j = int(np.argmin(abs(u-p)))
        center = ref["xy"][j]
        tangent = ref["xy"][j+2]-ref["xy"][j-2]
        tangent /= np.linalg.norm(tangent)
        perpendicular = np.array([tangent[1], -tangent[0]])
        line = np.vstack([center-10*perpendicular, center+10*perpendicular])
        gate_lines.append(line.tolist())
        for lap in laps:
            hits, _ = find_crossings(lap.time, lap.xy, line)
            near = [h for h in hits if abs(np.interp(h.time,lap.time,lap.progress)-p)<0.05]
            if len(near) != 1:
                raise ValueError(f"Ambiguous gate {p} on lap {lap.number}: {len(near)}")
            hit = near[0]
            times[lap.number].append(hit.time)
            diagnostics.append({"lap":lap.number, "gate_progress":p,
                                "actual_progress":float(np.interp(hit.time,lap.time,lap.progress)),
                                "line_fraction":hit.line_fraction})
    for lap in laps:
        times[lap.number].append(float(lap.time[-1]))
    reference = np.diff(times[fastest.number])
    return {"lines":gate_lines, "diagnostics":diagnostics,
            "section_delta_s":{str(n):(np.diff(t)-reference).tolist() for n,t in times.items()}}

def save_csv(path, rows):
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def decorate(axes, distance=True):
    for ax in np.atleast_1d(axes):
        ax.grid(alpha=0.22)
        ax.spines[["top","right"]].set_visible(False)
        if distance:
            for name,a,b in SECTIONS:
                ax.axvspan(a*100,b*100,color={"A":"#178a7c","B":"#ac7800","C":"#6945a0"}[name],alpha=0.045)
            ax.set_xlim(0,100)
    np.atleast_1d(axes)[-1].set_xlabel("Normalised lap distance (%) - approximate geographic alignment" if distance else "Elapsed lap time (s)")

def finish(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png",dpi=190,facecolor="white")
    fig.savefig(OUT / "figures" / f"{name}.svg",facecolor="white")
    plt.close(fig)

def overview(laps,u,aligned,fastest):
    fig,axes=plt.subplots(2,1,figsize=(10,4.8),sharex=True,layout="constrained")
    palette=plt.get_cmap("tab20")
    for lap in laps:
        v=aligned[lap.number]
        axes[0].plot(u*100,v["speed"]*3.6,label=f"L{lap.number}",color=palette((lap.number-1)%20),
                     lw=2 if lap.number==fastest.number else 0.9)
        axes[1].plot(u*100,v["delta"],color=palette((lap.number-1)%20),
                     lw=2 if lap.number==fastest.number else 0.9)
    axes[0].set_ylabel("GPS speed (km/h)")
    axes[1].set_ylabel(f"Time behind L{fastest.number} (s)")
    axes[0].legend(ncol=7,fontsize=8,loc="upper center")
    axes[0].set_title("Every complete lap: speed and cumulative time gap",loc="left")
    axes[1].axhline(0,color="black",lw=.7)
    decorate(axes)
    finish(fig,"all_laps")

def detail_figure(laps,u,aligned,fastest,comparison,time_domain=False):
    selected=[fastest.number,comparison]
    channels=["speed","delta","throttle","brake"]
    labels=["GPS speed (km/h)","Time gap (s)","Throttle (%)","Front brake\n(unit unknown)"]
    if time_domain:
        channels=["speed","throttle","brake"]
        labels=["GPS speed (km/h)","Throttle (%)","Front brake\n(unit unknown)"]
    channels_labels=[(c,l) for c,l in zip(channels,labels) if c in aligned[fastest.number]]
    fig,axes=plt.subplots(len(channels_labels),1,figsize=(10,5.0),sharex=True,layout="constrained")
    for n in selected:
        for ax,(channel,label) in zip(axes,channels_labels):
            x=aligned[n]["elapsed"] if time_domain else u*100
            ax.plot(x,aligned[n][channel]*(3.6 if channel=="speed" else 1),
                    label=f"Lap {n}",color=COLORS.get(n,"#bd4c00"),lw=1.3)
            ax.set_ylabel(label)
    axes[0].legend(loc="upper right",ncol=2)
    axes[0].set_title(f"Lap {fastest.number} vs lap {comparison}",loc="left")
    if not time_domain:
        axes[1].axhline(0,color="black",lw=.7)
        for name,a,b in SECTIONS:
            axes[0].text((a+b)*50,1.01,name,ha="center",transform=axes[0].get_xaxis_transform())
    decorate(axes,not time_domain)
    finish(fig,f"detail_lap_{comparison}" + ("_time" if time_domain else ""))

def track_figure(session,laps,u,aligned,fastest,gates):
    fig,(ax,zoom)=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={"width_ratios":[2,1]},layout="constrained")
    for lap in laps:
        ax.plot(*lap.xy.T,color="#aebfc3",lw=.5,alpha=.5)
        zoom.plot(*lap.xy.T,color="#aebfc3",lw=.8,alpha=.7)
    ref=aligned[fastest.number]["xy"]
    for name,a,b in SECTIONS:
        mask=(u>=a)&(u<=b)
        ax.plot(*ref[mask].T,lw=2,label=f"{name}: {a:.0%}-{b:.0%}")
        mid=ref[int((a+b)/2*(len(u)-1))]
        ax.annotate(name,mid,xytext=(8,5),textcoords="offset points",weight="bold",fontsize=12)
    for line in gates["lines"]:
        ax.plot(*np.array(line).T,"k--",lw=1)
    for panel in [ax,zoom]:
        panel.plot(*session.line.T,color="#ca263b",lw=2,label="Start/finish")
        panel.set_aspect("equal")
        panel.set_xlabel("East of local origin (m)")
        panel.set_ylabel("North (m)")
        panel.grid(alpha=.2)
    p0,p1=ref[5],ref[30]
    ax.annotate("",p1,xytext=p0,arrowprops={"arrowstyle":"->","lw":2,"color":"black"})
    ax.set_title("Complete-lap GPS paths and analysis sections",loc="left")
    ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.03,1.0))
    zoom.set_xlim(-12,12); zoom.set_ylim(-12,12)
    zoom.set_title("Start/finish detail",loc="left")
    finish(fig,"track_map")

def diagnostics_figure(laps,u,aligned,fastest):
    fig,axes=plt.subplots(2,1,figsize=(10,5),layout="constrained")
    reference=aligned[fastest.number]["xy"]
    for lap in laps:
        separation=np.linalg.norm(aligned[lap.number]["xy"]-reference,axis=1)
        axes[0].plot(u*100,separation,label=f"L{lap.number}",lw=.9)
        axes[1].plot(lap.progress*100,lap.speed*3.6,lw=.8)
    axes[0].set_ylabel("Position separation (m)")
    axes[0].set_title("Alignment diagnostic: separation from fastest at equal normalised distance",loc="left")
    axes[0].legend(ncol=7,fontsize=8)
    axes[1].set_ylabel("GPS speed (km/h)")
    decorate(axes)
    finish(fig,"alignment_check")


def acceleration_figures(session,laps,u,aligned,fastest,second):
    """Export derived-G overlays and show dependence on derivative smoothing."""
    fig,axes=plt.subplots(2,1,figsize=(10,5.4),sharex=True,layout="constrained")
    palette=plt.get_cmap("tab20")
    for lap in laps:
        for ax,key in zip(axes,["glat","glong"]):
            ax.plot(u*100,aligned[lap.number][key],label=f"L{lap.number}",
                    color=palette((lap.number-1)%20),lw=1.8 if lap.number==fastest.number else .8)
    axes[0].set_title("Every complete lap: GPS-derived path-relative acceleration (11-sample cubic fit)",loc="left",fontsize=10)
    axes[0].legend(ncol=7,fontsize=8,loc="upper center")
    for ax,label in zip(axes,["Glat estimate (g), +left","Glong estimate (g), +speeding up"]):
        ax.set_ylabel(label)
        ax.axhline(0,color="black",lw=.6)
    decorate(axes)
    finish(fig,"acceleration_laps")

    # Compare baseline with shorter/longer windows; do not fit a filter to
    # expected peak g or use these estimates for the lap-time calculation.
    estimates={}
    complete=(session.time>=laps[0].time[0]) & (session.time<=laps[-1].time[-1])
    for window in [7,11,15]:
        lat,long,method=gps_acceleration(session.time,session.xy,window_samples=window)
        estimates[window]={"glat":lat,"glong":long}
    raw_v=np.gradient(session.xy,session.time,axis=0)
    raw_a=np.gradient(raw_v,session.time,axis=0)
    raw_lat,raw_long=path_acceleration_g(raw_v,raw_a)
    estimates["raw"]={"glat":raw_lat,"glong":raw_long}
    selected=[next(l for l in laps if l.number==number) for number in [fastest.number,second]]
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True,layout="constrained")
    for col,lap in enumerate(selected):
        for row,key in enumerate(["glat","glong"]):
            ax=axes[row,col]
            for window,color in [("raw","#c4c4c4"),(7,"#c87a22"),(11,"#007f72"),(15,"#6945a0")]:
                values=np.interp(lap.time,session.time,estimates[window][key])
                values=interp_distance(lap.distance,values,u*lap.distance[-1])
                ax.plot(u*100,values,color=color,lw=.7 if window=="raw" else 1.1,
                        label="Unfiltered finite differences" if window=="raw" else f"{window} samples")
            ax.set_ylabel(("Glat" if key=="glat" else "Glong")+" estimate (g)")
            ax.set_xlim(0,100)
            ax.grid(alpha=.2)
            ax.axhline(0,color="black",lw=.6)
            if row==0:
                ax.set_title(f"Lap {lap.number}: filter sensitivity",loc="left")
                ax.legend(fontsize=7,ncol=2)
            else:
                ax.set_xlabel("Normalised lap distance (%)")
    finish(fig,"acceleration_filter_check")
    stats={"scope":"all complete laps, evaluated on original GPS timestamps",
           "warning":"sample/filter-dependent estimates, not validated grip or measured body accelerations",
           "windows":{}}
    for window in [7,11,15]:
        stats["windows"][str(window)]={}
        for key in ["glat","glong"]:
            value=estimates[window][key][complete]
            difference=np.abs(value-estimates[11][key][complete])
            stats["windows"][str(window)][key]={
                "finite_samples":int(np.isfinite(value).sum()),
                "p95_abs_g":float(np.nanquantile(np.abs(value),.95)),
                "max_abs_g":float(np.nanmax(np.abs(value))),
                "p95_abs_difference_from_11_g":float(np.nanquantile(difference,.95)),
                "max_abs_difference_from_11_g":float(np.nanmax(difference))}
    return stats

def build_report(summary, rows, u, aligned, fastest, second, extra):
    # Generated first draft; edit output/report.md and render with report.py.
    s=summary
    def row(n,name):
        return next(r for r in rows if r["lap"]==n and r["section"]==name)
    a,b,c=[row(second,k)["delta_change_s"] for k in "ABC"]
    ga,gb,gc=s["gate_check"]["section_delta_s"][str(second)]
    max_delta=float(aligned[second]["delta"].max())
    max_p=float(u[np.argmax(aligned[second]["delta"])])*100
    ex=s["gate_check"]["section_delta_s"][str(extra)]
    rfast=[row(fastest,k) for k in "ABC"]
    rsecond=[row(second,k) for k in "ABC"]
    speeds="\n".join(f"| {k} | {rf['entry_kmh']:.1f}/{rf['minimum_kmh']:.1f}/{rf['exit_kmh']:.1f} | {rs['entry_kmh']:.1f}/{rs['minimum_kmh']:.1f}/{rs['exit_kmh']:.1f} |" for k,rf,rs in zip("ABC",rfast,rsecond))
    table="\n".join(f"| {r['lap']} | {r['lap_time_s']:.3f} | {r['gap_to_fastest_s']:.3f} | {r['length_m']:.2f} |" for r in s["laps"])
    gate_table="\n".join(f"| {k} | {row(second,k)['delta_change_s']:+.3f} | {s['gate_check']['section_delta_s'][str(second)][i]:+.3f} | {s['gate_check']['section_delta_s'][str(extra)][i]:+.3f} |" for i,k in enumerate("ABC"))
    text=f"""# TBRe VDG - Task 2 telemetry analysis
Draft for review | FSUK 2025 endurance data | 29 October 2026

## Aim and method
Compare all complete laps and identify where the fastest lap gains or loses time. This Python implementation retains the supplied MATLAB start/finish coordinates, direction test and linear crossing-time interpolation. It adds a finite-segment check: all {s['crossings']} accepted crossings lie within the segment, giving {len(s['laps'])} complete laps. Partial data before/after these crossings is excluded.

The GPS has {s['channels']['GPS']['samples']:,} positions at {s['channels']['GPS']['median_dt_s']:.3f} s intervals. A WGS84 local tangent-plane approximation gives horizontal metres; summed sample-to-sample distances give lap distance. Speed is a central finite difference of travelled distance, with no extra smoothing. Altitude is excluded. Throttle ({s['channels']['throttle']['samples']:,} samples, %) and front brake ({s['channels']['brake']['samples']:,}, unit unspecified) are linearly interpolated using their own logger timestamps, without extrapolation. Shared clock timing is assumed, not independently verified.

The viewer/CSV also include GPS-derived Glat/Glong (g): centred 11-sample cubic position derivatives over 1.0 s, projected left/along travel. These path-relative estimates are not IMU measurements; filtering changes peaks. See acceleration_laps.png and acceleration_filter_check.png. Original speed, distance and timing calculations are unchanged.

## Every complete lap
| Lap | Time (s) | Gap (s) | GPS length (m) |
| --- | --- | --- | --- |
{table}

Lap {fastest} is fastest; lap {second} is second, {s['laps'][second-1]['gap_to_fastest_s']:.3f} s behind. Three decimals show the calculation, not millisecond measurement accuracy. Lap lengths span {s['length_range_m'][0]:.2f}-{s['length_range_m'][1]:.2f} m.
![Complete lap paths, analysis sections and start/finish zoom](figures/track_map.png)

<!-- pagebreak -->
# All-lap comparison and the closest rival
![Speed and cumulative time gap for all complete laps](figures/all_laps.png)

Elapsed time is interpolated onto 1,001 points of normalised lap distance (0-100%). Delta = comparison elapsed time - fastest elapsed time: positive means behind; rising means losing time locally. Every trace starts at zero and ends at its measured lap-time gap. This is a comparison baseline, not exact geographic alignment.

![Shared-axis speed, delta, throttle and front brake for fastest and second fastest](figures/detail_lap_{second}.png)

Lap {second} loses {a:.3f} s to lap {fastest} in A (0-30%), changes by {b:+.3f} s in B (30-60%), then recovers {-c:.3f} s in C (60-100%). Its largest distance-aligned deficit is {max_delta:.3f} s at {max_p:.1f}%. The fastest overall lap is therefore not fastest throughout. Physical section gates give {ga:+.3f}, {gb:+.3f}, {gc:+.3f} s. The opening loss and final recovery persist; B changes sign and does not support a firm local claim.

<!-- pagebreak -->
# Section checks, another comparison and limits
![Fastest against the next fastest complete lap, showing a different local trade-off](figures/detail_lap_{extra}.png)

Lap {extra} is the next fastest after lap {second}, {s['laps'][extra-1]['gap_to_fastest_s']:.3f} s behind overall. The physical gates give section differences of {ex[0]:+.3f}, {ex[1]:+.3f}, {ex[2]:+.3f} s. Negative values mean this lap is quicker than lap {fastest} over that section.

## Three broad sections
A/B/C are analysis sections marked on the map, not official sectors. The two internal physical gates are 20 m lines normal to lap {fastest}'s path at 30% and 60%. Each lap has one accepted nearby crossing per gate. They compare the same transverse locations rather than assuming equal progress.
| Section | L{second} normalised (s) | L{second} gates (s) | L{extra} gates (s) |
| --- | --- | --- | --- |
{gate_table}

Entry/minimum/exit GPS speed (km/h) at normalised section boundaries:
| Section | L{fastest} entry/min/exit | L{second} entry/min/exit |
| --- | --- | --- |
{speeds}

## Interpretation and limitations
Speed and pedal traces provide context, not proof of cause. Traffic, grip, vehicle condition and driver intent are unknown; brake units/calibration are unspecified. Power ({s['channels']['power']['samples']:,} samples, kW per supplied comments) is available but unused.

The worst-lap 95th-percentile normalised-position separation is {s['alignment_worst_p95_m']:.2f} m (see alignment_check.png). This combines real line variation and alignment error, not GPS accuracy. GPS sampling cuts corners and noise can inflate distance. Accuracy and clock synchronisation specifications are unavailable; small local differences remain uncertain.

Next step: check physical gates at individual corners and validate GPS speed/channel synchronisation against independent speed/timing. The local Flask viewer selects laps and distance/time plots; future sources can reuse the Session/Lap calculations.

Sources: supplied Extension_Task_Base_Script.m (sections 5-10), Onboarding_Live_Example.mlx, onboarding slides (22 October 2026), and four MAT files. Generated with AI assistance; review the calculations and interpretation before submission.
"""
    (OUT/"report.md").write_text(text)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA,help="MAT data directory")
    args=parser.parse_args()
    (OUT/"figures").mkdir(parents=True,exist_ok=True)
    session,crossings,detection,laps,u,aligned,fastest=load_analysis(args.data)
    ordered=sorted(laps,key=lambda lap:lap.duration)
    second,extra=ordered[1].number,ordered[2].number
    table=[{"lap":l.number,"lap_time_s":l.duration,"gap_to_fastest_s":l.duration-fastest.duration,
            "length_m":float(l.distance[-1]),"start_logger_s":float(l.time[0]),
            "end_logger_s":float(l.time[-1])} for l in laps]
    rows=section_rows(laps,u,aligned,fastest)
    acceleration_diagnostics=acceleration_figures(session,laps,u,aligned,fastest,second)
    gates=gate_check(laps,u,aligned,fastest)
    channel_audit={}
    for name,t,v in [("GPS",session.time,session.xy)]+[(k,c.time,c.value) for k,c in session.channels.items()]:
        channel_audit[name]={"samples":len(t),"start_s":float(t[0]),"end_s":float(t[-1]),
                             "median_dt_s":float(np.median(np.diff(t))),
                             "max_dt_s":float(np.max(np.diff(t))),
                             "kind":"recorded" if name=="GPS" else session.channels[name].kind,
                             "unit":"local horizontal metres" if name=="GPS" else session.channels[name].unit,
                             "finite_samples":int(np.sum(np.all(np.isfinite(v),axis=1) if np.ndim(v)>1 else np.isfinite(v)))}
    alignment={str(l.number):{"p95_separation_m":float(np.quantile(np.linalg.norm(aligned[l.number]["xy"]-aligned[fastest.number]["xy"],axis=1),.95)),
                              "max_separation_m":float(np.max(np.linalg.norm(aligned[l.number]["xy"]-aligned[fastest.number]["xy"],axis=1))),
                              "repeated_distance_samples":int(np.sum(np.diff(l.distance)==0))}
               for l in laps}
    summary={"source":session.metadata,"channels":channel_audit,
             "crossings":len(crossings),"crossing_detection":detection,
             "excluded_before_first_s":float(crossings[0].time-session.time[0]),
             "excluded_after_last_s":float(session.time[-1]-crossings[-1].time),
             "fastest_lap":fastest.number,"second_lap":second,"extra_comparison":extra,"laps":table,
             "length_range_m":[min(r["length_m"] for r in table),max(r["length_m"] for r in table)],
             "alignment":alignment,"alignment_worst_p95_m":max(v["p95_separation_m"] for v in alignment.values()),
             "gate_check":gates,"section_metrics":rows,
             "input_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data.iterdir()) if p.is_file()},
             "endpoint_check":"PASS for every complete lap",
             "acceleration_diagnostics":acceleration_diagnostics}
    save_csv(OUT/"lap_summary.csv",table)
    save_csv(OUT/"section_metrics.csv",rows)
    (OUT/"audit.json").write_text(json.dumps(summary,indent=2))
    export=[]
    for lap in laps:
        for j,p in enumerate(u):
            v=aligned[lap.number]
            export.append({"lap":lap.number,"normalised_distance":float(p),
                           "equivalent_fastest_distance_m":float(p*fastest.distance[-1]),
                           "elapsed_s":float(v["elapsed"][j]),"delta_s":float(v["delta"][j]),
                           "speed_kmh":float(v["speed"][j]*3.6),
                           **{f"{k}_g" if k in ("glat","glong") else k:float(v[k][j]) for k in session.channels}})
    save_csv(OUT/"aligned_telemetry.csv",export)
    plt.rcParams.update({"font.size":9,"axes.titlesize":11})
    overview(laps,u,aligned,fastest)
    track_figure(session,laps,u,aligned,fastest,gates)
    diagnostics_figure(laps,u,aligned,fastest)
    for n in [second,extra]:
        detail_figure(laps,u,aligned,fastest,n)
        detail_figure(laps,u,aligned,fastest,n,True)
    build_report(summary,rows,u,aligned,fastest.number,second,extra)
    from report import render_report
    render_report()
    print(json.dumps({"crossings":len(crossings),"complete_laps":len(laps),
                      "fastest":fastest.number,"second":second,"gap_s":ordered[1].duration-fastest.duration,
                      "gate_section_deltas":gates["section_delta_s"],
                      "alignment_worst_p95_m":summary["alignment_worst_p95_m"]},indent=2))

if __name__=="__main__":
    main()
