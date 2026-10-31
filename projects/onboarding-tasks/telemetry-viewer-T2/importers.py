"""File adapters for the local workbench. Raw uploads are retained unchanged."""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import loadmat
from telemetry import Session, Channel, local_xy, validate_series

MAX_SAMPLES = 200_000
MAX_COLUMNS = 64
RESERVED = {"speed","elapsed","delta","local_delta","xy","distance","progress","glat","glong"}

def read_columns(path):
    path=Path(path)
    if path.suffix.lower()==".mat":
        try:
            mat=loadmat(path,squeeze_me=True)
        except (ValueError,NotImplementedError,OSError) as error:
            raise ValueError("Use numeric MATLAB v5/v7 files (v7.3/HDF5 is not supported).") from error
        columns={}
        for key,value in mat.items():
            if key.startswith("_"):
                continue
            array=np.asarray(value)
            if array.ndim==1 and array.dtype.kind in "fiu":
                if len(array)>MAX_SAMPLES:
                    raise ValueError(f"{path.name}: at most {MAX_SAMPLES:,} samples per channel")
                columns[key]=array.astype(float)
        if not columns:
            raise ValueError(f"{path.name}: no numeric vector channels found")
        return columns
    if path.suffix.lower()==".gpx":
        try:
            root=ET.parse(path).getroot()
        except ET.ParseError as error:
            raise ValueError("GPX file is not valid XML") from error
        points=[p for p in root.iter() if p.tag.rsplit("}",1)[-1]=="trkpt"]
        if len(points)<2 or len(points)>MAX_SAMPLES:
            raise ValueError("GPX needs 2-200,000 timed track points")
        columns={"latitude":[],"longitude":[],"time":[]}
        elevation=[]
        for point in points:
            children={c.tag.rsplit("}",1)[-1]:c.text for c in point}
            if not children.get("time"):
                raise ValueError("GPX track points need timestamps")
            columns["latitude"].append(point.attrib.get("lat",""))
            columns["longitude"].append(point.attrib.get("lon",""))
            columns["time"].append(children["time"])
            elevation.append(children.get("ele",""))
        if all(elevation):
            columns["elevation"]=elevation
        return {key:np.array(values) for key,values in columns.items()}
    if path.suffix.lower() not in (".csv",".tsv"):
        raise ValueError("Choose CSV, TSV, GPX or MATLAB .mat files")
    with path.open(encoding="utf-8-sig",newline="") as stream:
        sample=stream.read(8192); stream.seek(0)
        try:
            dialect=csv.Sniffer().sniff(sample,delimiters=",;\t")
        except csv.Error:
            dialect=csv.excel_tab if path.suffix.lower()==".tsv" else csv.excel
        reader=csv.DictReader(stream,dialect=dialect)
        names=reader.fieldnames
        if not names or len(names)>MAX_COLUMNS or len(names)!=len(set(names)) or any(not x.strip() for x in names):
            raise ValueError("CSV needs unique column headers (maximum 64 columns)")
        columns={name:[] for name in names}
        for row_index,row in enumerate(reader):
            if row_index>=MAX_SAMPLES:
                raise ValueError(f"At most {MAX_SAMPLES:,} CSV rows are supported")
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"CSV row {row_index+2} has an inconsistent number of fields")
            for name in names:
                columns[name].append(row[name].strip())
    if not columns or not len(next(iter(columns.values()))):
        raise ValueError("File contains no samples")
    return {name:np.array(values) for name,values in columns.items()}

def inspect_sources(files):
    """files: [{id, name, path}], paths supplied only by the server."""
    columns={}
    descriptions=[]
    for source in files:
        loaded=read_columns(source["path"])
        for name,values in loaded.items():
            key=f"{source['id']}:{name}"
            columns[key]=values
            descriptions.append({"key":key,"file":source["name"],"name":name,
                                 "samples":len(values),
                                 "preview":[str(v)[:80] for v in values[:3]]})
    return columns,descriptions

def numeric(columns,key,allow_nan=False):
    if key not in columns:
        raise ValueError(f"Select a valid column for {key or 'the required field'}")
    try:
        values=np.array([float(x) if str(x).strip() else np.nan for x in columns[key]],dtype=float)
    except (ValueError,TypeError) as error:
        raise ValueError(f"{key}: expected numeric values") from error
    if len(values)<2 or np.any(np.isinf(values)) or (not allow_nan and not np.all(np.isfinite(values))):
        raise ValueError(f"{key}: missing/invalid samples; GPS and timestamps must be finite")
    return values

def timestamps(columns,key,unit="s"):
    if unit=="iso":
        try:
            values=[]
            for value in columns[key]:
                parsed=datetime.fromisoformat(str(value).replace("Z","+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("ISO timestamps need a timezone offset or Z")
                values.append(parsed.timestamp())
            result=np.array(values)
        except (KeyError,ValueError,TypeError) as error:
            raise ValueError(f"{key}: ISO timestamps need a timezone offset or Z") from error
    elif unit in ("s","ms"):
        result=numeric(columns,key)/(1000 if unit=="ms" else 1)
    else:
        raise ValueError("Time units must be seconds, milliseconds or ISO 8601")
    if np.any(np.diff(result)<=0):
        raise ValueError(f"{key}: duplicate or decreasing timestamps; resolve them before importing")
    return result

def build_session(columns,mapping,track=None):
    """Map independent source clocks/units to canonical horizontal metres/seconds."""
    if not isinstance(mapping,dict):
        raise ValueError("Choose the source columns")
    time=timestamps(columns,mapping.get("time"),mapping.get("time_unit","s"))
    mode=mapping.get("coordinates","gps")
    if mode=="gps":
        lon=numeric(columns,mapping.get("longitude"))
        lat=numeric(columns,mapping.get("latitude"))
        if len(lon)!=len(time) or len(lat)!=len(time):
            raise ValueError("GPS time, latitude and longitude must have matching sample counts")
        if np.any(abs(lon)>180) or np.any(abs(lat)>90):
            raise ValueError("Latitude/longitude must be decimal degrees")
        if track is not None and track.get("origin_lon_lat") is None:
            raise ValueError("This track uses an unreferenced XY frame. Import matching XY data or create a new track.")
        origin=np.asarray(track["origin_lon_lat"] if track else [lon.mean(),lat.mean()])
        xy=local_xy(np.column_stack([lon,lat]),origin)
        origin=origin.tolist()
    elif mode=="xy":
        x=numeric(columns,mapping.get("x"))
        y=numeric(columns,mapping.get("y"))
        if mapping.get("xy_unit","m") not in ("m","mm"):
            raise ValueError("Planar coordinates must be metres or millimetres")
        scale=.001 if mapping.get("xy_unit")=="mm" else 1.
        xy=np.column_stack([x,y])*scale
        origin=track.get("origin_lon_lat") if track else None
    else:
        raise ValueError("Select latitude/longitude or planar X/Y")
    validate_series(time,xy,"GPS")
    channels={}
    specs=mapping.get("channels",[])
    if not isinstance(specs,list) or len(specs)>24:
        raise ValueError("At most 24 optional channels may be mapped")
    for spec in specs:
        key=str(spec.get("name","")).strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,29}",key) or key in RESERVED or key in channels:
            raise ValueError("Channel names must be unique letters/numbers/underscores; calculated channel names are reserved")
        values=numeric(columns,spec.get("value"),allow_nan=True)
        clock=timestamps(columns,spec.get("time"),spec.get("time_unit","s"))
        offset=float(spec.get("offset_s",0))
        if not np.isfinite(offset):
            raise ValueError("Clock offset must be a finite number of seconds")
        clock=clock+offset
        if len(values)!=len(clock):
            raise ValueError(f"{key}: value and timestamp sample counts differ")
        unit=str(spec.get("unit","unknown")).strip()[:40] or "unknown"
        if key=="throttle":
            if unit not in ("%","fraction"):
                raise ValueError("Throttle units must be % or fraction")
            if unit=="fraction":
                values=values*100
            unit="%"
        if key=="power":
            if unit not in ("kW","W"):
                raise ValueError("Power units must be kW or W for energy integration")
            if unit=="W":
                values=values/1000
            unit="kW"
        channels[key]=Channel(clock,values,unit,kind="recorded")
    dt=np.diff(time)
    notices=[]
    if np.max(dt)>max(.5,3*np.median(dt)):
        notices.append("GPS sample gaps: crossing/section timing around gaps is uncertain.")
    line=np.asarray(track["line"],float) if track else np.zeros((2,2))
    metadata={"source":"user-imported data","origin_lon_lat":origin,
              "mapping":mapping,"notices":notices,
              "units_source":"explicit import mapping; units and clock offsets are user supplied"}
    return Session(time,xy,line,channels,metadata)

def gate_at_index(xy,index,width=20.,angle_degrees=0.,flip=False):
    index=int(index)
    if not 0<=index<len(xy):
        raise ValueError("Choose a start/finish location on the track")
    width=float(width); angle=float(angle_degrees)
    if not np.isfinite(width) or not 2<=width<=100 or not np.isfinite(angle):
        raise ValueError("Gate width must be 2-100 m and angle must be finite")
    tangent=xy[min(index+3,len(xy)-1)]-xy[max(index-3,0)]
    norm=np.linalg.norm(tangent)
    if norm<.1:
        raise ValueError("Choose a moving part of the GPS trace for the finish")
    perpendicular=np.array([tangent[1],-tangent[0]])/norm
    radians=np.deg2rad(angle)
    rotation=np.array([[np.cos(radians),-np.sin(radians)],[np.sin(radians),np.cos(radians)]])
    direction=rotation@perpendicular
    if flip:
        direction=-direction
    return np.vstack([xy[index]-width/2*direction,xy[index]+width/2*direction])
