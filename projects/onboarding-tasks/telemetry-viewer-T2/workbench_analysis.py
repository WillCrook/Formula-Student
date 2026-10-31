"""Geographic sections, candidate corners and reusable comparison calculations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter
from telemetry import find_crossings, interp_distance, add_gps_acceleration, split_laps

@dataclass
class Run:
    id: str
    name: str
    track_id: str
    session: object
    laps: list
    notices: list

def prepare_run(run_id,name,track_id,session):
    notices=list(session.metadata.get("notices",[]))
    if "glat" not in session.channels:
        try:
            add_gps_acceleration(session)
        except ValueError as error:
            notices.append("GPS acceleration unavailable: "+str(error))
    crossings,detection=find_crossings(session.time,session.xy,session.line)
    if len(crossings)<2:
        raise ValueError("This finish gate gives fewer than two forward crossings. Move, rotate or reverse it.")
    laps=split_laps(session,crossings)
    # Keep missing supporting-channel spans visible as gaps in plots too.
    for key,channel in session.channels.items():
        if channel.kind!="recorded":
            continue
        gaps=np.diff(channel.time)>max(.5,3*np.median(np.diff(channel.time)))
        if not np.any(gaps):
            continue
        notices.append(f"{key}: long sample gaps remain unavailable.")
        for lap in laps:
            bracket=np.searchsorted(channel.time,lap.time,side="right")-1
            valid=(bracket>=0)&(bracket<len(channel.time)-1)
            clipped=np.clip(bracket,0,len(channel.time)-2)
            missing=valid&gaps[clipped]&(lap.time>channel.time[clipped])&(lap.time<channel.time[clipped+1])
            lap.channels[key][missing]=np.nan
    if len(laps)>500:
        raise ValueError("More than 500 laps detected; check the finish gate and GPS data")
    session.metadata["crossing_detection"]=detection
    return Run(run_id,name,track_id,session,laps,notices)

def lap_path(lap,points=1001):
    u=np.linspace(0,1,points)
    xy=np.column_stack([interp_distance(lap.distance,lap.xy[:,j],u*lap.distance[-1]) for j in range(2)])
    return u,xy

def suggest_corners(path,length):
    """Editable proposals from sustained signed curvature, not official turns."""
    path=np.asarray(path,float)
    ds=length/(len(path)-1)
    window=max(7,int(round(10/ds))|1)
    window=min(window,(len(path)//2)*2-1)
    dx=savgol_filter(path,window,3,deriv=1,delta=ds,axis=0,mode="interp")
    ddx=savgol_filter(path,window,3,deriv=2,delta=ds,axis=0,mode="interp")
    denom=np.linalg.norm(dx,axis=1)**3
    curvature=np.divide(dx[:,0]*ddx[:,1]-dx[:,1]*ddx[:,0],denom,
                        out=np.zeros(len(path)),where=denom>.05)
    # Hysteresis: seed at |k|>.025 1/m, extend to .010; bridge short same-sign gaps.
    candidates=[]
    for sign in [1,-1]:
        active=sign*curvature>.010
        indices=np.flatnonzero(active)
        groups=[]
        for index in indices:
            if groups and (index-groups[-1][-1])*ds<=4:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        for group in groups:
            start,end=group[0],group[-1]
            segment=sign*curvature[start:end+1]
            arc=(end-start)*ds
            angle=float(np.trapezoid(np.maximum(segment,0),dx=ds))
            if segment.max()<.025 or arc<4 or angle<np.deg2rad(15):
                continue
            peak=start+int(np.argmax(segment))
            candidates.append({"start":max(0,(start-2)/(len(path)-1)),
                               "end":min(1,(end+2)/(len(path)-1)),
                               "position":peak/(len(path)-1),
                               "direction":"L" if sign==1 else "R",
                               "heading_change_deg":float(np.rad2deg(angle)),
                               "confirmed":False})
    candidates.sort(key=lambda c:c["position"])
    for i,corner in enumerate(candidates):
        corner["id"]=f"corner-{i+1}"
        corner["name"]=f"T{i+1}"
    return candidates,{"method":"signed curvature of cubic-smoothed reference path",
                       "smoothing_span_m":(window-1)*ds,"seed_curvature_per_m":.025,
                       "extension_curvature_per_m":.010,"bridge_gap_m":4,
                       "minimum_arc_m":4,"minimum_heading_change_deg":15,
                       "warning":"candidate bends, manually editable; not official corner numbers or geometric apex measurements"}

def validate_intervals(items,partition=False):
    if not isinstance(items,list) or len(items)>100 or (partition and not items):
        raise ValueError("Provide 1-100 sectors or at most 100 corners")
    result=[]
    for i,item in enumerate(items):
        try:
            start,end=float(item["start"]),float(item["end"])
        except (TypeError,ValueError,KeyError) as error:
            raise ValueError("Enter numeric start/end positions") from error
        if not np.isfinite(start+end) or not 0<=start<end<=1 or end-start<.001:
            raise ValueError("Each range must lie within 0-100%, with start before end")
        name=str(item.get("name",f"Section {i+1}")).strip()[:40]
        if not name:
            raise ValueError("Give each section a name")
        position=float(item.get("position",(start+end)/2))
        if not np.isfinite(position) or not start<=position<=end:
            position=(start+end)/2
        result.append({"id":str(item.get("id",f"section-{i+1}"))[:64],
                       "name":name,"start":start,"end":end,"position":position,
                       "direction":str(item.get("direction",""))[:1],
                       "confirmed":bool(item.get("confirmed",True))})
    result.sort(key=lambda x:x["start"])
    if partition:
        if abs(result[0]["start"])>1e-8 or abs(result[-1]["end"]-1)>1e-8:
            raise ValueError("Sectors must cover the complete lap from 0% to 100%")
        if any(abs(a["end"]-b["start"])>1e-8 for a,b in zip(result,result[1:])):
            raise ValueError("Sectors must be contiguous, with no gaps or overlaps")
    return result

def gate_line(track,progress):
    path=np.asarray(track["path"],float)
    j=int(np.clip(round(progress*(len(path)-1)),0,len(path)-1))
    tangent=path[min(j+4,len(path)-1)]-path[max(j-4,0)]
    norm=np.linalg.norm(tangent)
    if norm<.1:
        raise ValueError("Reference path is stationary here; choose another boundary")
    direction=np.array([tangent[1],-tangent[0]])/norm
    center=np.array([np.interp(progress,np.linspace(0,1,len(path)),path[:,k]) for k in [0,1]])
    return np.vstack([center-10*direction,center+10*direction])

def boundary_time(track,run,lap,progress):
    if progress<=1e-10:
        return float(lap.time[0]),None
    if progress>=1-1e-10:
        return float(lap.time[-1]),None
    try:
        line=gate_line(track,progress)
    except ValueError as error:
        return None,str(error)
    hits,_=find_crossings(lap.time,lap.xy,line)
    nearby=[h for h in hits if abs(np.interp(h.time,lap.time,lap.progress)-progress)<.06]
    if len(nearby)!=1:
        return None,"Gate crossing is missing or ambiguous"
    hit=nearby[0]
    gap=lap.time[hit.index+1]-lap.time[hit.index]
    normal=np.median(np.diff(run.session.time))
    if gap>max(.5,3*normal):
        return None,"GPS gap crosses the timing gate"
    return hit.time,None

def interval_times(track,run,lap,start,end):
    t0,error0=boundary_time(track,run,lap,start)
    t1,error1=boundary_time(track,run,lap,end)
    if error0 or error1:
        return None,error0 or error1
    if t1<=t0:
        return None,"Timing gates crossed out of order"
    return (t0,t1),None

def logged_energy(channel,start,end):
    """Native-clock trapezoidal integral: kW*s=kJ. No extrapolation/gap bridging."""
    if channel is None or channel.unit!="kW":
        return None
    time,value=channel.time,channel.value
    if start<time[0] or end>time[-1] or end<=start:
        return None
    bracket=(time[:-1]<end)&(time[1:]>start)
    if np.any(np.diff(time)[bracket]>max(.5,3*np.median(np.diff(time)))):
        return None
    inside=(time>start)&(time<end)
    clock=np.r_[start,time[inside],end]
    samples=np.r_[np.interp(start,time,value),value[inside],np.interp(end,time,value)]
    if not np.all(np.isfinite(samples)):
        return None
    return float(np.trapezoid(samples,clock))

def lap_id(run,lap):
    return f"{run.id}:{lap.number}"

def lap_summary(run,lap):
    return {"id":lap_id(run,lap),"run_id":run.id,"run":run.name,"number":lap.number,
            "label":f"{run.name} · L{lap.number}","duration_s":lap.duration,
            "length_m":float(lap.distance[-1]),
            "energy_kj":logged_energy(run.session.channels.get("power"),lap.time[0],lap.time[-1])}

def interval_metrics(track,run,lap,start,end):
    times,error=interval_times(track,run,lap,start,end)
    result={"id":lap_id(run,lap),"run":run.name,"lap":lap.number,"status":error or "ok"}
    if times is None:
        return {**result,"time_s":None,"entry_kmh":None,"minimum_kmh":None,"exit_kmh":None,"energy_kj":None}
    t0,t1=times
    mask=(lap.time>t0)&(lap.time<t1)
    speeds=np.r_[np.interp(t0,lap.time,lap.speed),lap.speed[mask],np.interp(t1,lap.time,lap.speed)]*3.6
    return {**result,"time_s":float(t1-t0),"start_time":t0,"end_time":t1,
            "entry_kmh":float(speeds[0]),"minimum_kmh":float(speeds.min()),"exit_kmh":float(speeds[-1]),
            "energy_kj":logged_energy(run.session.channels.get("power"),t0,t1)}

def resample_lap(run,lap,start,end,track,points=701):
    """Whole-lap normalisation; focused views normalise BETWEEN physical gates."""
    metrics=interval_metrics(track,run,lap,start,end)
    q=np.linspace(0,1,points)
    if metrics["status"]=="ok":
        t0,t1=metrics["start_time"],metrics["end_time"]
        d0,d1=np.interp([t0,t1],lap.time,lap.distance)
        target=d0+q*(d1-d0)
    else:
        target=(start+q*(end-start))*lap.distance[-1]
    elapsed=interp_distance(lap.distance,lap.time-lap.time[0],target)
    result={"id":lap_id(run,lap),"label":f"{run.name} · L{lap.number}","number":lap.number,
            "u":start+q*(end-start),"elapsed":elapsed,
            "x":interp_distance(lap.distance,lap.xy[:,0],target),
            "y":interp_distance(lap.distance,lap.xy[:,1],target),
            "speed":interp_distance(lap.distance,lap.speed,target)*3.6,
            **{key:interp_distance(lap.distance,value,target) for key,value in lap.channels.items()},
            "metrics":metrics}
    return result

def sector_statistics(track,runs):
    sectors=[]
    for sector in track["sectors"]:
        samples=[]
        for run in runs:
            for lap in run.laps:
                metric=interval_metrics(track,run,lap,sector["start"],sector["end"])
                if metric["time_s"] is not None:
                    samples.append(metric)
        times=np.array([m["time_s"] for m in samples])
        best=min(samples,key=lambda x:x["time_s"]) if samples else None
        sectors.append({**sector,"count":len(times),"best_s":float(times.min()) if len(times) else None,
                        "median_s":float(np.median(times)) if len(times) else None,
                        "std_s":float(np.std(times,ddof=1)) if len(times)>1 else None,
                        "p25_s":float(np.quantile(times,.25)) if len(times) else None,
                        "p75_s":float(np.quantile(times,.75)) if len(times) else None,
                        "best_lap":best["id"] if best else None,
                        "best_entry_kmh":best["entry_kmh"] if best else None,
                        "best_exit_kmh":best["exit_kmh"] if best else None,"samples":samples})
    complete=[s for s in sectors if s["count"]]
    composite=sum(s["best_s"] for s in sectors) if len(complete)==len(sectors) else None
    best_complete=min((lap.duration for run in runs for lap in run.laps),default=None)
    return {"sectors":sectors,"composite_s":composite,"best_complete_s":best_complete,
            "scope_laps":sum(len(r.laps) for r in runs),
            "potential_s":best_complete-composite if composite is not None and best_complete is not None else None}

def compare(track,runs,selected,reference,start=0.,end=1.):
    if not np.isfinite(start+end) or not 0<=start<end<=1 or end-start<.001:
        raise ValueError("Select a range within 0-100%")
    lookup={lap_id(run,lap):(run,lap) for run in runs for lap in run.laps}
    if reference not in lookup or not selected or any(key not in lookup for key in selected):
        raise ValueError("Select valid laps and a reference on this track")
    selected=list(dict.fromkeys([reference]+selected))
    if len(selected)>64:
        raise ValueError("Compare up to 64 laps at a time")
    traces=[resample_lap(*lookup[key],start,end,track) for key in selected]
    ref=traces[0]
    reference_metric=ref["metrics"]
    for trace in traces:
        trace["delta"]=trace["elapsed"]-ref["elapsed"]
        valid=trace["metrics"]["status"]=="ok" and reference_metric["status"]=="ok"
        trace["local_delta"]=trace["delta"]-trace["delta"][0] if valid else np.full(len(trace["u"]),np.nan)
        trace["metrics"]["delta_s"]=(trace["metrics"]["time_s"]-reference_metric["time_s"]) if valid else None
        if valid:
            assert abs(trace["local_delta"][-1]-trace["metrics"]["delta_s"])<1e-7
        full_u,full_xy=lap_path(lookup[trace["id"]][1])
        trace["full_u"]=full_u
        trace["full_x"],trace["full_y"]=full_xy.T
        trace["full_elapsed"]=interp_distance(lookup[trace["id"]][1].distance,
                                               lookup[trace["id"]][1].time-lookup[trace["id"]][1].time[0],
                                               full_u*lookup[trace["id"]][1].distance[-1])
    for trace in traces:
        lap=lookup[trace["id"]][1]
        trace["full_speed"]=interp_distance(lap.distance,lap.speed,trace["full_u"]*lap.distance[-1])*3.6
    ref_full=traces[0]["full_elapsed"]
    length=track["length_m"]
    for trace in traces:
        delta=trace["full_elapsed"]-ref_full
        smooth=savgol_filter(delta,31,2)
        trace["local_rate"]=np.gradient(smooth,np.linspace(0,length,len(smooth)))*100
    descriptors={"speed":{"label":"Speed","unit":"km/h","kind":"derived"},
                 "delta":{"label":"Cumulative gap","unit":"s","kind":"derived"},
                 "local_delta":{"label":"Section gap","unit":"s","kind":"derived"}}
    for run in runs:
        for key,channel in run.session.channels.items():
            descriptors[key]={"label":{"glat":"Glat","glong":"Glong","brake":"Front brake"}.get(key,key.replace("_"," ").title()),
                              "unit":channel.unit,"kind":channel.kind}
    statistics=sector_statistics(track,runs)
    summaries=[lap_summary(run,lap) for run in runs for lap in run.laps]
    return {"track":track,"traces":traces,"reference":reference,"start":start,"end":end,
            "channels":descriptors,"statistics":statistics,"all_laps":summaries,
            "notices":[note for run in runs for note in run.notices],
            "gates":[gate_line(track,p).tolist() for p in [start,end] if 0<p<1]}
