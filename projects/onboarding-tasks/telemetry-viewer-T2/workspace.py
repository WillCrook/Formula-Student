"""Project-local saved runs, track definitions, staged imports and saved views."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid
import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree
from werkzeug.utils import secure_filename
from telemetry import load_onboarding, Channel
from importers import inspect_sources, build_session, gate_at_index
from workbench_analysis import prepare_run, lap_path, suggest_corners, validate_intervals, lap_summary, compare

ROOT=Path(__file__).resolve().parent
DEFAULT_DATA=ROOT/"data/original/VD Onboarding 22-10 MATLAB"

def json_safe(value):
    if isinstance(value,np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value,np.generic):
        return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    if isinstance(value,dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    return value

def identifier(value):
    value=str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,70}",value):
        raise ValueError("Invalid workspace identifier")
    return value

class Workspace:
    def __init__(self,directory=None):
        self.directory=Path(directory) if directory else ROOT/"data/workspace"
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.cache={}
        self.drafts={}
        self.file=self.directory/"workspace.json"
        if self.file.exists():
            self.catalog=json.loads(self.file.read_text())
            if self.catalog.get("version")!=1:
                raise ValueError("Unsupported saved workspace version")
        else:
            session=load_onboarding(DEFAULT_DATA)
            gps=loadmat(DEFAULT_DATA/"Onboarding_GPS.mat",squeeze_me=True)
            session.channels["altitude"]=Channel(session.time,np.asarray(gps["GPSalt"],float).ravel(),"source units")
            run=prepare_run("onboarding","FSUK25","onboarding",session)
            fastest=min(run.laps,key=lambda lap:lap.duration)
            u,path=lap_path(fastest)
            corners,method=suggest_corners(path,float(fastest.distance[-1]))
            track={"id":"onboarding","name":"FSUK 2025 · Silverstone","path":path.tolist(),
                   "length_m":float(fastest.distance[-1]),"line":session.line.tolist(),
                   "origin_lon_lat":session.metadata["origin_lon_lat"],"corners":corners,
                   "corner_method":method,"reference_lap_id":f"onboarding:{fastest.number}",
                   "sectors":[{"id":f"sector-{i+1}","name":name,"start":a,"end":b}
                              for i,(name,a,b) in enumerate([("Sector 1",0,.3),("Sector 2",.3,.6),("Sector 3",.6,1)])]}
            self.catalog={"version":1,"tracks":{"onboarding":track},
                          "runs":{"onboarding":{"id":"onboarding","name":"FSUK25","track_id":"onboarding","builtin":True}},
                          "views":[]}
            self.cache["onboarding"]=run
            self.save()

    def save(self):
        with self.lock:
            temporary=self.file.with_suffix(".tmp")
            temporary.write_text(json.dumps(json_safe(self.catalog),indent=2,allow_nan=False))
            temporary.replace(self.file)

    def track(self,track_id):
        track_id=identifier(track_id)
        if track_id not in self.catalog["tracks"]:
            raise ValueError("Unknown track")
        return self.catalog["tracks"][track_id]

    def load_run(self,run_id):
        run_id=identifier(run_id)
        if run_id in self.cache:
            return self.cache[run_id]
        if run_id not in self.catalog["runs"]:
            raise ValueError("Unknown run")
        config=self.catalog["runs"][run_id]
        if config.get("builtin"):
            session=load_onboarding(DEFAULT_DATA)
            gps=loadmat(DEFAULT_DATA/"Onboarding_GPS.mat",squeeze_me=True)
            session.channels["altitude"]=Channel(session.time,np.asarray(gps["GPSalt"],float).ravel(),"source units")
        else:
            sources=self.sources_for(run_id,config["sources"])
            for source,original in zip(sources,config["sources"]):
                if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()!=original["sha256"]:
                    raise ValueError("A saved source file changed; re-import it before comparison")
            columns,_=inspect_sources(sources)
            session=build_session(columns,config["mapping"],self.track(config["track_id"]))
        session.line=np.asarray(self.track(config["track_id"])["line"],float)
        run=prepare_run(run_id,config["name"],config["track_id"],session)
        self.cache[run_id]=run
        return run

    def runs_for(self,track_id):
        self.track(track_id)
        return [self.load_run(key) for key,value in self.catalog["runs"].items() if value["track_id"]==track_id]

    def overview(self):
        tracks=[]
        for key,track in self.catalog["tracks"].items():
            tracks.append({"id":key,"name":track["name"],"length_m":track["length_m"],
                           "corners":len(track["corners"]),"sectors":len(track["sectors"])})
        runs=[]
        errors=[]
        for key,config in self.catalog["runs"].items():
            try:
                run=self.load_run(key)
                runs.append({"id":key,"name":run.name,"track_id":run.track_id,"notices":run.notices,
                             "laps":[lap_summary(run,lap) for lap in run.laps]})
            except (ValueError,OSError,KeyError) as error:
                errors.append({"run":config["name"],"error":str(error)})
        return json_safe({"tracks":tracks,"runs":runs,"views":self.catalog["views"],"errors":errors})

    def analysis(self,selection):
        if not isinstance(selection,dict):
            raise ValueError("Choose a valid comparison")
        track=self.track(selection.get("track","onboarding"))
        runs=self.runs_for(track["id"])
        all_laps=[lap_summary(run,lap) for run in runs for lap in run.laps]
        ordered=sorted(all_laps,key=lambda lap:lap["duration_s"])
        if not ordered:
            raise ValueError("No complete laps on this track")
        reference=selection.get("reference") or ordered[0]["id"]
        selected=selection.get("laps") or [lap["id"] for lap in ordered[:2]]
        start,end=float(selection.get("start",0)),float(selection.get("end",1))
        return compare(track,runs,selected,reference,start,end)

    def sources_for(self,draft_id,sources):
        base=self.directory/"imports"/identifier(draft_id)
        result=[]
        for source in sources:
            filename=source["file"]
            if Path(filename).name!=filename:
                raise ValueError("Invalid saved filename")
            result.append({"id":source["id"],"name":source["name"],"path":str(base/filename)})
        return result

    def stage(self,uploads):
        if not uploads or len(uploads)>12:
            raise ValueError("Choose 1-12 CSV/TSV/MAT files")
        draft_id="run_"+uuid.uuid4().hex[:14]
        directory=self.directory/"imports"/draft_id
        directory.mkdir(parents=True)
        sources=[]
        total=0
        for i,upload in enumerate(uploads):
            filename=secure_filename(upload.filename or "")
            if Path(filename).suffix.lower() not in (".csv",".tsv",".mat",".gpx"):
                raise ValueError("Choose CSV, TSV, GPX or .mat data files")
            content=upload.read(20*1024*1024+1)
            total+=len(content)
            if len(content)>20*1024*1024 or total>50*1024*1024:
                raise ValueError("Limit uploads to 20 MB per file and 50 MB in total")
            saved=f"f{i}_{filename}"
            (directory/saved).write_bytes(content)
            sources.append({"id":f"f{i}","name":filename,"file":saved,
                            "sha256":hashlib.sha256(content).hexdigest()})
        draft={"id":draft_id,"sources":sources}
        (directory/"draft.json").write_text(json.dumps(draft,indent=2))
        columns,descriptions=inspect_sources(self.sources_for(draft_id,sources))
        self.drafts[draft_id]=(draft,columns,descriptions)
        return {"id":draft_id,"columns":descriptions,"files":[s["name"] for s in sources]}

    def draft(self,draft_id):
        draft_id=identifier(draft_id)
        if draft_id not in self.drafts:
            file=self.directory/"imports"/draft_id/"draft.json"
            if not file.exists():
                raise ValueError("Import preview expired or does not exist")
            draft=json.loads(file.read_text())
            columns,descriptions=inspect_sources(self.sources_for(draft_id,draft["sources"]))
            self.drafts[draft_id]=(draft,columns,descriptions)
        return self.drafts[draft_id]

    def preview(self,config):
        draft,columns,_=self.draft(config["draft"])
        track=self.track(config["track"]) if config.get("track") else None
        session=build_session(columns,config["mapping"],track)
        index=np.unique(np.r_[np.arange(0,len(session.time),max(1,len(session.time)//1800)),len(session.time)-1]).astype(int)
        response={"x":session.xy[index,0],"y":session.xy[index,1],"indices":index,
                  "samples":len(session.time),"notices":session.metadata["notices"],
                  "line":None,"complete_laps":None,"lap_times":[]}
        if track is not None:
            session.line=np.asarray(track["line"],float)
        elif config.get("gate_index") is not None:
            session.line=gate_at_index(session.xy,config["gate_index"],config.get("gate_width",20),
                                       config.get("gate_angle",0),bool(config.get("gate_flip",False)))
        else:
            return response,None
        run=prepare_run(draft["id"],str(config.get("name","Imported run"))[:60],track["id"] if track else "new",session)
        response.update({"line":session.line,"complete_laps":len(run.laps),
                         "lap_times":[lap.duration for lap in run.laps[:8]],"notices":run.notices})
        return response,run

    def commit(self,config):
        draft,_,_=self.draft(config["draft"])
        if draft["id"] in self.catalog["runs"]:
            raise ValueError("This import has already been added")
        preview,run=self.preview(config)
        if run is None:
            raise ValueError("Choose a finish location, then preview the detected laps")
        name=str(config.get("name","Imported run")).strip()[:60] or "Imported run"
        if config.get("track"):
            track=self.track(config["track"])
            path=np.asarray(track["path"],float)
            points=np.vstack([lap.xy for lap in run.laps])
            separation=cKDTree(path).query(points)[0]
            if np.quantile(separation,.95)>25:
                raise ValueError("Coordinates do not match this track. Check the coordinate frame or create a new track.")
            track_id=track["id"]
        else:
            track_id="track_"+uuid.uuid4().hex[:12]
            fastest=min(run.laps,key=lambda lap:lap.duration)
            _,path=lap_path(fastest)
            corners,method=suggest_corners(path,float(fastest.distance[-1]))
            track={"id":track_id,"name":str(config.get("track_name",name)).strip()[:60] or name,
                   "origin_lon_lat":run.session.metadata["origin_lon_lat"],
                   "line":run.session.line.tolist(),"path":path.tolist(),"length_m":float(fastest.distance[-1]),
                   "corners":corners,"corner_method":method,"reference_lap_id":f"{run.id}:{fastest.number}",
                   "sectors":[{"id":f"sector-{i+1}","name":f"S{i+1}","start":i/3,"end":(i+1)/3} for i in range(3)]}
            self.catalog["tracks"][track_id]=track
        run.name=name
        run.track_id=track_id
        self.catalog["runs"][run.id]={"id":run.id,"name":name,"track_id":track_id,
                                     "sources":draft["sources"],"mapping":config["mapping"],"builtin":False}
        self.cache[run.id]=run
        self.save()
        return {"run":run.id,"track":track_id,"laps":len(run.laps)}

    def edit_track(self,track_id,changes):
        with self.lock:
            track=dict(self.track(track_id))
            if "corners" in changes:
                track["corners"]=validate_intervals(changes["corners"])
            if "sectors" in changes:
                track["sectors"]=validate_intervals(changes["sectors"],partition=True)
            if "name" in changes:
                track["name"]=str(changes["name"]).strip()[:60] or track["name"]
            self.catalog["tracks"][track_id]=track
            self.save()
            return track

    def save_view(self,payload):
        selection=payload.get("selection",{})
        self.analysis(selection)  # Validate IDs, track and range before persisting.
        view={"id":"view_"+uuid.uuid4().hex[:12],"name":str(payload.get("name","Saved view")).strip()[:60] or "Saved view",
              "selection":{k:selection[k] for k in ["track","reference","laps","start","end","domain","channels","local_delta","gg_color","map_color"] if k in selection}}
        with self.lock:
            self.catalog["views"].append(view)
            self.save()
        return view
