import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import savemat
from telemetry import Channel
from importers import read_columns, build_session, gate_at_index
from workspace import Workspace, ROOT, json_safe
from workbench_analysis import prepare_run, logged_energy, suggest_corners, validate_intervals, interval_times
from viewer import create_app

TMP=ROOT/"tmp"
TMP.mkdir(exist_ok=True)

def circle_csv():
    time=np.arange(0,112,.1)
    phase=.2+.2*time
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(["time","x","y","power","throttle"])
    writer.writerows(zip(time,40*np.sin(phase),40*np.cos(phase),np.full(len(time),10.),np.full(len(time),.5)))
    return stream.getvalue().encode()

def circle_config(draft):
    return {"draft":draft,"name":"Circle test","track_name":"Test circuit","track":None,
            "mapping":{"coordinates":"xy","time":"f0:time","time_unit":"s","x":"f0:x","y":"f0:y","xy_unit":"m",
                       "channels":[{"name":"power","value":"f0:power","time":"f0:time","time_unit":"s","unit":"kW"},
                                   {"name":"throttle","value":"f0:throttle","time":"f0:time","time_unit":"s","unit":"fraction"}]},
            "gate_index":10,"gate_width":20,"gate_angle":0,"gate_flip":False}

class SectionAndEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=TMP)
        cls.workspace=Workspace(Path(cls.temp.name))
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_reference_change_and_physical_section_delta(self):
        data=self.workspace.analysis({"reference":"onboarding:11","laps":["onboarding:12"],"start":.1,"end":.25})
        self.assertEqual(data["reference"],"onboarding:11")
        for trace in data["traces"]:
            self.assertEqual(trace["metrics"]["status"],"ok")
            self.assertAlmostEqual(trace["local_delta"][0],0)
            self.assertAlmostEqual(trace["local_delta"][-1],trace["metrics"]["delta_s"])
        whole=self.workspace.analysis({"reference":"onboarding:11","laps":["onboarding:12"]})
        self.assertAlmostEqual(whole["traces"][1]["delta"][-1],-.2743087375,places=8)

    def test_consistency_and_best_sector_partition(self):
        data=self.workspace.analysis({})
        stats=data["statistics"]
        self.assertEqual(stats["scope_laps"],13)
        self.assertLessEqual(stats["composite_s"],stats["best_complete_s"])
        self.assertAlmostEqual(stats["composite_s"],sum(s["best_s"] for s in stats["sectors"]))
        run=self.workspace.load_run("onboarding")
        for lap in run.laps:
            times=[interval_times(data["track"],run,lap,s["start"],s["end"])[0] for s in data["track"]["sectors"]]
            self.assertAlmostEqual(sum(b-a for a,b in times),lap.duration)
        for sector in stats["sectors"]:
            values=[m["time_s"] for m in sector["samples"]]
            self.assertAlmostEqual(sector["std_s"],np.std(values,ddof=1))
            self.assertIn(sector["best_lap"],[lap["id"] for lap in data["all_laps"]])

    def test_native_clock_energy_and_unavailable_coverage(self):
        channel=Channel(np.array([0.,.2,.4,.6,.8,1.]),np.full(6,10.),"kW")
        self.assertAlmostEqual(logged_energy(channel,.1,.9),8.)
        ramp=Channel(channel.time,2*channel.time,"kW")
        self.assertAlmostEqual(logged_energy(ramp,.2,.8),.6)
        self.assertIsNone(logged_energy(channel,-.1,.8))
        self.assertIsNone(logged_energy(None,0,1))
        self.assertIsNone(logged_energy(Channel(np.array([0.,.1,.2,2.,2.1]),np.ones(5),"kW"),0,2))
        missing=Channel(channel.time,np.array([1,1,np.nan,1,1,1]),"kW")
        self.assertIsNone(logged_energy(missing,.1,.9))

    def test_supporting_channel_gaps_are_not_plotted_as_measurements(self):
        import copy
        session=copy.deepcopy(self.workspace.load_run("onboarding").session)
        lap=self.workspace.load_run("onboarding").laps[0]
        start=lap.time[0]+5
        power=session.channels["power"]
        keep=(power.time<start)|(power.time>start+2)
        session.channels["power"]=Channel(power.time[keep],power.value[keep],"kW")
        run=prepare_run("gapped","Gapped","onboarding",session)
        inside=(run.laps[0].time>start+.2)&(run.laps[0].time<start+1.8)
        self.assertTrue(np.all(np.isnan(run.laps[0].channels["power"][inside])))
        self.assertIsNone(logged_energy(session.channels["power"],lap.time[0],lap.time[-1]))

    def test_corner_geometry_and_partition_validation(self):
        path=np.column_stack([np.linspace(0,100,1001),np.zeros(1001)])
        corners,_=suggest_corners(path,100)
        self.assertEqual(corners,[])
        theta=np.linspace(0,2*np.pi,1001)
        circle=20*np.column_stack([np.cos(theta),np.sin(theta)])
        corners,_=suggest_corners(circle,40*np.pi)
        self.assertTrue(corners)
        self.assertTrue(all(c["direction"]=="L" and not c["confirmed"] for c in corners))
        with self.assertRaises(ValueError):
            validate_intervals([{"name":"A","start":0,"end":.4},{"name":"B","start":.5,"end":1}],True)
        with self.assertRaises(ValueError):
            validate_intervals([{"name":"A","start":None,"end":1}],True)

class ImportAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=TMP)
        self.path=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_csv_units_and_independent_clock_offset(self):
        columns={"t":np.array([0,100,200,300]),"x":np.array([0,1000,2000,3000]),"y":np.zeros(4),
                 "pt":np.array([-1.,0,1,2]),"p":np.array([1000.,2000,3000,4000]),"th":np.array([0.,.5,1.,.5])}
        mapping={"coordinates":"xy","time":"t","time_unit":"ms","x":"x","y":"y","xy_unit":"mm",
                 "channels":[{"name":"power","value":"p","time":"pt","unit":"W","offset_s":1},
                             {"name":"throttle","value":"th","time":"t","time_unit":"ms","unit":"fraction"}]}
        session=build_session(columns,mapping)
        np.testing.assert_allclose(session.time,[0,.1,.2,.3])
        np.testing.assert_allclose(session.xy[:,0],[0,1,2,3])
        np.testing.assert_allclose(session.channels["power"].value,[1,2,3,4])
        np.testing.assert_allclose(session.channels["power"].time,[0,1,2,3])
        np.testing.assert_allclose(session.channels["throttle"].value,[0,50,100,50])

    def test_mat_vectors_and_gpx_time(self):
        file=self.path/"source.mat"
        savemat(file,{"GPStime":np.array([0,.1,.2]),"GPSlat":np.array([52.,52.,52.]),"GPSlon":np.array([-1.,-1.001,-1.002])})
        columns=read_columns(file)
        self.assertEqual(len(columns["GPStime"]),3)
        gpx=self.path/"track.gpx"
        gpx.write_text('<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg><trkpt lat="52" lon="-1"><ele>100</ele><time>2026-10-29T12:00:00Z</time></trkpt><trkpt lat="52.0001" lon="-1"><ele>101</ele><time>2026-10-29T12:00:01Z</time></trkpt></trkseg></trk></gpx>')
        columns=read_columns(gpx)
        session=build_session(columns,{"coordinates":"gps","time":"time","time_unit":"iso","latitude":"latitude","longitude":"longitude"})
        self.assertEqual(session.time[1]-session.time[0],1)
        self.assertEqual(len(columns["elevation"]),2)

    def test_invalid_clocks_and_reserved_channels(self):
        columns={"t":np.array([0.,0.,1.]),"x":np.arange(3.),"y":np.arange(3.)}
        mapping={"coordinates":"xy","time":"t","x":"x","y":"y"}
        with self.assertRaisesRegex(ValueError,"duplicate"):
            build_session(columns,mapping)
        columns["t"]=np.arange(3.)
        mapping["channels"]=[{"name":"glat","value":"x","time":"t","unit":"g"}]
        with self.assertRaisesRegex(ValueError,"reserved"):
            build_session(columns,mapping)

    def test_finish_gate_orientation(self):
        xy=np.column_stack([np.arange(20.),np.zeros(20)])
        line=gate_at_index(xy,10,width=20)
        self.assertAlmostEqual(line[0,0],10)
        self.assertGreater(line[0,1],line[1,1])
        np.testing.assert_allclose(gate_at_index(xy,10,flip=True),line[::-1])

class WorkbenchAPITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=TMP)
        self.app=create_app(Path(self.temp.name),update_report=False)
        self.client=self.app.test_client()
    def tearDown(self):
        self.temp.cleanup()

    def stage_circle(self):
        response=self.client.post("/api/import/stage",data={"files":(io.BytesIO(circle_csv()),"circle.csv")},content_type="multipart/form-data")
        self.assertEqual(response.status_code,200,response.data)
        return circle_config(response.get_json()["id"])

    def test_import_preview_commit_reload_and_cross_run(self):
        config=self.stage_circle()
        preview=self.client.post("/api/import/preview",json=config)
        self.assertEqual(preview.status_code,200,preview.data)
        self.assertGreaterEqual(preview.get_json()["complete_laps"],2)
        added=self.client.post("/api/import/commit",json=config)
        self.assertEqual(added.status_code,200,added.data)
        new=added.get_json()
        workspace=Workspace(Path(self.temp.name))
        run=workspace.load_run(new["run"])
        self.assertEqual(run.track_id,new["track"])
        self.assertGreaterEqual(len(run.laps),2)
        for lap in run.laps:
            self.assertAlmostEqual(run.session.channels["throttle"].value.mean(),50.)
        second=self.stage_circle()
        second["track"]=new["track"];second["name"]="Circle repeat";second["gate_index"]=None
        added2=self.client.post("/api/import/commit",json=second)
        self.assertEqual(added2.status_code,200,added2.data)
        payload={"track":new["track"],"reference":new["run"]+":1","laps":[added2.get_json()["run"]+":1"],"start":.1,"end":.3}
        comparison=self.client.post("/api/analysis",json=payload)
        self.assertEqual(comparison.status_code,200,comparison.data)
        data=comparison.get_json()
        self.assertEqual(len(data["traces"]),2)
        for trace in data["traces"]:
            metric=trace["metrics"]
            self.assertEqual(metric["status"],"ok")
            self.assertAlmostEqual(metric["energy_kj"],10*metric["time_s"],places=8)

    def test_track_edits_saved_view_and_exports(self):
        overview=self.client.get("/api/workspace").get_json()
        self.assertEqual(len(overview["runs"][0]["laps"]),13)
        data=self.client.post("/api/analysis",json={}).get_json()
        corners=data["track"]["corners"]
        corners[0]["name"]="Opening bend";corners[0]["confirmed"]=True
        result=self.client.put("/api/tracks/onboarding",json={"corners":corners})
        self.assertEqual(result.status_code,200,result.data)
        saved=self.client.post("/api/views",json={"name":"Opening test","selection":{"track":"onboarding","reference":"onboarding:11","laps":["onboarding:12"],"start":.1,"end":.25}})
        self.assertEqual(saved.status_code,200,saved.data)
        reopened=Workspace(Path(self.temp.name))
        self.assertEqual(reopened.track("onboarding")["corners"][0]["name"],"Opening bend")
        self.assertEqual(reopened.catalog["views"][0]["name"],"Opening test")
        for format_name in ["csv","json","pdf"]:
            response=self.client.post("/api/export/"+format_name,json=saved.get_json()["selection"])
            self.assertEqual(response.status_code,200,response.data[:200])
            self.assertGreater(len(response.data),1000)
            if format_name=="csv":
                self.assertIn(b"section_gap_s",response.data[:250])
            if format_name=="pdf":
                self.assertTrue(response.data.startswith(b"%PDF"))
            if format_name=="json":
                parsed=response.get_json()
                self.assertEqual(parsed["selection"]["reference"],"onboarding:11")

    def test_invalid_sector_edit_does_not_partially_change_corners(self):
        original=self.client.post("/api/analysis",json={}).get_json()["track"]["corners"]
        changed=json.loads(json.dumps(original))
        changed[0]["name"]="Must not persist"
        response=self.client.put("/api/tracks/onboarding",json={"corners":changed,"sectors":[{"name":"Invalid","start":.1,"end":1}]})
        self.assertEqual(response.status_code,400)
        after=self.client.post("/api/analysis",json={}).get_json()["track"]["corners"]
        self.assertEqual(after,original)

    def test_bad_imports_and_missing_selection(self):
        self.assertEqual(self.client.post("/api/import/stage",data={}).status_code,400)
        self.assertEqual(self.client.post("/api/analysis",json={"start":.7,"end":.2}).status_code,400)
        self.assertEqual(self.client.post("/api/analysis",json={"reference":"unknown:1"}).status_code,400)
        self.assertEqual(self.client.put("/api/tracks/onboarding",json={"sectors":[{"name":"Bad","start":.2,"end":1}]}).status_code,400)
        self.assertEqual(self.client.post("/api/import/stage",data={"files":(io.BytesIO(b"bad"),"script.py")},content_type="multipart/form-data").status_code,400)
        self.assertEqual(self.client.post("/api/import/preview",json={"draft":"../../escape"}).status_code,400)

if __name__=="__main__":
    unittest.main()
