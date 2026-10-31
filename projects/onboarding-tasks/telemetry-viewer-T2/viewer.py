"""Local telemetry workbench: interactive API, imports, saved views and exports."""
from __future__ import annotations
import io
import json
from pathlib import Path
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from analyse import ROOT, OUT
from workspace import Workspace, json_safe
from exports import update_assumptions, comparison_csv, comparison_pdf, comparison_figure, RENDER_LOCK

def create_app(directory=None,update_report=True):
    app=Flask(__name__,template_folder=str(ROOT/"templates"),static_folder=str(ROOT/"static"))
    app.config["MAX_CONTENT_LENGTH"]=52*1024*1024
    app.config["UPDATE_REPORT"]=update_report
    workspace=Workspace(directory)
    app.extensions["telemetry_workspace"]=workspace

    @app.errorhandler(ValueError)
    def bad_value(error):
        return jsonify({"error":str(error)}),400

    @app.errorhandler(KeyError)
    def missing_value(error):
        return jsonify({"error":"Missing or invalid field: "+str(error)}),400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify({"error":"Choose at most 50 MB of files in total"}),413

    @app.after_request
    def fresh(response):
        if request.path.startswith("/api/") or request.path=="/":
            response.headers["Cache-Control"]="no-store"
        return response

    def body():
        value=request.get_json(silent=True)
        if not isinstance(value,dict):
            raise ValueError("Provide a JSON object")
        return value

    def selection_from_args():
        raw=request.args.get("selection")
        return json.loads(raw) if raw else {
            "track":request.args.get("track","onboarding"),
            "reference":request.args.get("reference"),
            "laps":request.args.getlist("lap"),
            "start":float(request.args.get("start",0)),
            "end":float(request.args.get("end",1)),
        }

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/workspace")
    def overview():
        return jsonify(workspace.overview())

    @app.post("/api/analysis")
    def analysis():
        return jsonify(json_safe(workspace.analysis(body())))

    @app.get("/api/analysis")
    def analysis_get():
        return jsonify(json_safe(workspace.analysis(selection_from_args())))

    @app.post("/api/import/stage")
    def stage():
        return jsonify(json_safe(workspace.stage(request.files.getlist("files"))))

    @app.post("/api/import/preview")
    def preview():
        result,_=workspace.preview(body())
        return jsonify(json_safe(result))

    @app.post("/api/import/commit")
    def commit():
        result=workspace.commit(body())
        if app.config["UPDATE_REPORT"]:
            update_assumptions(workspace,render=True)
        return jsonify(result)

    @app.put("/api/tracks/<track_id>")
    def edit_track(track_id):
        track=workspace.edit_track(track_id,body())
        if app.config["UPDATE_REPORT"]:
            update_assumptions(workspace,render=True)
        return jsonify(json_safe(track))

    @app.post("/api/views")
    def save_view():
        return jsonify(json_safe(workspace.save_view(body())))

    @app.post("/api/export/<format_name>")
    def export(format_name):
        selection=body()
        data=workspace.analysis(selection)
        if format_name=="csv":
            result=comparison_csv(data)
            mime="text/csv"
        elif format_name=="json":
            result=json.dumps(json_safe({"selection":selection,"analysis":data}),indent=2,allow_nan=False).encode()
            mime="application/json"
        elif format_name=="pdf":
            result=comparison_pdf(data,workspace,selection)
            mime="application/pdf"
        else:
            abort(404)
        return Response(result,mimetype=mime,
                        headers={"Content-Disposition":f'attachment; filename="telemetry-comparison.{format_name}"'})

    @app.get("/plot.svg")
    def legacy_svg():
        try:
            numbers=tuple(sorted(set(int(n) for n in request.args.getlist("lap"))))
        except ValueError:
            abort(400)
        domain=request.args.get("domain","distance")
        if not numbers or any(n<1 or n>13 for n in numbers) or domain not in ("distance","time"):
            abort(400)
        data=workspace.analysis({"track":"onboarding","reference":"onboarding:12",
                                 "laps":[f"onboarding:{n}" for n in numbers]})
        with RENDER_LOCK:
            fig=comparison_figure(data,domain)
            stream=io.BytesIO(); fig.savefig(stream,format="svg")
        return Response(stream.getvalue(),mimetype="image/svg+xml")

    @app.get("/report.pdf")
    def report():
        return send_from_directory(OUT/"pdf","telemetry_report_draft.pdf")

    @app.get("/track.png")
    def track_image():
        return send_from_directory(OUT/"figures","track_map.png")

    return app

if __name__=="__main__":
    create_app().run(host="127.0.0.1",port=5052,debug=False)
