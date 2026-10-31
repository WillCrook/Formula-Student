"""Downloadable comparisons and the workbench assumptions appendix."""
from __future__ import annotations
import csv
import io
from pathlib import Path
import threading
from xml.sax.saxutils import escape
import numpy as np
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from workspace import json_safe
from workbench_analysis import sector_statistics, lap_summary
ROOT=Path(__file__).resolve().parent
RENDER_LOCK=threading.RLock()

ASSUMPTIONS=[
("Position and clocks","GPS coordinates are projected onto a local WGS84 tangent plane; planar imports must already share a metre-based frame. Coordinates/units and clock offsets selected during import are user assertions, not verified sensor calibration. Supporting channels are interpolated on their own clocks without extrapolation."),
("Incomplete laps and gaps","A lap runs between forward crossings of the finite finish gate. Partial data is excluded. Imports reject duplicate/decreasing timestamps and invalid GPS coordinates. GPS gaps are retained and reported; section crossings spanning a gap above max(0.5 s, 3 median intervals) are unavailable. The supplied complete laps are retained rather than filtering slow laps as driver errors."),
("Reference path and alignment","Each track retains the fastest complete lap at track creation as a fixed reference path. Changing the comparison lap does not move physical gates. Whole-lap plots use normalised travelled distance. Focused plots use proportional distance between shared physical entry/exit gates. Interior correspondence is approximate even when timing endpoints are physical. Time-domain plots compare elapsed times, not equal positions, and omit time-gap channels."),
("Section timing","Internal gates are 20 m transverse lines normal to the fixed reference path. A crossing must occur within 6 percentage points of its expected progress; missing or multiple accepted crossings make that section unavailable. Gates cannot wrap across the start/finish. Entry/minimum/exit speeds use the measured-time section boundaries; minimum speed is not identified as a geometric apex. The original section_metrics.csv retains its normalised baseline; workbench section exports use physical gates."),
("Gap convention","Cumulative delta is selected elapsed time minus reference elapsed time. Positive means behind. Section delta subtracts the entry gap and starts at zero; its finish equals the physical section-time difference. An unavailable gate does not receive an invented section time; approximate traces may still be displayed."),
("Corner suggestions","Signed curvature comes from a cubic fit over approximately 10 m of the fixed reference path. Seeds exceed 0.025 per metre, extend to 0.010 per metre, bridge same-sign gaps up to 4 m, and require at least 4 m arc and 15 degrees heading change. Suggestions are editable, not official turns. Short bends, chicanes and bends across start/finish can be split or merged differently. A marker is a curvature centre, not a measured apex. Using turns as sectors places partition boundaries midway between ordered turn markers, retaining the intervening straights."),
("GPS acceleration and G-G","Glat and Glong use centred 11-sample cubic first/second position derivatives on uniform GPS timestamps, with g0=9.80665 m/s^2. Positive Glat is left; positive Glong is speeding up. The first/last five session samples and speeds below 2 m/s are unavailable. Irregular GPS sampling disables these derived channels instead of silently resampling. These are horizontal path-relative estimates, without body heading, sideslip, banking, roll or gravitational accelerometer-load correction. Filtering changes peaks; G-G scatter is not a validated friction envelope."),
("Consistency and composite benchmark","Statistics include every complete lap in the loaded runs on the active track; valid physical sections only. Spread is sample standard deviation (n-1); unavailable for n<2. The best-sector sum combines the fastest observed time in each non-overlapping partition sector. Different entry/exit states and conditions mean it is not automatically an achievable lap or a predicted improvement."),
("Power and energy","Power is in kW per the supplied comments or explicit import mapping (W is converted). Integrate the native power timestamps with trapezoids and interpolated boundaries: kW*s=kJ; kWh=kJ/3600. Coverage gaps/missing values produce unavailable energy. Logged energy is not assumed to be battery energy or wheel energy: measurement point, losses and regeneration semantics are unverified. Any signed imported power is integrated algebraically."),
("Visual overlays","The gain/loss map shows a smoothed whole-lap delta slope in s per 100 m, using a centred 31-point quadratic fit on the normalised grid. It is a visual aid, not independent sector timing or proof of cause. G-G speed colours use derived GPS speed; corner colours use editable candidate membership. Pedal-event detection/comparison is intentionally absent."),
("Persistence and imports","Raw CSV/TSV, timed GPX and numeric MATLAB v5/v7 sources are retained unchanged under data/workspace/imports. MATLAB v7.3/HDF5 and a FastF1 adapter are not implemented. GPX imports require timed track points, with ISO timezone information. Supported imports have explicit channel/time/unit mapping; each run is compared only within its assigned track. A 95th-percentile nearest-reference-path separation over 25 m rejects a run assigned to an existing track. This is a coarse compatibility check, not coordinate accuracy validation. Saved views and track edits persist locally; no files are uploaded to an external service."),
]

def assumptions_text(workspace):
    lines=["# Workbench assumptions and data provenance",
           "Local interactive viewer | Assumptions for review | 29 October 2026",""]
    for title,text in ASSUMPTIONS:
        lines.extend([f"## {title}",text,""])
    lines.extend(["## Loaded runs",""])
    for config in workspace.catalog["runs"].values():
        name=str(config["name"]).replace("|","/").replace("\n"," ")
        lines.append(f"{name}: "+("supplied onboarding MAT files" if config.get("builtin") else ", ".join(s["name"] for s in config["sources"]))+".")
        if not config.get("builtin"):
            mapping=config["mapping"]
            lines.append("Import mapping: "+str(json_safe(mapping)).replace("\n"," ")+".")
        try:
            run=workspace.load_run(config["id"])
            acceleration=run.session.metadata.get("gps_acceleration")
            if acceleration:
                lines.append(f"GPS derivative fit: {acceleration['window_samples']} samples spanning {acceleration['window_span_s']:.3f} s; {acceleration['min_speed_mps']:.1f} m/s minimum travel speed.")
            for notice in run.notices:
                lines.append("Data note: "+notice)
        except (ValueError,OSError) as error:
            lines.append("Data note: "+str(error))
        lines.append("")
    lines.extend(["## Current workspace results",""])
    for track in workspace.catalog["tracks"].values():
        try:
            runs=workspace.runs_for(track["id"])
            stats=sector_statistics(track,runs)
            lines.append(str(track["name"]).replace("|","/")+f": {stats['scope_laps']} complete laps; {len(track['corners'])} editable turn suggestions.")
            if stats["composite_s"] is not None:
                lines.append(f"Best complete lap: {stats['best_complete_s']:.3f} s. Best-sector sum: {stats['composite_s']:.3f} s. Difference: {stats['potential_s']:.3f} s (composite benchmark, not a predicted achievable improvement).")
            lines.extend(["| Sector | Best (s) | Median (s) | Std (s) | n |","| --- | --- | --- | --- | --- |"])
            for sector in stats["sectors"]:
                def formatted(value):
                    return "-" if value is None else f"{value:.3f}"
                lines.append("| "+str(sector["name"]).replace("|","/")+" | "+" | ".join(formatted(sector[key]) for key in ["best_s","median_s","std_s"])+" | "+str(sector["count"])+" |")
            energy=[lap_summary(run,lap)["energy_kj"] for run in runs for lap in run.laps]
            energy=[e for e in energy if e is not None]
            if energy:
                lines.append(f"Complete-lap logged energy spans {min(energy):.1f}-{max(energy):.1f} kJ over {len(energy)} laps with coverage.")
            lines.append("")
        except (ValueError,OSError) as error:
            lines.append("Workspace summary unavailable: "+str(error))
    return "\n".join(lines)

def update_assumptions(workspace,render=False):
    text=assumptions_text(workspace)
    with RENDER_LOCK:
        (ROOT/"output").mkdir(exist_ok=True)
        (ROOT/"output/workbench_assumptions.md").write_text(text)
        if render:
            from report import render_report
            render_report()
    return text

def comparison_csv(data):
    stream=io.StringIO(newline="")
    channels=[key for key in data["channels"] if key not in ("delta","local_delta","speed")]
    fields=["lap","track_progress","reference_distance_m","elapsed_s","cumulative_gap_s","section_gap_s","speed_kmh"]+[
        key+("_g" if key in ("glat","glong") else "") for key in channels]
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for trace in data["traces"]:
        for j,u in enumerate(trace["u"]):
            row={"lap":trace["label"],"track_progress":u,"reference_distance_m":u*data["track"]["length_m"],
                 "elapsed_s":trace["elapsed"][j],"cumulative_gap_s":trace["delta"][j],
                 "section_gap_s":trace["local_delta"][j],"speed_kmh":trace["speed"][j]}
            for key in channels:
                row[key+("_g" if key in ("glat","glong") else "")]=trace[key][j] if key in trace else None
            writer.writerow({k:("" if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in row.items()})
    return stream.getvalue().encode("utf-8-sig")

def comparison_figure(data,domain="distance",local=False,channels=None):
    available=data["channels"]
    channels=channels or ["speed","delta","throttle","brake","glat","glong"]
    channels=[key for key in channels if key in available and (domain=="distance" or key not in ("delta","local_delta"))]
    if not channels:
        channels=["speed"]
    fig=Figure(figsize=(10,max(4,1.35*len(channels))),layout="constrained")
    axes=fig.subplots(len(channels),1,sharex=True,squeeze=False).ravel()
    palette=["#087f72","#da7747","#6271bb","#a5639c","#6b9453","#c09b43"]
    for n,trace in enumerate(data["traces"]):
        for ax,key in zip(axes,channels):
            value_key="local_delta" if key=="delta" and local else key
            if value_key not in trace:
                continue
            x=trace["elapsed"] if domain=="time" else trace["u"]*data["track"]["length_m"]
            ax.plot(x,trace[value_key],label=trace["label"],lw=1.1,color=palette[n%len(palette)])
            label=available[value_key]["label"]
            if key in ("glat","glong"):
                label+=" estimate"
            ax.set_ylabel(label+" ("+available[value_key]["unit"]+")",fontsize=8)
            ax.grid(alpha=.2)
            if key in ("delta","glat","glong"):
                ax.axhline(0,color="black",lw=.5)
    axes[0].legend(fontsize=7,ncol=min(4,len(data["traces"])))
    axes[-1].set_xlabel("Elapsed lap time (s)" if domain=="time" else "Reference-path distance (m)")
    return fig

def comparison_pdf(data,workspace,selection):
    with RENDER_LOCK:
        styles=getSampleStyleSheet()
        body=styles["BodyText"]; body.fontSize=9; body.leading=12
        stream=io.BytesIO()
        story=[Paragraph("Telemetry comparison",styles["Title"]),
               Paragraph(escape(data["track"]["name"]),styles["Heading2"]),
               Paragraph(f"Selected range: {100*data['start']:.1f}-{100*data['end']:.1f}% | Reference: "+escape(data["reference"]),body),
               Spacer(1,10)]
        rows=[["Lap","Time (s)","Gap (s)","Entry/min/exit km/h","Energy kJ"]]
        def fmt(value,digits=3):
            return "-" if value is None else f"{value:.{digits}f}"
        for trace in data["traces"]:
            m=trace["metrics"]
            rows.append([Paragraph(escape(trace["label"]),body),fmt(m["time_s"]),fmt(m["delta_s"]),
                         "/".join(fmt(m[k],1) for k in ["entry_kmh","minimum_kmh","exit_kmh"]),fmt(m["energy_kj"],1)])
        table=Table(rows,colWidths=[145,65,65,145,70],repeatRows=1)
        table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
                                   ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#deeeeb")),
                                   ("VALIGN",(0,0),(-1,-1),"TOP"),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([table,Spacer(1,12)])
        for trace in data["traces"]:
            if trace["metrics"]["status"]!="ok":
                story.append(Paragraph(escape(trace["label"]+": "+trace["metrics"]["status"]),body))

        fig=comparison_figure(data,selection.get("domain","distance"),bool(selection.get("local_delta",False)),
                              channels=["speed","delta","glat","glong"])
        image=io.BytesIO(); fig.savefig(image,format="png",dpi=150); image.seek(0)
        story.append(Image(image,width=510,height=510*fig.get_figheight()/fig.get_figwidth()))
        story.append(PageBreak())
        story.append(Paragraph("Assumptions and interpretation",styles["Heading1"]))
        for title,text in ASSUMPTIONS:
            story.append(Paragraph(escape(title),styles["Heading3"]))
            story.append(Paragraph(escape(text),body))
        for notice in data["notices"]:
            story.append(Paragraph(escape(notice),body))
        story.append(Paragraph("Selected sources",styles["Heading3"]))
        for run_id in dict.fromkeys(trace["id"].split(":")[0] for trace in data["traces"]):
            source=workspace.catalog["runs"][run_id]
            files="supplied onboarding MAT files" if source.get("builtin") else ", ".join(item["name"] for item in source["sources"])
            story.append(Paragraph(escape(source["name"]+": "+files),body))
        styles["Heading3"].keepWithNext=1
        body.allowOrphans=0
        body.allowWidows=0
        def footer(canvas,doc):
            canvas.setFont("Helvetica",8)
            canvas.setFillColor(colors.HexColor("#63786e"))
            canvas.drawString(35,16,"Telemetry comparison | Review draft")
            canvas.drawRightString(A4[0]-35,16,str(doc.page))
        doc=SimpleDocTemplate(stream,pagesize=A4,leftMargin=35,rightMargin=35,topMargin=30,bottomMargin=30)
        doc.build(story,onFirstPage=footer,onLaterPages=footer)
        return stream.getvalue()
