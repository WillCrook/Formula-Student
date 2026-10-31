"""Check generated artifacts and unchanged local copies of supplied inputs."""
import csv
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/"output"
    manifest=json.loads((ROOT/"data/source_manifest.json").read_text())
    for relative,expected in manifest.items():
        actual=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        assert actual==expected, f"Changed supplied input: {relative}"
    audit=json.loads((out/"audit.json").read_text())
    for key in ["glat","glong"]:
        assert audit["channels"][key]["kind"]=="derived"
        assert audit["channels"][key]["unit"]=="g"
    with (out/"lap_summary.csv").open() as stream:
        laps=list(csv.DictReader(stream))
    assert len(laps)==audit["crossings"]-1==13
    with (out/"aligned_telemetry.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    for lap in laps:
        values=[r for r in rows if r["lap"]==lap["lap"]]
        assert len(values)==1001
        assert float(values[0]["delta_s"])==0
        assert abs(float(values[-1]["delta_s"])-float(lap["gap_to_fastest_s"]))<1e-9
        assert all(np.isfinite(float(r[k])) for r in values for k in
                   ["elapsed_s","delta_s","speed_kmh","throttle","brake","power","glat_g","glong_g"])
    pdf=PdfReader(out/"pdf/telemetry_report_draft.pdf")
    assert len(pdf.pages)>=4, "Expected three analysis pages plus assumptions appendix"
    text="\n".join(p.extract_text() for p in pdf.pages)
    for expected in ["62.407","62.682","0.274","Interpretation and limitations","Sources:","Workbench assumptions and data provenance","Current workspace results"]:
        assert expected in text, f"Missing report content: {expected}"
    for name in ["track_map","all_laps","alignment_check","detail_lap_11","detail_lap_13",
                 "detail_lap_11_time","detail_lap_13_time","acceleration_laps","acceleration_filter_check"]:
        for suffix in ["png","svg"]:
            assert (out/f"figures/{name}.{suffix}").stat().st_size>1000
    print("PASS: source hashes, 13 CSV laps, 13,013 aligned rows, finite channels, delta endpoints, analysis PDF plus assumptions appendix and 18 figure files")

if __name__=="__main__":
    main()
