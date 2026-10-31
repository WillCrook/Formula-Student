# TBRe telemetry workbench

A local Python/Flask viewer for comparing laps, corners and runs. All code, source
files, imported data and dependencies stay inside this project directory.

## Start

```sh
.venv/bin/python viewer.py
```

Open <http://127.0.0.1:5052>. Stop with Ctrl-C.

The viewer serves its Plotly bundle locally. It does not need an internet
connection after setup, and does not send telemetry to a plotting service.

## Use the viewer

1. Tick laps and choose the reference lap. References can come from another run
   assigned to the same track.
2. Pick a turn/sector, click a turn on the map, or drag over the telemetry.
   All channel axes zoom together. Hover links the plotted samples to their
   positions on the map.
3. Use **Reset gap at entry** to see time gained/lost within the selected section.
   The Section tab shows physical-gate times and entry/minimum/exit GPS speeds.
4. The other tabs show sector consistency, a best-sector composite benchmark,
   logged energy, and GPS-derived G-G scatter.
5. **Save view** stores the selected laps, reference, range and display settings.
   Load it from **Saved views**. Track edits and imported sources persist across
   server restarts.
6. Export the selected comparison as CSV, JSON, PDF or plot PNG.

Turn numbers are initially suggestions, not official circuit numbering.
**Edit** lets you rename turns, move bounds, confirm, merge, split or remove them.
Timing sectors are separately editable and must partition the complete lap.
Using turns as sectors places partition boundaries midway between turn markers.

## Import a run

Use **Import data** and select one or more CSV, TSV, timed GPX or numeric MATLAB
v5/v7 files. MATLAB v7.3/HDF5 is not supported.

- Preview the source columns and map position/time fields.
- Choose decimal-degree latitude/longitude, or X/Y in metres/millimetres.
- Map other channels to their own timestamps and state their units. Known clock
  offsets can be entered in seconds. Custom numeric channels are supported.
- For a new track, click the path to place the finish gate, then adjust its width,
  rotation or direction. Preview the complete laps before adding the run.
- For another run on an existing track, select that track. It reuses the fixed
  coordinate origin and finish line. Planar X/Y must already share that frame.

Numeric clocks may use seconds or milliseconds. ISO 8601 timestamps need a
timezone offset or Z. Duplicate/decreasing timestamps are rejected. GPS sample
gaps are reported; irregular GPS sampling disables the acceleration estimator.
Supporting-channel gaps and missing coverage remain unavailable.

The supplied four MAT files can also be selected together:
GPStime/GPSlat/GPSlon for position, ThrottleTime/Throttle, BrakeTime/BrakeFront,
and PowerTime/Power. GPSalt is available as an optional channel; its supplied
unit/datum metadata is unspecified. Throttle is % and power is kW per the starter
comments; front-brake units are unknown.

CSV example:

```csv
time,latitude,longitude,power_kw,throttle_pct
0.0,52.07812,-1.01110,12.0,60
0.1,52.07813,-1.01109,13.0,70
```

Each channel may instead come from its own file and timestamp column.
Raw inputs and mapping choices are retained under `data/workspace/imports/`.
The browser import does not execute MATLAB code.

## Reports and assumptions

[Analysis report + assumptions appendix](output/pdf/telemetry_report_draft.pdf)

The first three pages retain the original onboarding analysis. The appendix
records coordinate/clock assumptions, import mappings, corner proposals,
physical gates, filtering, composite-lap limitations, energy semantics and the
current workspace summary. It refreshes when runs or track definitions change.

- `output/report.md`: editable original analysis source.
- `output/workbench_assumptions.md`: generated assumptions/provenance appendix.
- `output/pdf/telemetry_report_draft.pdf`: combined PDF.
- The PDF exported from the viewer contains the selected comparison and method
  assumptions.
- `data/workspace/workspace.json`: local track definitions, run mappings and
  saved views. This directory is ignored by Git but persists on disk.

To rebuild the combined PDF after editing the report:

```sh
.venv/bin/python report.py
```

To regenerate the original analysis figures/tables and base Markdown:

```sh
.venv/bin/python analyse.py
```

**analyse.py overwrites output/report.md.** It retains the workbench appendix
when rendering. The original `section_metrics.csv` is the normalised-distance
baseline; workbench comparison exports use physical section boundaries.

## Calculation guide

`telemetry.py` retains the supplied finite-segment, direction-filtered,
interpolated lap crossings, horizontal metre coordinates, cumulative distance,
GPS speed and time-gap conventions. Glat/Glong are calculated from centred cubic
GPS position derivatives, in g; they are not measured IMU/body-axis loads.

`workbench_analysis.py` adds fixed-path corner suggestions, common physical
section gates, focused between-gate distance alignment, native-clock power
integration, sample-standard-deviation consistency and best-sector composites.

`importers.py` handles file formats, explicit units/time mapping and finish
placement. `workspace.py` handles persistence and track/run compatibility.
`viewer.py` exposes the local API. `static/workbench.js` and the HTML/CSS provide
the interface; `exports.py` creates comparisons and the report appendix.

The original supplied files remain unchanged under `data/original/` and at the
pre-existing MATLAB file locations. Source hashes are checked by
`verify_outputs.py`. The supplied baseline remains 14 crossings, 13 complete
laps, fastest lap 12 at 62.407 s and lap 11 at +0.274 s.

## Setup and checks

Python 3.12 was used. The delivered environment is project-local.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python analyse.py
.venv/bin/python verify_outputs.py
.venv/bin/python -m pip check
```

Node is only needed for frontend checks, not for running the application.
These checks were run with Node 26.12.1:

```sh
npm ci --ignore-scripts
npm run check:js
npm run test:ui
npm run test:plots
```

The DOM workflow checks start disposable local workspaces automatically. They
cover imports, cross-run selection, turn edits, zoom/cursor event flows, saved
views and exports. The Plotly smoke check uses the actual SVG renderer in a DOM
with approximate text metrics. Neither test replaces visual browser review.

A live FastF1 adapter is not implemented. It can use the same source-independent
run/track model later. Pedal-event detection/comparison is not included.
