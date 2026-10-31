# Validation - 29 October 2026

## Completed checks

- 32 Python unittest checks passed, covering the original lap/acceleration
  calculations and the new workbench import, section, energy and persistence paths.
- The DOM workflow test passed: reference selection, all laps, time/distance,
  range zoom, local delta, linked cursor, map turn selection, analysis tabs,
  turn merge/split, sector editing, saved views, exports, new-track import,
  cross-run import and saved-view restoration.
- The actual Plotly 4.1.1 SVG renderer passed a DOM smoke check for shared-axis
  telemetry, the track map, time-domain plots, consistency boxes and G-G scatter.
- JavaScript syntax check and Python compilation passed.
- The documented analyse.py run, verify_outputs.py and pip check passed.
- Source hashes match the supplied originals. The pre-existing root MATLAB
  copies are also unchanged.
- Original lap-summary and normalised section-result CSVs retain their baseline
  values: 14 crossings, 13 complete laps, fastest lap 12 at 62.407274 s and
  second lap 11 at 62.681582 s.
- Focused section delta starts at zero and ends at the physical section-time
  difference. Partition sector times sum to each lap time.
- Constant/ramp power integration was checked on its native timestamps, including
  missing coverage, sample gaps and NaNs. Supporting-channel gaps remain
  unavailable in plots.
- Import tests cover CSV unit conversion, independent timestamps/offsets, MAT
  vectors, timed GPX, duplicate clocks, reserved calculated-channel names,
  finish-gate direction, saved-source hashes and cross-run coordinate compatibility.
- Invalid track updates are atomic: an invalid sector edit cannot partly change
  the corner definitions.
- CSV/JSON/PDF comparison responses passed. A focused PDF export was rendered
  and all three pages visually checked. PNG action wiring was checked at DOM
  level; a real browser download was not exercised.
- The combined main report has five pages: three analysis pages plus two
  assumptions/provenance pages. All five final rendered pages were visually
  inspected, including headings, tables and footers.

## Scope and limits

A real browser could not be attached: the browser inventory was empty and the
native computer-use pipe failed to start. Responsive visual layout, pointer hit
testing and actual browser file/download dialogs remain unverified.
DOM tests stub Plotly for workflow coverage; the separate renderer test uses the
real Plotly library with approximate text metrics. These are not a substitute
for browser visual review.

Automatic turns remain editable suggestions, not official numbering. The
best-sector sum for the initial three physical sectors is 61.835 s, a composite
benchmark rather than an achievable-lap prediction. Logged energy is not assumed
to represent battery or wheel energy. Full assumptions and import mappings are
in output/workbench_assumptions.md and the combined PDF.

All application data, sources, tests, dependencies and outputs stay inside this
project directory. Test imports use disposable workspaces. Nothing was sent or
submitted externally.
