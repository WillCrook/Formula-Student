# Task 1: TBRe26 rear-heave damper C-V curves

This is a small MATLAB starting implementation and an informal draft report for review. It analyses the supplied Blue Valve force-velocity data. All new files and generated outputs stay in this folder. No submission is sent.

## Run

Requires **MATLAB R2025b** (the version used here), with no extra toolboxes or packages.

Open this folder as MATLAB's Current Folder and run:

```matlab
analyse_cv_curves
```

Or from Terminal in this folder:

```sh
/Applications/MATLAB_R2025b.app/bin/matlab -batch "analyse_cv_curves"
```

For the focused derivative checks alone:

```matlab
test_incremental_damping
```

Rerunning replaces this task's generated files in `output/`. It does not change the four supplied files. Figures are created invisibly, exported, then closed; existing user figures are left alone. The script uses `clear`, so save any workspace variables you need before running.

## Start with these outputs

- `output/TBRe26_rear_heave_CV_report_draft.pdf`: three-page draft; review before submitting.
- `output/report_draft.md`: editable text copy with relative figure links.
- `output/01_force_velocity.png`: representative source force-velocity curves.
- `output/02_cv_full.png`: full incremental-damping range, including near zero.
- `output/03_cv_zoom.png`: the same rates with the vertical axis limited to 0-6000 N s/m. Higher values are clipped for display only.
- `output/04_near_zero_force.png`: near-zero force-velocity source points.
- `output/05_all_clicks.png`: all 12 settings, with full-range logarithmic damping-rate axes.
- Matching SVG files are provided for scalable figure export.
- `output/comparisons.csv`: all settings at signed shaft velocities of 25, 50, 100 and 200 mm/s.
- `output/curves.csv`: every sorted source point and its calculated nodal rate.
- `output/near_zero.csv`: first-point diagnostics, complete ranges and extrema.
- `output/source_audit.csv`, `output/verification.txt`: source agreement and numerical checks.
- `output/analysis.mat`: the curves and comparison tables for further MATLAB inspection.

To edit the report wording and rebuild the PDF, edit **`build_report.m`**, then rerun the main script. The Markdown copy is also editable but is regenerated on each run; edits made only to that copy do not update the MATLAB-rendered PDF.

## What the supplied files do

The four files in this folder were checked byte-for-byte against the supplied archive and kept unchanged.

- `Taster1_Task1.m` demonstrates a linear damper, F = c*v, with c = 1500 N s/m.
- `Taster1_Task2.m` plots one real click setting and interpolates force. Its name refers to the first session's live example, not the later GPS recruitment task.
- `TBRe_BlueValve_Data.m` provides `Clicks = 0:11`, with `Bump{click+1}` and `Rebound{click+1}`. Each curve contains `v_mmps`, `F_N` and a click label.
- `BLUE Valve Raw Data.xlsx`, Sheet1: curve names on row 1, variable names on row 4, units on row 5 and numerical pairs from row 6. Bump uses A:X and rebound AA:AX.

The data function's `SourceFile` text names an earlier workbook filename. The analysis checks against the actual supplied workbook above, rather than trusting that metadata.

The 14 October slides call damping-rate curves an extension. The 22 October slides and the email quoted in the brief make dF/dv the recruitment task. The core deliverable is therefore C-V, not just running the starter plots. The optional vehicle damping-ratio study is outside this implementation.

## Calculation walkthrough

1. Read each bump and rebound branch separately. Check the workbook headers, units, paired numerical values and click labels against the data function.
2. Sort velocity and its corresponding force using the same index. Reject duplicate velocities, nonfinite samples, inconsistent lengths or incorrect branch signs.
3. Convert v from mm/s to m/s: `v_mps = v_mmps / 1000`.
4. Calculate `C = gradient(F_N, v_mps)` in N s/m. MATLAB's interior estimate is
   `(F(i+1)-F(i-1))/(v(i+1)-v(i-1))`; endpoints use adjacent one-sided secants. This uses the actual nonuniform coordinates.
5. Keep the branches independent, including their two separate origin slopes. Retain +v/+F bump and -v/-F rebound. A positive rebound derivative is compatible with both coordinates being negative.
6. Linearly interpolate the computed nodal rates and supplied forces to common signed velocities, only inside every curve's range. These are interpolated finite-difference estimates, not an exact continuous derivative.
7. Export all points and all 12 settings, while showing clicks 0, 4, 8 and 11 in the report for readability. This selection and the 6000 N s/m plot zoom are presentation choices.

Do not replace dF/dv with F/v: F/v is a secant from the origin, generally different from the local slope. The separate zero-to-first diagnostic in `near_zero.csv` is explicitly an adjacent-point secant from the supplied (0,0); it is not used instead of `gradient` for the curves.

## What to understand before submitting

- At +50 mm/s, bump force rises from about 34.9 N at click 0 to 397.7 N at click 11. However, the local rate is about 3619 N s/m at click 8 and 3505 N s/m at click 11. Force ordering does not imply derivative ordering.
- At matched 50 mm/s magnitude, click 11 has C about 3505 N s/m in bump and 2605 N s/m in rebound.
- Click 8's bump C falls from about 3619 at 50 mm/s to 746 N s/m at 200 mm/s. Not every setting is uniformly digressive: click 0 rises from about 627 to 675 N s/m over that interval.
- The zero-to-first secants at clicks 9-11 are approximately 495239 N s/m in bump and 414387 N s/m in rebound. The full plot retains them. The cause is not established by these files.
- Low-speed means damper shaft speed. More damping is not automatically better. A vehicle optimum needs effective modal mass, stiffness, motion ratios and a response objective.

## Verification and limitations

Executed with MATLAB R2025b Update 7. All 24 curves match the workbook to absolute tolerance 1e-9 (in the source units); all 4703 samples pass ordering, duplicates, finite-value and sign checks. All comparison velocities are within the supplied ranges.

The focused tests cover the constant 1500 N s/m coefficient on unsorted, uneven velocity grids in both signed branches, force/velocity pair preservation, a hand-calculated nonlinear secant example, SI scaling and invalid-input rejection. This validates the numerical implementation, not physical accuracy of the source dyno curves.

No points are removed, smoothed or extrapolated. Finite differences amplify noise and rounding; sparse or uneven spacing affects local accuracy. Origin values are treated as supplied, not independently verified measurements. Test temperature, hysteresis, repeatability and uncertainty are not established. A useful next step is to ask about the source and test conditions for the near-zero region, ideally with repeat or denser measurements.

Sources read for context: the supplied MATLAB/data archive, `TBRe27 VD Onboarding Session 1 (14-10-26).pdf`, `TBRe VDG Onboarding Session 2 (22-10-26).pdf`, and the recruitment email as quoted in the pasted brief.
