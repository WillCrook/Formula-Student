# Suspension analysis instructions

## Purpose and scope

This project processes Formula Student IMU acceleration channels as inputs to suspension durability work. The current script characterises acceleration; it does not estimate suspension force, stress, fatigue damage or component life.

## Files

- `src/Suspension_Load_Cycle_Analysis.py` — CSV loading, optional smoothing, extrema-based swing counts, RMS summaries, plots and CSV output.
- `data/` — local input data; ignored by Git.
- `output/` — local generated plots and results; ignored by Git.

## Engineering assumptions

- The source file has 14 metadata rows before the CSV header and contains `InlineAcc`, `LateralAcc` and `VerticalAcc` columns.
- The script treats those channels as acceleration in g, applies an optional centred five-sample rolling mean, and computes RMS values from the retained samples.
- Cycle counts currently use differences between adjacent detected extrema. This is not standard rainflow counting and must not be presented as a fatigue-life estimate.
- `Mag` is the non-negative vector magnitude; signed swing categories on that signal do not represent directional acceleration.
- Confirm sensor orientation, units, sample timing, filtering and the interpretation of any results against the data source before engineering use.

## Dependencies and commands

Dependencies are in `requirements.txt` (NumPy, pandas and Matplotlib). From this directory, create/activate a virtual environment, install them with `python -m pip install -r requirements.txt`, then run `python src/Suspension_Load_Cycle_Analysis.py`. The script reads the named CSV from `data/` and writes plots and `results.csv` to `output/`.

There are currently no automated tests. For changes to the calculations, add focused tests for the affected signal-processing behavior and compare any output changes against a known input. Do not commit local raw data or generated output by default.
