# Suspension IMU analysis

Processes three IMU acceleration channels, reports RMS summaries and extrema-based swing counts, and can plot the channels. It is an acceleration-characterisation tool, not a suspension force or fatigue-life calculator.

## Run

From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/Suspension_Load_Cycle_Analysis.py
```

Place an input CSV in `data/`. The current filename is configured near the top of the script. Results are written to `output/`. Both folders are ignored by Git.

See [`AGENTS.md`](AGENTS.md) for file roles, dependencies and the calculation assumptions and limitations.
