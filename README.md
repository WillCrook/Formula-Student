# Formula Student engineering tools

Independent tools for vehicle engineering work with Team Bath Racing Electric. The current projects are:

- [`suspension-analysis`](projects/suspension-analysis/README.md): Python processing and visualisation of suspension IMU acceleration data.
- [`tyre-load-analysis`](projects/tyre-load-analysis/README.md): an early Python collection of vehicle parameters for tyre-load calculations; it does not yet calculate loads.

Each project keeps its code, documentation, dependencies and local data together. Start work in the project directory you need and follow its `AGENTS.md`; there is no shared runtime package. The root `requirements.txt` is retained for compatibility, while dependencies are listed per project.

Raw data and generated outputs are kept out of Git by `.gitignore`. The existing IMU dataset and prior plots/results are present locally under `projects/suspension-analysis/` and are not tracked.
