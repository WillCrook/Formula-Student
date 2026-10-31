# Task-local instructions
Keep all source data, code, dependencies and generated outputs inside this directory.
Original supplied files under data/original must remain unchanged.
Python calculations use seconds, metres and metres/second internally.
Run: .venv/bin/python -m unittest discover -s tests -v
Run: .venv/bin/python analyse.py
Run: .venv/bin/python verify_outputs.py
For interface changes run: npm run check:js; npm run test:ui; npm run test:plots
Node dependencies are test-only and project-local.
The editable report is output/report.md; render with .venv/bin/python report.py.
Do not send or submit the report.
