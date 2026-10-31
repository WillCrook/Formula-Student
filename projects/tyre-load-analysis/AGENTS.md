# Tyre-load analysis instructions

## Purpose and current state

This project contains an initial Python `Tyre_Loads` class with vehicle-parameter values intended for tyre-load calculations. The file currently defines constants only; it has no load model, executable entry point, tests or external dependencies.

## Files

- `src/Tyres_and_Suspension_Members.py` — current parameter scaffold.

## Engineering assumptions and validation

Several values have comments questioning their meaning or source, including front weight distribution and track widths. Treat all values as provisional until checked against an authoritative car specification. Preserve units in names or documentation, state the sign convention and load-transfer model before adding calculations, and distinguish measured inputs from assumed values.

## Running and testing

The current file uses only the Python standard library and can be syntax-checked with `python -m py_compile src/Tyres_and_Suspension_Members.py`. It has no automated tests or calculation to run yet. Add focused tests when implementing a defined load model; document its equations, limits and expected units first.
