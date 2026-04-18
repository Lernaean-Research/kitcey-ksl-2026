# Reproducibility Notes

## Determinism
- Use fixed seeds for all simulation and scorecard runs.
- Archive both timeseries and summary outputs for each run.

## Primary Commands
- Bounded instantiation (Sec. 3.9):
  `python code/toy_model/bounded_instantiation.py`
- Scorecard generation:
  `python code/scorecard/generate_scorecard_report.py`
- SPARC eigenmode analysis:
  `python code/sparc_analysis/ksp_irs_eigenmode_test.py`

## Data Policy
- Raw SPARC files are not committed.
- Use `python data/sparc/download_rotmod_ltg.py` to fetch Rotmod_LTG.
