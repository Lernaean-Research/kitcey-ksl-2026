# Contributing Guide

## Scope

This repository is the reproducibility companion for the KSL/KSP v2.2.0 manuscript and Zenodo record.

## Ground Rules

1. Keep changes additive and traceable.
2. Do not commit large raw SPARC archives; use the download helper in `data/sparc/`.
3. Preserve deterministic behavior by using fixed seeds where scripts already define them.
4. For manuscript-linked claims, include exact run commands and generated artifact paths.

## Development Setup

Use either:

- `conda env create -f environment.yml`
- `pip install -r requirements.txt`

## Verification Before Pull Request

1. Run bounded instantiation:
   - `python code/toy_model/bounded_instantiation.py`
2. If scorecard logic changed, run:
   - `python code/scorecard/generate_scorecard_report.py`
3. Confirm outputs are produced in `outputs/csv` and `outputs/json`.

## Pull Request Expectations

1. Explain scientific or reproducibility motivation.
2. Link related issue.
3. List commands executed and artifacts checked.
4. Keep commit history clean and descriptive.
