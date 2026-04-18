# Environment Setup

## Preferred (Conda)

```bash
conda env create -f environment.yml
conda activate ksl-ksp-bridge
```

## Alternative (pip)

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Smoke Tests

```bash
python code/toy_model/bounded_instantiation.py
python code/scorecard/generate_scorecard_report.py
```

The first command reproduces the Section 3.9 bounded instantiation artifacts.
