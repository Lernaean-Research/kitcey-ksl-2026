# Kit-State Lifecycle (KSL) & Kitcey Synergy Principle (KSP) v2.2.0

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.19638157.svg)](https://doi.org/10.5281/zenodo.19638157)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Reproducible code and artifacts** for the manuscript  
"Kit-State Lifecycle: A Comprehensive Unified Bridge from Quantum Manifestation to Classical Structure"  
R. D. Kitcey, April 2026, v2.2.0

**Zenodo (citable preprint)**: [10.5281/zenodo.19638157](https://doi.org/10.5281/zenodo.19638157)

## Quick Start
1. `git clone https://github.com/Lernaean-Research/kitcey-ksl-2026.git`
2. `cd kitcey-ksl-2026`
3. `conda env create -f environment.yml` (or `pip install -r requirements.txt`)
4. Run the bounded toy model: `python code/toy_model/bounded_instantiation.py`

## Directory Map

```
kitcey-ksl-2026/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── ENVIRONMENT.md
├── requirements.txt
├── environment.yml
├── code/
│   ├── toy_model/
│   ├── sparc_analysis/
│   ├── scorecard/
│   └── utils/
├── data/
│   ├── sparc/
│   └── toy_inputs/
├── outputs/
│   ├── figures/
│   ├── csv/
│   └── json/
├── docs/
│   ├── scorecard_template.md
│   ├── run_report_examples/
│   ├── glossary.md
│   └── reproducibility_notes.md
├── reproducibility/
│   ├── bounded_instantiation/
│   └── capsule/
└── archive/
    └── zenodo_v2.2.0.md
```

## Reproducibility
- All bridge observables are derived from the single governing operator $L(a, \chi)$.
- Bounded-domain mechanism check (Sec. 3.9) and 150-timestep simulation are fully executable.
- SPARC rotation-curve pipeline (Application E) produces spectral-gap, retention, and closure-ledger outputs for all 175 galaxies.

## Citation
Please cite both the Zenodo record and this repository when using the code or data.

## Contact & Issues
Questions or reproducibility issues -> open a GitHub Issue.
