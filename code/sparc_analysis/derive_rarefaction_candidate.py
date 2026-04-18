from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "ksp_analysis" / "KSP_IRS_eigenmode_summary.csv"
DATA_DIR = ROOT / "dark_gravity" / "Rotmod_LTG"
OUT_DIR = ROOT / "ksp_analysis"

COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]


def load_galaxy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=COLS, dtype=float, engine="python")
    df = df.dropna(how="all")
    df["Vbar"] = np.sqrt(df["Vgas"] ** 2 + df["Vdisk"] ** 2 + df["Vbul"] ** 2)
    df["DeltaV2"] = df["Vobs"] ** 2 - df["Vbar"] ** 2
    return df


summary = pd.read_csv(SUMMARY)
usable = summary.dropna(subset=["Vobs_outer_mean", "Vbar_outer_mean", "DeltaV2_outer_mean"]).copy()
neg = usable[usable["DeltaV2_outer_mean"] < 0].copy()
pos = usable[usable["DeltaV2_outer_mean"] > 0].copy()

usable["outer_baryon_ratio"] = usable["Vbar_outer_mean"] / usable["Vobs_outer_mean"]
usable["closure_margin"] = (usable["Vbar_outer_mean"] ** 2 - usable["Vobs_outer_mean"] ** 2) / np.maximum(usable["Vbar_outer_mean"] ** 2, 1e-9)
usable["abs_signed_fit"] = np.abs(usable["V_signed_fit"])

neg["outer_baryon_ratio"] = neg["Vbar_outer_mean"] / neg["Vobs_outer_mean"]
neg["closure_margin"] = (neg["Vbar_outer_mean"] ** 2 - neg["Vobs_outer_mean"] ** 2) / np.maximum(neg["Vbar_outer_mean"] ** 2, 1e-9)
neg["abs_signed_fit"] = np.abs(neg["V_signed_fit"])

report_lines = []
report_lines.append("# Rarefaction Candidate Derivation\n")
report_lines.append("## Sample Counts\n")
report_lines.append(f"- Usable galaxies with outer summary metrics: {len(usable)}")
report_lines.append(f"- Negative-branch galaxies (DeltaV2_outer_mean < 0): {len(neg)}")
report_lines.append(f"- Positive-branch galaxies (DeltaV2_outer_mean > 0): {len(pos)}\n")

if len(neg) > 0:
    report_lines.append("## Negative-Branch Descriptives\n")
    report_lines.append(f"- Median outer baryon ratio Vbar/Vobs: {neg['outer_baryon_ratio'].median():.3f}")
    report_lines.append(f"- Median closure margin: {neg['closure_margin'].median():.3f}")
    report_lines.append(f"- Median signed-fit amplitude |V_signed_fit|: {neg['abs_signed_fit'].median():.3f} km/s")
    report_lines.append(f"- Mean negative fraction in outer subset: {neg['negative_fraction'].mean():.3f}\n")

    corr_ratio = neg[["outer_baryon_ratio", "abs_signed_fit"]].corr(method="spearman").iloc[0, 1]
    corr_margin = neg[["closure_margin", "abs_signed_fit"]].corr(method="spearman").iloc[0, 1]
    report_lines.append("## Empirical Correlations Within Negative Branch\n")
    report_lines.append(f"- Spearman rho(outer_baryon_ratio, |V_signed_fit|): {corr_ratio:.3f}")
    report_lines.append(f"- Spearman rho(closure_margin, |V_signed_fit|): {corr_margin:.3f}\n")

    report_lines.append("## Candidate Rarefaction Law\n")
    report_lines.append(
        "A minimal testable phenomenological law from the current data is to treat rarefaction amplitude as a signed response branch activated when the outer baryonic closure margin changes sign:\n"
    )
    report_lines.append("$$")
    report_lines.append("R_{\\rm rare}(g) := \\sqrt{\\max(V_{\\rm bar}^2 - V_{\\rm obs}^2, 0)}")
    report_lines.append("$$")
    report_lines.append("with activation condition")
    report_lines.append("$$")
    report_lines.append("\\Delta V^2_{\\rm outer} = V_{\\rm obs}^2 - V_{\\rm bar}^2 < 0.")
    report_lines.append("$$")
    report_lines.append(
        "A more normalized law, better suited for cross-galaxy comparison, is to use the closure margin as the control coordinate:\n"
    )
    report_lines.append("$$")
    report_lines.append("\\mathcal{R}_{\\rm rare} := \\sqrt{\\max\\!\\left(\\frac{V_{\\rm bar}^2 - V_{\\rm obs}^2}{V_{\\rm bar}^2}, 0\\right)}.")
    report_lines.append("$$\n")
    report_lines.append(
        "This does not yet derive a microscopic law. It does, however, produce a falsifiable empirical branch law: galaxies with persistent negative outer closure margin should exhibit a nonzero rarefaction amplitude that scales monotonically with closure margin if the deficit branch is real rather than noise.\n"
    )

    report_lines.append("## Immediate Test Suggested\n")
    report_lines.append("1. Stratify negative-branch galaxies by closure margin.")
    report_lines.append("2. Test whether |V_signed_fit| increases monotonically with closure margin.")
    report_lines.append("3. Reject the rarefaction branch if the negative subset is indistinguishable from observational scatter under bootstrap or permutation resampling.\n")

    report_lines.append("## Negative-Branch Galaxies\n")
    report_lines.append("```\n")
    report_lines.append(neg[["Galaxy", "DeltaV2_outer_mean", "V_signed_fit", "outer_baryon_ratio", "closure_margin", "negative_fraction"]].sort_values("DeltaV2_outer_mean").to_string(index=False))
    report_lines.append("\n```")
else:
    report_lines.append("No negative-branch galaxies were identified in the usable summary subset.\n")

(OUT_DIR / "Rarefaction_candidate_note.md").write_text("\n".join(report_lines), encoding="utf-8")
print(f"Wrote {(OUT_DIR / 'Rarefaction_candidate_note.md')}")