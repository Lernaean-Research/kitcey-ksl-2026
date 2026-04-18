import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Manuscript linkage: Application E SPARC spectral/retention analysis.


print("=== SymPy Derivation of Inverse-Square Eigenmode ===")
r, G, Mbar, beta = sp.symbols("r G M_bar beta", positive=True)
M_IRS = beta * r
M_tot = Mbar + M_IRS
v_tot2 = G * M_tot / r
v_IRS2 = G * M_IRS / r
print("Analytic flat-curve condition (outer disk):")
sp.pprint(sp.simplify(v_IRS2))


ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "sparc" / "Rotmod_LTG"
OUTPUT_DIR = REPO_ROOT / "outputs" / "csv"
OUTPUT_DIR.mkdir(exist_ok=True)

COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
OUTER_FRACTION = 0.4


def load_galaxy(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        names=COLS,
        dtype=float,
        engine="python",
    )
    df = df.dropna(how="all")
    df["Vbar"] = np.sqrt(df["Vgas"] ** 2 + df["Vdisk"] ** 2 + df["Vbul"] ** 2)
    df["DeltaV2"] = df["Vobs"] ** 2 - df["Vbar"] ** 2
    df["VIRS"] = np.sqrt(np.maximum(df["DeltaV2"], 0))
    df["Vdef"] = np.sqrt(np.maximum(-df["DeltaV2"], 0))
    df["Vsigned"] = np.sign(df["DeltaV2"]) * np.sqrt(np.abs(df["DeltaV2"]))
    df["errVsigned"] = df["errV"] * np.abs(df["Vobs"]) / np.maximum(np.sqrt(np.abs(df["DeltaV2"])), 1e-6)
    df["errVsigned"] = df["errVsigned"].replace([np.inf, -np.inf], np.nan).fillna(df["errV"])
    df["errVsigned"] = np.maximum(df["errVsigned"], 1e-6)
    return df


def fit_outer_flatness(df: pd.DataFrame) -> dict:
    r_max = df["Rad"].max()
    r_cut = max(5.0, r_max * (1 - OUTER_FRACTION))
    outer = df[df["Rad"] >= r_cut].copy()

    if len(outer) < 3:
        return {
            "valid": False,
            "r_cut": r_cut,
            "V_IRS_mean": np.nan,
            "V_signed_mean": np.nan,
            "chi2_dof": np.nan,
            "positive_fraction": np.nan,
            "negative_fraction": np.nan,
        }

    def const(x, v):
        return np.full_like(np.asarray(x, dtype=float), v, dtype=float)

    popt, _ = curve_fit(const, outer["Rad"], outer["Vsigned"], sigma=outer["errVsigned"], absolute_sigma=True)
    v_signed_fit = float(popt[0])
    residuals = (outer["Vsigned"] - v_signed_fit) / outer["errVsigned"]
    chi2 = float(np.sum(residuals ** 2))
    chi2_dof = chi2 / max(len(outer) - 1, 1)

    return {
        "valid": True,
        "r_cut": r_cut,
        "N_outer": len(outer),
        "V_IRS_mean": float(np.mean(outer["VIRS"])),
        "V_def_mean": float(np.mean(outer["Vdef"])),
        "V_signed_mean": v_signed_fit,
        "DeltaV2_outer_mean": float(np.mean(outer["DeltaV2"])),
        "chi2_dof": chi2_dof,
        "flatness_good": chi2_dof < 2.0,
        "positive_fraction": float(np.mean(outer["DeltaV2"] > 0)),
        "negative_fraction": float(np.mean(outer["DeltaV2"] < 0)),
    }


if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")


results = []
galaxy_files = sorted(DATA_DIR.glob("*_rotmod.dat"))

print(f"Found {len(galaxy_files)} galaxies. Starting analysis...\n")

for i, file_path in enumerate(galaxy_files, 1):
    galaxy_name = file_path.stem.replace("_rotmod", "")
    try:
        df = load_galaxy(file_path)
        stats = fit_outer_flatness(df)

        outer_mask = df["Rad"] >= stats.get("r_cut", 0)
        row = {
            "Galaxy": galaxy_name,
            "N_points": len(df),
            "R_max_kpc": float(df["Rad"].max()),
            "Vobs_outer_mean": float(df.loc[outer_mask, "Vobs"].mean()) if stats["valid"] else np.nan,
            "Vbar_outer_mean": float(df.loc[outer_mask, "Vbar"].mean()) if stats["valid"] else np.nan,
            "DeltaV2_outer_mean": stats.get("DeltaV2_outer_mean", np.nan),
            "V_IRS_fit": stats["V_IRS_mean"],
            "V_def_fit": stats.get("V_def_mean", np.nan),
            "V_signed_fit": stats.get("V_signed_mean", np.nan),
            "chi2_dof": stats["chi2_dof"],
            "flatness_good": stats.get("flatness_good", False),
            "positive_fraction": stats.get("positive_fraction", np.nan),
            "negative_fraction": stats.get("negative_fraction", np.nan),
            "file": file_path.name,
        }
        results.append(row)

        if i % 25 == 0 or i == len(galaxy_files):
            print(
                f"  [{i:3d}/{len(galaxy_files)}] {galaxy_name:12} -> "
                f"V_signed = {row['V_signed_fit']:.1f} km/s  (chi2/dof = {row['chi2_dof']:.2f})"
            )
    except Exception as exc:
        print(f"  [!] Skipped {galaxy_name} ({exc})")


summary_df = pd.DataFrame(results)
summary_path = OUTPUT_DIR / "KSP_IRS_eigenmode_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nAnalysis complete. Summary saved to {summary_path}")
print(f"Galaxies with good flatness (chi2/dof < 2): {int(summary_df['flatness_good'].sum())} / {len(summary_df)}")
print(f"Galaxies with net positive outer residual support: {int((summary_df['DeltaV2_outer_mean'] > 0).sum())} / {len(summary_df)}")
print(f"Galaxies with net negative outer residual support: {int((summary_df['DeltaV2_outer_mean'] < 0).sum())} / {len(summary_df)}")


plt.figure(figsize=(8, 5))
plt.hist(summary_df["V_signed_fit"].dropna(), bins=30, color="teal", edgecolor="black")
plt.xlabel("Fitted outer V_signed (km/s)")
plt.ylabel("Number of galaxies")
plt.title("KSP Signed Outer Residual Distribution (all galaxies)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "V_signed_histogram.png", dpi=200)
plt.close()


example_file = DATA_DIR / "NGC2403_rotmod.dat"
if example_file.exists():
    df_ex = load_galaxy(example_file)
    stats_ex = fit_outer_flatness(df_ex)

    plt.figure(figsize=(10, 6))
    plt.errorbar(df_ex["Rad"], df_ex["Vobs"], yerr=df_ex["errV"], fmt="o", label="V_obs", color="black")
    plt.plot(df_ex["Rad"], df_ex["Vbar"], label="V_bar (baryonic)", color="blue")
    plt.plot(df_ex["Rad"], df_ex["Vsigned"], label="V_signed residual", color="red", linestyle="--")
    plt.axhline(
        stats_ex["V_signed_mean"],
        color="red",
        alpha=0.6,
        label=f"Flat V_signed fit = {stats_ex['V_signed_mean']:.1f} km/s",
    )
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Circular velocity (km/s)")
    plt.title("NGC 2403 - Signed Residual Eigenmode Test (KSP Application E)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "NGC2403_example.png", dpi=200)
    plt.close()


print("\nAll KSP falsification metrics (F1/F3/F5) are now computable from this output.")
print("Next step: plug the summary CSV into Bayesian model comparison vs. LCDM / MOND.")