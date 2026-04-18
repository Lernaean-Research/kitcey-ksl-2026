from math import erf
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "ksp_analysis" / "KSP_IRS_eigenmode_summary.csv"
DATA_DIR = ROOT / "dark_gravity" / "Rotmod_LTG"
OUT_DIR = ROOT / "ksp_analysis"

COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
N_SIM = 4000
RNG = np.random.default_rng(20260413)


def norm_sf(z: float) -> float:
    return 0.5 * (1.0 - erf(z / np.sqrt(2.0)))


def load_galaxy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=COLS, dtype=float, engine="python")
    df = df.dropna(how="all").copy()
    df["Vbar"] = np.sqrt(df["Vgas"] ** 2 + df["Vdisk"] ** 2 + df["Vbul"] ** 2)
    return df


def outer_subset(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 6:
        return df.iloc[0:0].copy()
    outer_n = max(3, len(df) // 3)
    return df.tail(outer_n).copy()


def simulate_null_metrics(outer: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean_vbar = outer["Vbar"].to_numpy()
    err_v = outer["errV"].fillna(0.0).to_numpy()
    err_v = np.where(err_v > 0, err_v, np.maximum(0.05 * mean_vbar, 2.0))
    draws = RNG.normal(loc=mean_vbar, scale=err_v, size=(N_SIM, len(outer)))
    delta_v2 = draws**2 - mean_vbar[np.newaxis, :] ** 2
    delta_outer_mean = delta_v2.mean(axis=1)
    negative_fraction = (delta_v2 < 0).mean(axis=1)
    return delta_outer_mean, negative_fraction


summary = pd.read_csv(SUMMARY)
results = []

for row in summary.itertuples(index=False):
    file_path = DATA_DIR / row.file
    if not file_path.exists():
        continue

    df = load_galaxy(file_path)
    outer = outer_subset(df)
    if outer.empty:
        continue

    observed_delta = float(np.mean(outer["Vobs"] ** 2 - outer["Vbar"] ** 2))
    observed_neg_fraction = float(np.mean((outer["Vobs"] ** 2 - outer["Vbar"] ** 2) < 0))
    sim_delta, sim_neg_fraction = simulate_null_metrics(outer)

    delta_mean = float(sim_delta.mean())
    delta_std = float(sim_delta.std(ddof=1)) if len(sim_delta) > 1 else np.nan
    neg_mean = float(sim_neg_fraction.mean())
    neg_std = float(sim_neg_fraction.std(ddof=1)) if len(sim_neg_fraction) > 1 else np.nan

    delta_z = (observed_delta - delta_mean) / delta_std if delta_std and np.isfinite(delta_std) and delta_std > 0 else np.nan
    neg_z = (observed_neg_fraction - neg_mean) / neg_std if neg_std and np.isfinite(neg_std) and neg_std > 0 else np.nan

    p_delta_left = float((np.sum(sim_delta <= observed_delta) + 1) / (len(sim_delta) + 1))
    p_neg_right = float((np.sum(sim_neg_fraction >= observed_neg_fraction) + 1) / (len(sim_neg_fraction) + 1))
    p_negative_under_null = float(np.mean(sim_delta < 0))

    results.append(
        {
            "Galaxy": row.Galaxy,
            "file": row.file,
            "outer_points": len(outer),
            "observed_delta_outer_mean": observed_delta,
            "observed_negative_fraction": observed_neg_fraction,
            "null_delta_mean": delta_mean,
            "null_delta_std": delta_std,
            "null_negative_fraction_mean": neg_mean,
            "null_negative_fraction_std": neg_std,
            "delta_z": delta_z,
            "negative_fraction_z": neg_z,
            "p_delta_left": p_delta_left,
            "p_negative_fraction_right": p_neg_right,
            "p_negative_under_null": p_negative_under_null,
        }
    )

res = pd.DataFrame(results)
res.to_csv(OUT_DIR / "rarefaction_noise_test.csv", index=False)

neg_obs = res[res["observed_delta_outer_mean"] < 0].copy()
neg_sig = neg_obs[neg_obs["p_delta_left"] < 0.05].copy()

observed_neg_count = int((res["observed_delta_outer_mean"] < 0).sum())
null_neg_probs = []
for _, row in res.iterrows():
    sim_prob = row["p_negative_under_null"]
    if np.isfinite(sim_prob):
        null_neg_probs.append(sim_prob)

expected_noise_neg_count = float(np.sum(null_neg_probs)) if null_neg_probs else 0.0

global_z = np.nan
if len(res) > 0:
    count_var = float(np.sum(np.array(null_neg_probs) * (1.0 - np.array(null_neg_probs)))) if null_neg_probs else 0.0
    if count_var > 0:
        global_z = (observed_neg_count - expected_noise_neg_count) / np.sqrt(count_var)

lines = []
lines.append("# Rarefaction Noise Test\n")
lines.append("This tests the null model that the outer-disk negative branch is produced by observational noise around baryonic closure, with no real rarefaction response.\n")
lines.append("## Global Result\n")
lines.append(f"- Galaxies tested: {len(res)}")
lines.append(f"- Observed negative-branch count: {observed_neg_count}")
lines.append(f"- Expected negative count from null noise baseline: {expected_noise_neg_count:.2f}")
if np.isfinite(global_z):
    lines.append(f"- Global excess/deficit z-score: {global_z:.3f}")
    lines.append(f"- One-sided tail estimate for excess negative count: {norm_sf(global_z):.4g}")
else:
    lines.append("- Global excess/deficit z-score: not available")
lines.append("")

lines.append("## Interpretation Rule\n")
lines.append("- If the observed negative-branch count is close to the null expectation and per-galaxy p-values are broadly unremarkable, the pattern is compatible with noise.")
lines.append("- If the observed negative branch is substantially more common or more coherent than the null expectation, the pattern is informative and merits a fitted rarefaction law.\n")

lines.append("## Strongest Negative-Branch Candidates\n")
if not neg_sig.empty:
    lines.append("```\n")
    lines.append(neg_sig.sort_values("p_delta_left")[["Galaxy", "observed_delta_outer_mean", "observed_negative_fraction", "delta_z", "p_delta_left"]].to_string(index=False))
    lines.append("\n```")
else:
    lines.append("No galaxies reached p_delta_left < 0.05 under this null.\n")

lines.append("\n## All Observed Negative-Branch Galaxies\n")
if not neg_obs.empty:
    lines.append("```\n")
    lines.append(neg_obs.sort_values("p_delta_left")[["Galaxy", "observed_delta_outer_mean", "observed_negative_fraction", "delta_z", "p_delta_left"]].to_string(index=False))
    lines.append("\n```")
else:
    lines.append("No observed negative-branch galaxies in the tested subset.\n")

(OUT_DIR / "Rarefaction_noise_test_note.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_DIR / 'rarefaction_noise_test.csv'}")
print(f"Wrote {OUT_DIR / 'Rarefaction_noise_test_note.md'}")
