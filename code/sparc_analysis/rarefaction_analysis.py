"""rarefaction_analysis.py

Three-stage analysis of the signed SPARC outer-disk residual.

Stage 1 — Tight null model
    Combines Vobs measurement noise (errV, Gaussian) with stellar
    mass-to-light ratio (Upsilon) uncertainty on Vbar (log-normal).
    Under SPARC conventions Vdisk is stored at Upsilon_disk = 0.5 M/L
    and Vbul at Upsilon_bul = 0.7 M/L.  A fractional dex-perturbation
    in Upsilon propagates as half that dex to the velocity contribution
    (V_star ∝ sqrt(Upsilon)), so:
        sigma_Vdisk_dex = 0.5 * 0.15  = 0.075 dex  (0.15 dex on Upsilon_disk)
        sigma_Vbul_dex  = 0.5 * 0.20  = 0.100 dex  (0.20 dex on Upsilon_bul)
    This makes the null distribution substantially wider than a noise-only
    baseline, so rejection is conservative.

Stage 2 — BH-FDR galaxy classification
    One-sided left-tail p-value for the negative branch and right-tail for
    the positive branch, both corrected by Benjamini-Hochberg FDR at 5%.
    Three classes:
        NEGATIVE_SUPPORT  — outer Vobs < Vbar beyond combined noise
        POSITIVE_SUPPORT  — outer Vobs > Vbar beyond combined noise
        NOISE_CONSISTENT  — cannot be distinguished from null

Stage 3 — Conditional rarefaction fit
    Fit R_rare = A * closure_margin^alpha on NEGATIVE_SUPPORT galaxies only.
    Outputs Pearson r (log-log), slope, amplitude, and p-value.
    A significant positive slope is required before any manuscript-level
    rarefaction law claim.
"""
from math import erf
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dark_gravity" / "Rotmod_LTG"
OUT_DIR = ROOT / "ksp_analysis"

COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
N_SIM = 5000
RNG = np.random.default_rng(20260413)

SIGMA_VDISK_DEX = 0.5 * 0.15   # 0.075 dex  → ~17% fractional spread in V_disk
SIGMA_VBUL_DEX  = 0.5 * 0.20   # 0.100 dex  → ~23% fractional spread in V_bul
BH_ALPHA = 0.05


# ─── helpers ──────────────────────────────────────────────────────────────────

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


def simulate_tight_null(outer: pd.DataFrame) -> np.ndarray:
    """
    Simulate DeltaV2 = Vobs_null^2 - Vbar_fiducial^2 under the combined null.

    Under the null, the true galaxy velocity equals the baryonically predicted
    Vbar (with Upsilon drawn from the prior); observations add Gaussian errV.
    DeltaV2 is referenced against the fiducial (stored) Vbar so that the
    observed DeltaV2 uses the same baseline.
    """
    n = len(outer)
    Vdisk    = outer["Vdisk"].to_numpy()
    Vbul     = outer["Vbul"].to_numpy()
    Vgas     = outer["Vgas"].to_numpy()
    Vobs     = outer["Vobs"].to_numpy()
    errV     = outer["errV"].fillna(0.0).to_numpy()
    errV     = np.where(errV > 0, errV, np.maximum(0.05 * Vobs, 2.0))
    Vbar_fid = outer["Vbar"].to_numpy()

    # Upsilon perturbation as multiplicative scale on velocity columns
    scale_disk = 10.0 ** RNG.normal(0.0, SIGMA_VDISK_DEX, size=(N_SIM, n))
    scale_bul  = 10.0 ** RNG.normal(0.0, SIGMA_VBUL_DEX,  size=(N_SIM, n))

    Vbar_sim = np.sqrt(
        Vgas[np.newaxis, :]   ** 2
        + (scale_disk * Vdisk[np.newaxis, :]) ** 2
        + (scale_bul  * Vbul[np.newaxis, :])  ** 2
    )

    # Null Vobs = Vbar_sim + measurement noise  (no intrinsic excess)
    eps_v    = RNG.normal(0.0, errV[np.newaxis, :], size=(N_SIM, n))
    Vobs_null = Vbar_sim + eps_v

    # DeltaV2 referenced against fiducial Vbar (mirrors the observed definition)
    delta_v2_null = Vobs_null ** 2 - Vbar_fid[np.newaxis, :] ** 2
    return delta_v2_null.mean(axis=1)


def bh_reject(pvals: np.ndarray, alpha: float = BH_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg rejection mask at FDR level alpha."""
    n = len(pvals)
    if n == 0:
        return np.zeros(n, dtype=bool)
    order = np.argsort(pvals)
    thresholds = (np.arange(1, n + 1) / n) * alpha
    reject_ordered = pvals[order] <= thresholds
    last = np.where(reject_ordered)[0]
    mask = np.zeros(n, dtype=bool)
    if len(last) > 0:
        mask[order[: last[-1] + 1]] = True
    return mask


def fit_powerlaw(x: np.ndarray, y: np.ndarray):
    """Log-log linear regression for y = A * x^alpha.
    Returns (A, alpha, r, p_value, n_points)."""
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    if len(xv) < 3:
        return np.nan, np.nan, np.nan, np.nan, int(valid.sum())
    lr = stats.linregress(np.log10(xv), np.log10(yv))
    return 10 ** lr.intercept, lr.slope, lr.rvalue, lr.pvalue, int(valid.sum())


# ─── Stage 1 — per-galaxy null tests ─────────────────────────────────────────
rotmod_files = sorted(DATA_DIR.glob("*_rotmod.dat"))
results = []

for fpath in rotmod_files:
    galaxy = fpath.stem.replace("_rotmod", "")
    df = load_galaxy(fpath)
    outer = outer_subset(df)
    if outer.empty:
        continue

    Vobs_outer = float(outer["Vobs"].mean())
    Vbar_outer = float(outer["Vbar"].mean())
    observed_delta   = float(np.mean(outer["Vobs"] ** 2 - outer["Vbar"] ** 2))
    observed_neg_frac = float(np.mean((outer["Vobs"] ** 2 - outer["Vbar"] ** 2) < 0))
    closure_margin   = (Vbar_outer ** 2 - Vobs_outer ** 2) / max(Vbar_outer ** 2, 1e-9)
    R_rare           = float(np.sqrt(max(Vbar_outer ** 2 - Vobs_outer ** 2, 0.0)))

    sim_delta = simulate_tight_null(outer)
    null_mean = float(sim_delta.mean())
    null_std  = float(sim_delta.std(ddof=1))
    delta_z   = float((observed_delta - null_mean) / null_std) if null_std > 0 else np.nan

    p_left  = float((np.sum(sim_delta <= observed_delta) + 1) / (N_SIM + 1))
    p_right = float((np.sum(sim_delta >= observed_delta) + 1) / (N_SIM + 1))
    p_neg_under_null = float(np.mean(sim_delta < 0))

    results.append({
        "Galaxy": galaxy,
        "file": fpath.name,
        "outer_points": len(outer),
        "Vobs_outer": Vobs_outer,
        "Vbar_outer": Vbar_outer,
        "observed_delta": observed_delta,
        "observed_neg_frac": observed_neg_frac,
        "closure_margin": closure_margin,
        "R_rare": R_rare,
        "null_mean": null_mean,
        "null_std": null_std,
        "delta_z": delta_z,
        "p_left": p_left,
        "p_right": p_right,
        "p_neg_under_null": p_neg_under_null,
    })

res = pd.DataFrame(results)

# ─── Stage 2 — BH-FDR classification ─────────────────────────────────────────
reject_neg = bh_reject(res["p_left"].to_numpy())
reject_pos = bh_reject(res["p_right"].to_numpy())

classes = []
for i in range(len(res)):
    if reject_neg[i]:
        classes.append("NEGATIVE_SUPPORT")
    elif reject_pos[i]:
        classes.append("POSITIVE_SUPPORT")
    else:
        classes.append("NOISE_CONSISTENT")

res["class"] = classes

# ─── Stage 3 — rarefaction fit on negative-support subset ────────────────────
neg_sup = res[res["class"] == "NEGATIVE_SUPPORT"].copy()
fit_A = fit_alpha = fit_r = fit_p = np.nan
fit_n = 0
if len(neg_sup) >= 3:
    fit_A, fit_alpha, fit_r, fit_p, fit_n = fit_powerlaw(
        neg_sup["closure_margin"].to_numpy(),
        neg_sup["R_rare"].to_numpy(),
    )

# Spearman for robustness (no log assumption)
spear_r = spear_p = np.nan
if len(neg_sup) >= 3:
    xn = neg_sup["closure_margin"].to_numpy()
    yn = neg_sup["R_rare"].to_numpy()
    valid = np.isfinite(xn) & np.isfinite(yn)
    if valid.sum() >= 3:
        res_sp = stats.spearmanr(xn[valid], yn[valid])
        spear_r = float(res_sp.statistic)
        spear_p = float(res_sp.pvalue)

# ─── Save CSV ─────────────────────────────────────────────────────────────────
res.to_csv(OUT_DIR / "rarefaction_analysis.csv", index=False)

# ─── Classification scatter plot ──────────────────────────────────────────────
class_colors = {
    "POSITIVE_SUPPORT": "#2196F3",
    "NOISE_CONSISTENT": "#9E9E9E",
    "NEGATIVE_SUPPORT": "#E53935",
}
fig, ax = plt.subplots(figsize=(7, 5))
for cls, grp in res.groupby("class"):
    ax.scatter(grp["closure_margin"], grp["R_rare"],
               c=class_colors.get(cls, "k"), label=cls, alpha=0.7, s=25)

if np.isfinite(fit_A) and fit_n >= 3:
    cm_range = np.linspace(
        neg_sup["closure_margin"].clip(lower=1e-4).min(),
        neg_sup["closure_margin"].max(), 200
    )
    ax.plot(cm_range, fit_A * cm_range ** fit_alpha,
            "r--", lw=1.5, label=f"fit: A={fit_A:.2f}, α={fit_alpha:.2f}")

ax.set_xlabel("Closure margin  $(V_{\\rm bar}^2 - V_{\\rm obs}^2)/V_{\\rm bar}^2$")
ax.set_ylabel("$R_{\\rm rare}$  [km/s]")
ax.set_title("Outer-disk classification (tight null, BH-FDR)")
ax.legend(fontsize=8, loc="upper left")
ax.axvline(0, color="k", lw=0.7, ls=":")
fig.tight_layout()
fig.savefig(OUT_DIR / "rarefaction_classification.png", dpi=150)
plt.close(fig)

# ─── Report ───────────────────────────────────────────────────────────────────
counts = res["class"].value_counts()
neg_count   = int(counts.get("NEGATIVE_SUPPORT", 0))
pos_count   = int(counts.get("POSITIVE_SUPPORT", 0))
noise_count = int(counts.get("NOISE_CONSISTENT", 0))

null_neg_probs = res["p_neg_under_null"].dropna().to_numpy()
expected_noise_neg = float(null_neg_probs.sum())
observed_neg_raw   = int((res["observed_delta"] < 0).sum())
count_var = float(np.sum(null_neg_probs * (1.0 - null_neg_probs)))
global_z  = (observed_neg_raw - expected_noise_neg) / np.sqrt(count_var) if count_var > 0 else np.nan

lines = []
lines += [
    "# Rarefaction Analysis: Tight Null + Classification + Conditional Fit\n",
    "## Stage 1 — Null Model\n",
    "The tight null model combines two independent uncertainty sources:\n",
    "1. **Velocity measurement noise** — each `V_obs` point is perturbed by its `errV` (Gaussian).",
    "2. **Baryonic model uncertainty** — `Upsilon_disk` and `Upsilon_bul` are drawn from",
    "   log-normal priors (0.15 dex and 0.20 dex respectively), scaling the SPARC `Vdisk`",
    "   and `Vbul` columns accordingly (`V_star ∝ Upsilon^0.5`, so 0.15 dex → 0.075 dex in velocity).",
    "",
    "Under this combined null, the simulated `DeltaV^2 = V_obs_null^2 - V_bar_fiducial^2` has a",
    "substantially wider distribution than measurement noise alone.  Rejecting this null is",
    "more demanding and therefore more conservative.\n",
    "BH-FDR correction is applied at α = 0.05 to both one-sided p-value vectors simultaneously.\n",
]

lines += [
    "## Stage 2 — Galaxy Classification\n",
    f"| Class              | Count |",
    f"|:-------------------|------:|",
    f"| POSITIVE_SUPPORT   | {pos_count:5d} |",
    f"| NEGATIVE_SUPPORT   | {neg_count:5d} |",
    f"| NOISE_CONSISTENT   | {noise_count:5d} |",
    f"| **Total analysed** | **{len(res)}** |\n",
    "### Population-level test (raw negative count vs tight-null expectation)\n",
    f"- Galaxies with negative outer `DeltaV^2` (observed): {observed_neg_raw}",
    f"- Expected under tight null: {expected_noise_neg:.2f}",
]
if np.isfinite(global_z):
    direction = "deficit" if global_z < 0 else "excess"
    lines.append(f"- Population z-score: {global_z:.3f}  →  population-level **{direction}** "
                 f"of negative residuals relative to the tight null")
lines.append("")

lines += [
    "## Stage 3 — Conditional Rarefaction Fit\n",
    f"Fit performed on the {len(neg_sup)} BH-significant NEGATIVE_SUPPORT galaxies.\n",
    "**Candidate law**: `R_rare = A × closure_margin^alpha`\n",
]
if np.isfinite(fit_A):
    lines += [
        f"| Parameter          | Value  |",
        f"|:-------------------|-------:|",
        f"| A (amplitude)      | {fit_A:.4g} km/s |",
        f"| alpha (exponent)   | {fit_alpha:.4f} |",
        f"| Pearson r (log-log)| {fit_r:.4f} |",
        f"| p-value (log-log)  | {fit_p:.4g} |",
        f"| Spearman rho       | {spear_r:.4f} |",
        f"| Spearman p         | {spear_p:.4g} |",
        f"| n_fit              | {fit_n} |\n",
    ]
    if np.isfinite(fit_p) and fit_p < 0.05 and fit_n >= 3 and fit_alpha > 0:
        lines += [
            "**Result: the closure-margin slope is statistically significant within the",
            "negative-support subset.**",
            "This provides conditional empirical grounding for a rarefaction branch law.",
            "The scaling exponent α should be cross-validated on an independent sample before",
            "any manuscript-level law claim.\n",
        ]
    else:
        lines += [
            "**Result: the closure-margin slope is NOT significant within the negative-support subset.**",
            "The rarefaction amplitude does not yet show a statistically reliable monotonic",
            "scaling with closure margin.  No manuscript-level rarefaction law is currently justified.\n",
        ]
else:
    lines.append(
        "Insufficient NEGATIVE_SUPPORT galaxies for a power-law fit.  "
        "No manuscript-level law is currently justified.\n"
    )

if not neg_sup.empty:
    lines += [
        "### NEGATIVE_SUPPORT galaxy table\n",
        "```",
        neg_sup.sort_values("p_left")[
            ["Galaxy", "observed_delta", "closure_margin", "R_rare", "delta_z", "p_left"]
        ].to_string(index=False),
        "```\n",
    ]

lines += [
    "## Interpretation Summary\n",
    "- **NEGATIVE_SUPPORT** galaxies have outer-disk kinematics that remain significantly",
    "  *below* baryonic predictions even after accounting for Υ uncertainty and measurement",
    "  noise.  These are the only defensible candidates for a rarefaction or kinematic-deficit",
    "  response.",
    "- **NOISE_CONSISTENT** galaxies cannot be distinguished from baryonic closure within",
    "  current measurement and model uncertainties.  They should not be used to argue for",
    "  *or* against a rarefaction law.",
    "- **No manuscript-level rarefaction law** should be proposed until the NEGATIVE_SUPPORT",
    "  sample shows a significant, positive-slope R_rare vs closure_margin relationship",
    "  (Stage 3 above) and survives cross-validation.\n",
    "## Output files\n",
    "- `ksp_analysis/rarefaction_analysis.csv`  — full per-galaxy table with class labels",
    "- `ksp_analysis/rarefaction_classification.png` — scatter plot of all three classes",
    "- `ksp_analysis/Rarefaction_analysis_note.md`  — this report\n",
]

(OUT_DIR / "Rarefaction_analysis_note.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_DIR / 'rarefaction_analysis.csv'}")
print(f"Wrote {OUT_DIR / 'rarefaction_classification.png'}")
print(f"Wrote {OUT_DIR / 'Rarefaction_analysis_note.md'}")
