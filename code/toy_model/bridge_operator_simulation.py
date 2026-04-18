from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class SimulationConfig:
    n_steps: int = 600
    dtau: float = 0.02
    dim: int = 8
    q_dim: int = 4
    seed: int = 7
    chi0: float = 1.8
    chi_floor: float = 0.2
    chi_decay: float = 2.0
    a0: float = 0.05
    a_growth: float = 2.5
    g_scale: float = 1.2
    leak_scale: float = 1.0
    out_csv: str = "bridge_operator_timeseries.csv"


# Simple matrix exponential via eigendecomposition. The generator is symmetric,
# so this is stable for the toy model and keeps dependencies minimal.
def expm_symmetric(mat: np.ndarray, scale: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    return vecs @ np.diag(np.exp(scale * vals)) @ vecs.T


def normalized_density(dim: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=(dim, dim))
    rho = x @ x.T
    rho /= np.trace(rho)
    return rho


def projector(indices: List[int], dim: int) -> np.ndarray:
    p = np.zeros((dim, dim))
    for idx in indices:
        p[idx, idx] = 1.0
    return p


def stress_profile(t: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    return cfg.chi_floor + (cfg.chi0 - cfg.chi_floor) * np.exp(-cfg.chi_decay * t)


def activation_profile(t: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    return cfg.a0 + (1.0 - cfg.a0) * (1.0 - np.exp(-cfg.a_growth * t))


def generator_matrix(a: float, chi: float, cfg: SimulationConfig) -> np.ndarray:
    # Split state space into quantum-like and classical-like blocks.
    q = cfg.q_dim
    c = cfg.dim - q

    # Classical-sector stabilization increases with a and decreases with chi.
    g_cl = cfg.g_scale * max(0.0, a - 0.35 * chi)

    # Leakage from classical to quantum rises with stress and weakens with activation.
    leak = cfg.leak_scale * max(0.0, chi - 0.55 * a)

    l = np.zeros((cfg.dim, cfg.dim))

    # Quantum mixing block (high baseline scrambling).
    for i in range(q):
        for j in range(q):
            if i == j:
                l[i, j] = -0.9
            elif abs(i - j) == 1:
                l[i, j] = 0.35

    # Classical stabilization block (metastable basin when g_cl > 0).
    for i in range(q, cfg.dim):
        l[i, i] = -0.35 - g_cl
        if i + 1 < cfg.dim:
            l[i, i + 1] = 0.12
            l[i + 1, i] = 0.12

    # Cross-block coupling acts as stress-driven de-structuring leakage.
    for i in range(q):
        for j in range(q, cfg.dim):
            coupling = 0.05 + 0.25 * leak
            l[i, j] = coupling
            l[j, i] = coupling

    # Add weak diagonal damping for accretive behavior.
    l -= 0.15 * np.eye(cfg.dim)

    return l


def retention(u: np.ndarray, rho: np.ndarray, p_x: np.ndarray, p_y: np.ndarray) -> float:
    num = np.trace(p_y @ u @ p_x @ rho @ p_x @ u.T)
    den = np.trace(p_x @ rho)
    if den <= 1e-12:
        return 0.0
    val = float(np.real(num / den))
    return max(0.0, min(1.0, val))


def gap_metric(l: np.ndarray, p_cl: np.ndarray) -> float:
    ident = np.eye(l.shape[0])
    leak_op = (ident - p_cl) @ l @ p_cl
    # Small leakage norm means better classical isolation.
    return float(np.linalg.norm(leak_op, ord=2))


def chi_crit_estimate(a: float, cfg: SimulationConfig) -> float:
    # Analytical proxy from g_cl = 0 in this toy model.
    # g_cl = g_scale * max(0, a - 0.35 chi) => threshold chi = a / 0.35.
    return a / 0.35


def run(cfg: SimulationConfig) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)

    p_q = projector(list(range(cfg.q_dim)), cfg.dim)
    p_cl = projector(list(range(cfg.q_dim, cfg.dim)), cfg.dim)

    rho = normalized_density(cfg.dim, rng)

    t_grid = np.linspace(0.0, 1.0, cfg.n_steps)
    chi_vals = stress_profile(t_grid, cfg)
    a_vals = activation_profile(t_grid, cfg)

    rows: List[Dict[str, float]] = []

    for step, t in enumerate(t_grid):
        a = float(a_vals[step])
        chi = float(chi_vals[step])

        l = generator_matrix(a, chi, cfg)
        u = expm_symmetric(-cfg.dtau * l, 1.0)

        # Propagate and renormalize.
        rho = u @ rho @ u.T
        rho /= np.trace(rho)

        rq_cl = retention(u, rho, p_q, p_cl)
        rcl_q = retention(u, rho, p_cl, p_q)
        cl_support = float(np.real(np.trace(p_cl @ rho)))

        leak_norm = gap_metric(l, p_cl)
        chi_crit = chi_crit_estimate(a, cfg)

        rows.append(
            {
                "step": float(step),
                "tau": float(step * cfg.dtau),
                "a": a,
                "chi": chi,
                "chi_over_chicrit": chi / max(chi_crit, 1e-9),
                "chi_crit": chi_crit,
                "cl_support": cl_support,
                "R_q_to_cl": rq_cl,
                "R_cl_to_q": rcl_q,
                "leak_norm": leak_norm,
            }
        )

    return rows


def write_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, float]]) -> str:
    arr = lambda k: np.array([r[k] for r in rows], dtype=float)
    ratio = arr("chi_over_chicrit")
    cl_support = arr("cl_support")
    r_qcl = arr("R_q_to_cl")
    leak = arr("leak_norm")

    crossover_idx = np.where((ratio[:-1] > 1.0) & (ratio[1:] <= 1.0))[0]
    if len(crossover_idx) > 0:
        i = int(crossover_idx[0] + 1)
        cross_msg = f"threshold crossing near tau={rows[i]['tau']:.3f} (chi/chicrit <= 1)"
    else:
        cross_msg = "no threshold crossing in window"

    lines = [
        "Bridge operator simulation summary",
        f"steps={len(rows)}",
        cross_msg,
        f"final cl_support={cl_support[-1]:.4f}",
        f"max R_q_to_cl={r_qcl.max():.4f}",
        f"final leak_norm={leak[-1]:.4f}",
    ]
    return "\n".join(lines)


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description="Toy KSL bridge-operator simulation")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dtau", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="bridge_operator_timeseries.csv")
    args = parser.parse_args()

    cfg = SimulationConfig()
    cfg.n_steps = args.steps
    cfg.dtau = args.dtau
    cfg.seed = args.seed
    cfg.out_csv = args.out
    return cfg


def main() -> None:
    cfg = parse_args()
    rows = run(cfg)
    out_path = Path(cfg.out_csv).resolve()
    write_csv(rows, out_path)
    print(summarize(rows))
    print(f"wrote_timeseries={out_path}")


if __name__ == "__main__":
    main()
