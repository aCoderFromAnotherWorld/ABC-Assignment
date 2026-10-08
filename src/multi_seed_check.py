"""
Robustness check: run ABC with many different random seeds and report how
reliably (and how fast) it reaches the global minimum of the 2D Ackley function.

Run with:
    cd src
    python multi_seed_check.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ackley import ackley_vec
from artificial_bee_colony import ABCConfig, ArtificialBeeColony

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
N_SEEDS = 30
SUCCESS_TOL = 1e-3  # a run "succeeds" if its best value ends at or below this


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs = []
    for seed in range(N_SEEDS):
        r = ArtificialBeeColony(ackley_vec, ABCConfig(seed=seed)).run()
        runs.append(r)

    best = np.array([r.best_value for r in runs])
    cycles = np.array([r.cycles_run for r in runs])
    converged = np.array([r.converged for r in runs])
    success = best <= SUCCESS_TOL

    print(f"ABC multi-seed check ({N_SEEDS} seeds, default settings, 100 cycles max)")
    print("-" * 60)
    print(f"Converged to f <= 1e-6 (early stop): {converged.sum()}/{N_SEEDS}")
    print(f"Final best f <= {SUCCESS_TOL:g}:            {success.sum()}/{N_SEEDS}")
    print(
        f"Final best f   : mean {best.mean():.2e}, median {np.median(best):.2e}, worst {best.max():.2e}"
    )
    print(
        f"Cycles used    : mean {cycles.mean():.1f}, median {np.median(cycles):.0f}, "
        f"min {cycles.min()}, max {cycles.max()}"
    )
    for c in (20, 25):
        v = np.array([r.history_best[min(c, r.cycles_run) - 1] for r in runs])
        print(f"Best f after {c} cycles: mean {v.mean():.2e}, worst {v.max():.2e}")

    summary = {
        "n_seeds": N_SEEDS,
        "converged_1e-6": int(converged.sum()),
        f"final_best_le_{SUCCESS_TOL:g}": int(success.sum()),
        "best_value": {
            "mean": float(best.mean()),
            "median": float(np.median(best)),
            "worst": float(best.max()),
        },
        "cycles_used": {
            "mean": float(cycles.mean()),
            "median": float(np.median(cycles)),
            "min": int(cycles.min()),
            "max": int(cycles.max()),
        },
        "per_seed": [
            {"seed": s, "best_value": float(b), "cycles": int(c)}
            for s, (b, c) in enumerate(zip(best, cycles))
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "multi_seed_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 5))
    for r in runs:
        h = np.maximum(r.history_best, 1e-12)
        plt.semilogy(np.arange(1, len(h) + 1), h, alpha=0.5, linewidth=1)
    plt.axhline(
        1e-6, color="gray", linestyle="--", linewidth=1, label="Stop threshold (1e-6)"
    )
    plt.xlabel("Cycle")
    plt.ylabel("Best Ackley value (log scale)")
    plt.title(f"ABC convergence across {N_SEEDS} random seeds")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "multi_seed_convergence.png"), dpi=150)
    plt.close()
    print("\nSaved: output/multi_seed_summary.json, output/multi_seed_convergence.png")


if __name__ == "__main__":
    main()
