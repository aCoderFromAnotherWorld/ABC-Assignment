"""
Entry point for the ABC assignment.

Runs the Artificial Bee Colony on the 2D Ackley function, prints a cycle-by-cycle
convergence log, saves plots (convergence curve + solution on the Ackley surface)
to src/output/, then runs a classical-optimizer comparison plus reduced-cycle
ABC runs (20 and 25 cycles) for reference.
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# This finds the exact folder where main.py lives and adds it to Python's search path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Your original import on line 16 should come AFTER the code above
from ackley import BOUNDS, ackley, ackley_vec


# from ackley import BOUNDS, ackley, ackley_vec
from artificial_bee_colony import ABCConfig, ArtificialBeeColony
from scipy.optimize import minimize

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_main_abc(seed: int = 42):
    print("=" * 70)
    print("Artificial Bee Colony — 2D Ackley Function Global Minimum")
    print("=" * 70)
    cfg = ABCConfig(seed=seed)
    abc = ArtificialBeeColony(objective_fn=ackley_vec, config=cfg)
    result = abc.run(verbose=True)

    print("-" * 70)
    print(
        f"Finished after {result.cycles_run} cycle(s) "
        f"({'converged' if result.converged else 'reached max cycles'})"
    )
    print(
        f"Best solution found: x1 = {result.best_position[0]:.6f}, "
        f"x2 = {result.best_position[1]:.6f}"
    )
    print(
        f"Best value f(x1, x2) = {result.best_value:.8f}  (global minimum is 0 at (0,0))"
    )
    print("=" * 70)
    return result, cfg


def plot_convergence(result, path):
    cycles = np.arange(1, result.cycles_run + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(cycles, result.history_best, label="Best value (so far)", linewidth=2)
    plt.plot(cycles, result.history_mean, label="Mean food-source value", alpha=0.6)
    plt.axhline(
        0, color="gray", linestyle="--", linewidth=1, label="Global minimum (0)"
    )
    plt.xlabel("Cycle")
    plt.ylabel("Ackley function value")
    plt.title("ABC Convergence on the 2D Ackley Function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_surface_with_solution(result, path):
    low, high = BOUNDS
    grid = np.linspace(low, high, 250)
    X1, X2 = np.meshgrid(grid, grid)
    Z = ackley(X1, X2)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(X1, X2, Z, levels=50, cmap="viridis")
    fig.colorbar(contour, label="f(x1, x2)")
    ax.plot(
        0,
        0,
        marker="*",
        color="white",
        markersize=16,
        label="True global minimum (0,0)",
    )
    ax.plot(
        result.best_position[0],
        result.best_position[1],
        marker="o",
        color="red",
        markersize=10,
        label="ABC best solution",
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("2D Ackley Function — ABC Solution vs True Minimum")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def classical_comparison():
    """Compare against a classical derivative-free optimizer (Nelder-Mead)."""
    print("\nClassical optimizer comparison (scipy.optimize.minimize, Nelder-Mead)")
    x0 = np.array([4.0, -4.0])  # start far from the minimum
    res = minimize(lambda x: ackley(x[0], x[1]), x0, method="Nelder-Mead")
    print(f"  Start point:       x0 = ({x0[0]}, {x0[1]})")
    print(
        f"  Classical result:  x* = ({res.x[0]:.6f}, {res.x[1]:.6f}), f(x*) = {res.fun:.8f}"
    )
    return {"start": x0.tolist(), "x_star": res.x.tolist(), "f_star": float(res.fun)}


def reduced_cycle_abc_runs(seed: int = 42):
    """Run ABC with 20 and 25 cycles to see how quality scales with cycle count."""
    print("\nReduced-cycle ABC runs (for comparison)")
    results = {}
    for cycles in (20, 25):
        cfg = ABCConfig(cycles=cycles, seed=seed)
        result = ArtificialBeeColony(objective_fn=ackley_vec, config=cfg).run()
        print(
            f"  {cycles:3d} cycles -> best f = {result.best_value:.8f}, "
            f"x* = ({result.best_position[0]:.4f}, {result.best_position[1]:.4f})"
        )
        results[cycles] = {
            "best_value": result.best_value,
            "best_position": result.best_position.tolist(),
        }
    return results


def main():
    result, cfg = run_main_abc()

    convergence_path = os.path.join(OUTPUT_DIR, "convergence_curve.png")
    surface_path = os.path.join(OUTPUT_DIR, "ackley_surface_solution.png")
    plot_convergence(result, convergence_path)
    plot_surface_with_solution(result, surface_path)
    print(f"\nSaved plots:\n  {convergence_path}\n  {surface_path}")

    classical = classical_comparison()
    reduced = reduced_cycle_abc_runs()

    summary = {
        "abc_settings": {
            "swarm_size": cfg.swarm_size,
            "employed_bees (50% of swarm)": cfg.employed_bees,
            "onlooker_bees (50% of swarm)": cfg.onlooker_bees,
            "scouts": cfg.n_scouts,
            "limit": cfg.limit,
            "dimension": cfg.dimension,
            "cycles": cfg.cycles,
            "initialization": "x_ij = x_min,j + rand(0,1)*(x_max,j - x_min,j)",
            "neighbour_search": "v_ij = x_ij + phi*(x_ij - x_kj), phi=rand(-1,1), k!=i; keep better of f(x_i), f(v_i)",
            "onlooker_selection": "P(x_i) = fit_i / sum(fit), Roulette Wheel",
            "bounds": cfg.bounds,
        },
        "abc_result": {
            "best_position": result.best_position.tolist(),
            "best_value": result.best_value,
            "cycles_run": result.cycles_run,
            "converged": result.converged,
        },
        "classical_comparison": classical,
        "reduced_cycle_runs": reduced,
    }
    summary_path = os.path.join(OUTPUT_DIR, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved run summary: {summary_path}")


if __name__ == "__main__":
    main()
