"""
Generates a FULL per-cycle audit trail of the ABC run: one table (CSV) and one
plot (PNG) for every cycle, plus consolidated CSV/Excel files and an animated
GIF showing the food sources converging over time.

Run with:
    cd src
    python generate_detailed_report.py
"""

import os
import shutil

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ackley import BOUNDS, ackley, ackley_vec
from artificial_bee_colony import ABCConfig, ArtificialBeeColony

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "detailed")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def reset_output_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def run_abc(seed: int = 42):
    print("Running ABC with full per-cycle logging enabled...")
    cfg = ABCConfig(seed=seed)
    abc = ArtificialBeeColony(objective_fn=ackley_vec, config=cfg)
    result = abc.run(verbose=True, log_details=True)
    print(
        f"Done: {result.cycles_run} cycles "
        f"({'converged' if result.converged else 'reached max cycles'}), "
        f"best value = {result.best_value:.8f}"
    )
    return result


def save_per_cycle_tables(pop_df: pd.DataFrame):
    print("Writing one CSV table per cycle to output/detailed/tables/ ...")
    for cycle, df in pop_df.groupby("cycle"):
        df.drop(columns="cycle").to_csv(
            os.path.join(TABLES_DIR, f"cycle_{cycle:03d}.csv"), index=False
        )


def save_per_cycle_plots(pop_df: pd.DataFrame):
    print("Rendering one plot per cycle to output/detailed/plots/ ...")
    low, high = BOUNDS
    grid = np.linspace(low, high, 150)
    X1, X2 = np.meshgrid(grid, grid)
    Z = ackley(X1, X2)

    frame_paths = []
    for cycle in sorted(pop_df["cycle"].unique()):
        cdf = pop_df[pop_df["cycle"] == cycle]
        best = cdf[cdf["is_best"]]
        rest = cdf[~cdf["is_best"]]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        contour = ax.contourf(X1, X2, Z, levels=40, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="f(x1, x2)")
        ax.plot(
            0,
            0,
            marker="*",
            color="white",
            markersize=14,
            linestyle="None",
            label="True global minimum",
        )
        ax.scatter(
            rest["x1"],
            rest["x2"],
            s=22,
            color="red",
            alpha=0.8,
            edgecolors="black",
            linewidths=0.3,
            label="Food sources",
        )
        ax.scatter(
            best["x1"],
            best["x2"],
            s=70,
            color="orange",
            marker="D",
            edgecolors="black",
            linewidths=0.6,
            label="Best source",
        )
        ax.set_title(f"Cycle {cycle} — best f = {cdf['value'].min():.5f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(BOUNDS)
        ax.set_ylim(BOUNDS)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"cycle_{cycle:03d}.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        frame_paths.append(path)
    return frame_paths


def main():
    reset_output_dirs()
    result = run_abc()

    pop_df = pd.DataFrame(result.population_log)
    events_df = pd.DataFrame(result.events_log)
    summary_df = pd.DataFrame(result.summary_log)

    pop_df.to_csv(os.path.join(OUTPUT_DIR, "population_all_cycles.csv"), index=False)
    events_df.to_csv(os.path.join(OUTPUT_DIR, "events_all_cycles.csv"), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_all_cycles.csv"), index=False)

    save_per_cycle_tables(pop_df)
    frame_paths = save_per_cycle_plots(pop_df)

    gif_path = os.path.join(OUTPUT_DIR, "evolution.gif")
    print(f"Building animated GIF ({len(frame_paths)} frames) -> {gif_path}")
    imageio.mimsave(gif_path, [imageio.imread(p) for p in frame_paths], fps=6)

    excel_path = os.path.join(OUTPUT_DIR, "ABC_Detailed_Report.xlsx")
    print(f"Writing consolidated Excel workbook -> {excel_path}")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary (per cycle)", index=False)
        pop_df.to_excel(writer, sheet_name="Food sources (all cycles)", index=False)
        events_df.to_excel(writer, sheet_name="Bee events (all cycles)", index=False)

    print("\nAll detailed, per-cycle outputs are in: src/output/detailed/")


if __name__ == "__main__":
    main()
