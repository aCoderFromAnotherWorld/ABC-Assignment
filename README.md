# ABC-Assignment

Artificial Bee Colony (ABC) implementation that finds the **global minimum of the 2D Ackley function** using the swarm-intelligence algorithm Artificial Bee Colony.

## Problem

Minimize the 2D Ackley function:

```py
f(x1, x2) = -a * exp(-b * sqrt(0.5 * (x1^2 + x2^2))) - exp(0.5 * (cos(c*x1) + cos(c*x2))) + a + exp(1)
```

with `a = 20`, `b = 0.2`, `c = 2π`, over `-5 <= x1, x2 <= 5`. The known global minimum is `f(0, 0) = 0`.

## ABC Algorithm (as in the lecture slides, `docs/ABC_Algo.pdf`)

The code in `src/artificial_bee_colony.py` follows the lecture's steps and terminology:

| Lecture step                                                                                                                                   | In the code                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Step 1: Assign control parameters** — swarm size, employed bees (50% of swarm), onlookers (50% of swarm), scouts, limit, dimension           | `ABCConfig`: swarm size 100 → 50 employed + 50 onlookers, 1 scout, limit 20, dimension 2, 100 cycles |
| **Step 2: Initialize solutions and calculate fitness** — `x_ij = x_min,j + rand(0,1)(x_max,j − x_min,j)` (number of solutions = employed bees) | `_random_sources()`, then `objective_fn` on all 50 food sources                                      |
| **Step 3: Repeat until stopping criteria**                                                                                                     | the `for cycle` loop                                                                                 |
| ↳ Employed bees: `v_ij = x_ij + φ(x_ij − x_kj)`, φ = rand(−1,1), k = rand(1,n) ≠ i; keep the better of f(x_i), f(v_i)                          | `_neighbour()` + `_try_improve()` (phase `"employed"`)                                               |
| ↳ Onlookers: `P(x_i) = fit_i / Σ fit`, roulette wheel, then evolve/select as above                                                             | roulette wheel in `run()` + `_try_improve()` (phase `"onlooker"`)                                    |
| ↳ Scouts: a source not improved within `limit` cycles is abandoned and re-initialized with the Step-2 equation                                 | scout block in `run()`                                                                               |
| ↳ Memorize the best solution                                                                                                                   | `best_position` / `best_value`                                                                       |
| **Step 4: Stop** — 100 cycles, or best value ≤ 1e-6                                                                                            | end of `run()`                                                                                       |

Search domain `[-5, 5]` for both dimensions. Because we _minimize_, the fitness used in `P(x_i)` is `1/(1+f)` so a lower Ackley value gets a bigger roulette slice.

## Project Structure

```
ABC-Assignment/
├── README.md
└── src/
    ├── ackley.py                   # 2D Ackley function (scalar + vectorized)
    ├── artificial_bee_colony.py    # ABC engine: employed / onlooker / scout phases
    ├── main.py                     # Runs ABC, saves plots + JSON summary
    ├── test_abc.py                 # Unit tests (pytest)
    ├── generate_detailed_report.py # Per-cycle tables + plots + GIF + Excel
    ├── multi_seed_check.py         # 30-seed robustness check
    ├── requirements.txt
    └── output/                     # generated results (incl. detailed/)
```

## How to Run

```bash
cd src
pip install -r requirements.txt
python main.py                       # convergence log, plots, classical + 20/25-cycle comparison
python -m pytest test_abc.py -v      # tests
python generate_detailed_report.py   # per-cycle CSVs, PNGs, evolution.gif, ABC_Detailed_Report.xlsx
python multi_seed_check.py           # 30 seeds: success rate and cycles used
```

## Example Result (seed 42)

```
Finished after 52 cycle(s) (converged)
Best solution found: x1 = 0.000000, x2 = -0.000000
Best value f(x1, x2) = 0.00000034   (global minimum is 0 at (0,0))
```

Reduced-cycle runs: 20 cycles → f ≈ 0.00049, 25 cycles → f ≈ 0.00013. Nelder-Mead started at (4, −4) gets stuck in a local minimum (f ≈ 10.998), showing why a global method is useful on Ackley.

## Robustness (30 seeds)

All 30 seeds (0–29) reached f ≤ 1e-6 and stopped early, using 44.8 cycles on average (range 32–52); the worst final value was 9.9e-7. See `src/output/multi_seed_summary.json` and `multi_seed_convergence.png`.

## Understanding the ABC Steps (for the exam / presentation)

1. **Control parameters** — swarm 100 = 50 employed + 50 onlooker bees, 1 scout, limit 20, dimension 2.
2. **Initialization** — 50 random food sources in `[-5, 5]` via `x_min + rand(0,1)(x_max − x_min)`; fitness calculated.
3. **Employed bees** — each evolves its source to a neighbour `v_i` using a random partner `k ≠ i`, φ ∈ (−1,1); the better of `x_i`, `v_i` is kept (otherwise its trial counter grows).
4. **Onlookers** — probability per source `P(x_i)`, roulette wheel picks sources, which are evolved/selected the same way (good regions get more bees).
5. **Scouts** — a source stuck for more than `limit` cycles is abandoned and re-initialized randomly.
6. **Memorize** the best solution; repeat until the stopping criteria; **stop**.

## Notes

- `python` in this setup is a shell alias to `/opt/homebrew/bin/python3.13`; it was run with Python 3.13.
