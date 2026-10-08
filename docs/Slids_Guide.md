# Slides Guide — ABC on the 2D Ackley Function

Everything you need to edit the presentation by hand: what each slide says, every number on it, and where that number comes from.

- **Deck file:** `docs/ABC_Ackley.pptx` (edit this one; open it in PowerPoint, Keynote or Google Slides).
- **Fonts:** install Oswald and Public Sans (Google Fonts) first, or the text may reflow with a default font.
- **Length:** 12 slides, about 8 minutes (about 40 s per slide). Each slide already has speaker notes.
- **Still to fill in:** the cover placeholder `[Group name · member names]`.
- **Source of the algorithm terms and equations:** `docs/ABC_Algo.pdf` (lecture slides).

---

## 1. Look and feel (to keep new edits consistent)

| Item             | Value                                                                           |
| ---------------- | ------------------------------------------------------------------------------- |
| Dark background  | `#1C1A16` (cover, slide 4, slide 12)                                            |
| Light background | `#FAF5E8` (all other slides)                                                    |
| Accent (honey)   | `#F2B632` on dark slides; `#8A5A00` on light slides (for contrast)              |
| Body text        | `#3D382D` on light, `#FAF5E8` / `#CFC6AE` on dark                               |
| Cards            | `#FFFCF3` with `#E6DCC0` border (light); `#2A2720` with `#4A4436` border (dark) |
| Headings font    | Oswald 600 (h2 = 76 px)                                                         |
| Body font        | Public Sans (body text 30–34 px; footers 24 px)                                 |
| Slide size       | 1920 × 1080, margins 128 px                                                     |

---

## 2. Slide by slide

### Slide 1 — Cover

- Title: **Artificial Bee Colony**
- Subtitle: Finding the global minimum of the 2D Ackley function
- Tag line: Machine Learning · Assignment
- Placeholder to replace: `[Group name · member names]`

### Slide 2 — The problem: minimize the 2D Ackley function

- Formula (a = 20, b = 0.2, c = 2π):
  `f(x1, x2) = −a·exp(−b·√(0.5(x1² + x2²))) − exp(0.5(cos(c·x1) + cos(c·x2))) + a + e`
- Known global minimum: **f(0, 0) = 0**
- Search range: **−5 ≤ x1, x2 ≤ 5**
- Parameters: **a = 20, b = 0.2, c = 2π**
- Point: many local minima around one global minimum, so it is a classic test for optimizers.

### Slide 3 — Three groups of bees (from the lecture)

- **Employed bees:** stay on a food source and search the neighbourhood of that source.
- **Onlooker bees:** get food-source information from the employed bees and choose which sources to gather nectar from.
- **Scout bees:** when a food source is abandoned, its bee becomes a scout and searches for a new source.
- Mapping: food source = candidate solution `[x1, x2]`; nectar quality = fitness (Ackley value).
- Background: ABC was defined by Dervis Karaboga in 2005, inspired by honey bee foraging; it is a swarm-based metaheuristic.

### Slide 4 — Four steps (lecture flow)

1. Assign control parameters
2. Initialize solutions and calculate fitness (employed bees)
3. Repeat until the stopping criteria is met: employed bees → onlooker bees → scout bees → memorize the best solution
4. Stop

### Slide 5 — Steps 1 and 2: parameters and initialization

| Control parameter            | Value     |
| ---------------------------- | --------- |
| Swarm size                   | 100       |
| Employed bees (50% of swarm) | 50        |
| Onlooker bees (50% of swarm) | 50        |
| Scouts                       | 1         |
| Limit                        | 20 cycles |
| Dimension                    | 2         |
| Max cycles                   | 100       |

- Initialization equation: `x(i,j) = x_min,j + rand(0,1) · (x_max,j − x_min,j)`
- i = 1…50 food sources (one per employed bee); j = 1…2 dimensions (x1, x2); x_min = −5, x_max = +5.

### Slide 6 — Step 3: employed and onlooker bees

- **Employed bee phase:** `v(i,j) = x(i,j) + φ · (x(i,j) − x(k,j))`
  - φ = rand(−1, 1); k = random source, k ≠ i; j = one random dimension
  - Keep the better of f(x_i) and f(v_i)
- **Onlooker bee phase:** `P(x_i) = fit(x_i) / Σ fit(x_i)`
  - Roulette wheel picks a source by P, then evolves it exactly like an employed bee
  - We minimize, so `fit = 1/(1+f)`: lower Ackley value → bigger slice

### Slide 7 — Scouts, memory and stopping

- **Scout phase:** a source not improved for limit = 20 cycles is abandoned; its bee becomes a scout and restarts with the initialization equation (same as slide 5).
- **Memorize:** the best solution so far is stored every cycle, so the best value never gets worse.
- **Stop:** when f ≤ 1e-6 or cycle = 100.

### Slide 8 — The swarm moves to (0, 0)

Three plots from `src/output/detailed/plots/`:

| Image           | Caption on slide            | Value      |
| --------------- | --------------------------- | ---------- |
| `cycle_000.png` | Cycle 0 · best f = 4.3193   | 4.319263   |
| `cycle_008.png` | Cycle 8 · best f = 0.0119   | 0.011922   |
| `cycle_025.png` | Cycle 25 · best f = 0.00013 | 0.00012836 |

- Orange diamond = best food source, white star = true minimum (0, 0), red dots = other food sources.
- Any other cycle (0–52) can be swapped in from the same folder; its matching table is in `src/output/detailed/tables/cycle_XXX.csv`.

### Slide 9 — Convergence to the global minimum (seed 42)

- Image: `src/output/convergence_curve.png`
- **52** cycles to reach f ≤ 1e-6
- **3.4e-7** best value found (exact: 0.00000034)
- **(0, 0)**: x1 = 0.000000, x2 = −0.000000
- Mean food-source value also falls, so the whole swarm converges, not only the best bee.

Convergence log (best f so far): cycle 1 → 4.319263 · 10 → 0.011292 · 20 → 0.000491 · 30 → 0.000128 · 40 → 0.000016 · 50 → 0.000001

### Slide 10 — ABC against a classical optimizer (seed 42)

| Method                    | Best value f               | Outcome                  |
| ------------------------- | -------------------------- | ------------------------ |
| ABC, 20 cycles            | 0.00049 (exact 0.00049128) | Very close to 0          |
| ABC, 25 cycles            | 0.00013 (exact 0.00012836) | Closer still             |
| ABC, full run (52 cycles) | 0.00000034                 | Converged                |
| Nelder-Mead from (4, −4)  | 10.998 (exact 10.99826214) | Stuck in a local minimum |

- Nelder-Mead end point: x\* = (3.983079, −3.983104). It is a local method and stays in the basin where it starts.
- This covers the brief's request to "apply a classical algorithm and run for 20/25 iterations".

### Slide 11 — Robustness: 30 seeds (0–29)

- Image: `src/output/multi_seed_convergence.png`
- **30 / 30** seeds converged to f ≤ 1e-6
- **44.8** mean cycles used (median 45, range 32–52)
- **9.9e-7** worst final value over all seeds (mean 5.8e-7)
- After 20 cycles: mean best f 1.52e-3, worst 6.23e-3. After 25 cycles: mean 3.97e-4, worst 1.10e-3.

### Slide 12 — Strengths, limits and uses (lecture)

- **Advantages:** few control parameters; fast convergence; both exploration and exploitation.
- **Limitation:** search space is limited by the initial solutions.
- **Applications:** welded beam and coil spring design; digital filter optimization; fuzzy control design; data clustering (K-means local optimum); robot control.

---

## 3. Where every number comes from

| What                                | File                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| Main run log, 52 cycles, 3.4e-7     | `src/output/run_main_output.txt`, `src/output/run_summary.json`              |
| Settings table                      | `src/artificial_bee_colony.py` (`ABCConfig`)                                 |
| Nelder-Mead and 20/25-cycle results | `src/output/run_summary.json`                                                |
| 30-seed numbers                     | `src/output/multi_seed_summary.json`, `src/output/run_multi_seed_output.txt` |
| Per-cycle tables and plots          | `src/output/detailed/`                                                       |
| Tests (9 passed)                    | `src/output/run_pytest_output.txt`                                           |

All results use `seed = 42` unless stated. Re-running with the same seed reproduces the same numbers.

---

## 4. Re-running the code

```bash
cd src
python main.py                       # main run, plots, JSON summary, Nelder-Mead and 20/25-cycle comparison
python -m pytest test_abc.py -v      # 9 tests
python generate_detailed_report.py   # per-cycle CSVs, PNGs, evolution.gif, Excel
python multi_seed_check.py           # 30-seed check
```

---

## 5. Likely questions (exam / viva)

- **What are the three bee types?** Employed (search around their own source), onlookers (choose sources by probability), scouts (replace abandoned sources with random ones).
- **What does `limit` do?** After 20 cycles without improvement a source is abandoned and re-initialized, which restores exploration.
- **How do onlookers choose?** Roulette wheel with P(x_i) = fit_i / Σ fit; for minimization fit = 1/(1+f).
- **Why does the best value never get worse?** Greedy selection (only a better neighbour replaces a source) plus the memorized best.
- **Why did Nelder-Mead fail?** It is a local method; Ackley has many local minima, and it started at (4, −4).
- **Where is ABC used?** Mechanical design, digital filters, fuzzy control, clustering, robot control.

---

## 6. Things to double-check before presenting

- Fill in the cover placeholder with the group name and members.
- Slides 5–8 are the densest; check that nothing overflows after your edits.
- The subscript-style names (`x_min,j`, `x_i`) are written as plain text with underscores; replace them with proper subscripts if your editor allows it.
