"""
Artificial Bee Colony (ABC) optimization of the 2D Ackley function.

Follows the lecture slides (ABC_Algo.pdf) step by step:

    Step 1: Assign control parameters
            swarm size, employed bees (50% of swarm), onlookers (50% of swarm),
            scouts (1), limit, dimension of the problem
    Step 2: Initialize solutions (food sources) and calculate fitness (employed bees)
            x_ij = x_min,j + rand(0,1) * (x_max,j - x_min,j)
    Step 3: Repeat until the stopping criteria is met
            - Employed bee phase : v_ij = x_ij + phi * (x_ij - x_kj),
                                   phi = rand(-1,1), k = rand(1,n) != i,
                                   keep the better of f(x_i) and f(v_i)
            - Onlooker bee phase : P(x_i) = fit(x_i) / sum(fit), roulette wheel selection,
                                   then evolve & select exactly like the employed phase
            - Scout bee phase    : a source not improved for `limit` cycles is abandoned and
                                   re-initialized with the Step-2 equation
            - Memorize the best solution
    Step 4: Stop
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from ackley import BOUNDS, ackley_vec


@dataclass
class ABCConfig:
    swarm_size: int = 100  # total solutions (employed + onlooker bees)
    n_scouts: int = 1  # max scout bees per cycle
    dimension: int = 2  # dimension of the problem: [x1, x2]
    cycles: int = 100
    limit: int = 20  # cycles without improvement before a food source is abandoned
    bounds: tuple = BOUNDS
    convergence_tol: float = 1e-6  # stop early if best fitness <= this
    seed: int | None = None

    @property
    def employed_bees(self) -> int:  # 50% of the swarm = number of food sources n
        return self.swarm_size // 2

    @property
    def onlooker_bees(self) -> int:  # 50% of the swarm
        return self.swarm_size - self.employed_bees


@dataclass
class ABCResult:
    best_position: np.ndarray
    best_value: float
    history_best: list[float] = field(default_factory=list)  # best value per cycle
    history_mean: list[float] = field(default_factory=list)  # mean value per cycle
    cycles_run: int = 0
    converged: bool = False
    # Only populated when ArtificialBeeColony.run(log_details=True) is used.
    population_log: list[dict] = field(
        default_factory=list
    )  # one row per source per cycle
    events_log: list[dict] = field(default_factory=list)  # one row per bee action
    summary_log: list[dict] = field(default_factory=list)  # one row per cycle


class ArtificialBeeColony:
    """A small, readable ABC engine, specialized for minimizing a 2D function."""

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], np.ndarray] = ackley_vec,
        config: ABCConfig = None,
    ):
        self.objective_fn = objective_fn
        self.cfg = config or ABCConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ---------------------------------------------------------------- init
    def _random_sources(self, n: int) -> np.ndarray:
        """Food source initialization (also the scout equation):
        x_ij = x_min,j + rand(0,1) * (x_max,j - x_min,j)"""
        low, high = self.cfg.bounds
        return low + self.rng.random((n, self.cfg.dimension)) * (high - low)

    @staticmethod
    def _fitness(values: np.ndarray) -> np.ndarray:
        """Fitness of a solution from its Ackley value f(x). The slides use the fitness
        directly in P(x_i) = fit_i / sum(fit); because we MINIMIZE, we use the standard ABC
        transform 1/(1+f) (f>=0), so a smaller Ackley value gives a larger fitness."""
        return np.where(values >= 0, 1.0 / (1.0 + values), 1.0 + np.abs(values))

    # ------------------------------------------------------ neighbour search
    def _neighbour(
        self, foods: np.ndarray, i: int
    ) -> tuple[np.ndarray, int, int, float]:
        """Evolve a solution to its neighbourhood: v_ij = x_ij + phi * (x_ij - x_kj)
        with phi = rand(-1,1), j = a random dimension, k = rand(1,n) != i."""
        n = len(foods)
        j = int(self.rng.integers(0, self.cfg.dimension))
        k = int(self.rng.choice([m for m in range(n) if m != i]))
        phi = float(self.rng.uniform(-1.0, 1.0))
        candidate = foods[i].copy()
        candidate[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
        low, high = self.cfg.bounds
        candidate[j] = np.clip(candidate[j], low, high)
        return candidate, j, k, phi

    def _try_improve(self, foods, values, trials, i, phase, cycle, events, log_details):
        """Evolve source i into v_i, calculate f(v_i), and select the better of f(x_i), f(v_i)."""
        candidate, j, k, phi = self._neighbour(foods, i)
        new_value = float(self.objective_fn(candidate[None, :])[0])
        improved = new_value < values[i]
        if log_details:
            events.append(
                {
                    "cycle": cycle,
                    "phase": phase,
                    "source": i + 1,
                    "partner_source": k + 1,
                    "dimension": f"x{j + 1}",
                    "phi": phi,
                    "old_x1": float(foods[i, 0]),
                    "old_x2": float(foods[i, 1]),
                    "old_value": float(values[i]),
                    "new_x1": float(candidate[0]),
                    "new_x2": float(candidate[1]),
                    "new_value": new_value,
                    "improved": bool(improved),
                }
            )
        if improved:
            foods[i] = candidate
            values[i] = new_value
            trials[i] = 0
        else:
            trials[i] += 1

    # ----------------------------------------------------------------- log
    @staticmethod
    def _log_population(log, cycle, foods, values, trials):
        order = np.argsort(values)
        for rank, idx in enumerate(order, start=1):
            log.append(
                {
                    "cycle": cycle,
                    "source": int(idx) + 1,
                    "x1": float(foods[idx, 0]),
                    "x2": float(foods[idx, 1]),
                    "value": float(values[idx]),
                    "trials": int(trials[idx]),
                    "rank": rank,
                    "is_best": rank == 1,
                }
            )

    # --------------------------------------------------------------- run
    def run(self, verbose: bool = False, log_details: bool = False) -> ABCResult:
        cfg = self.cfg
        # Step 1: control parameters (cfg) -> n food sources = employed bees
        sn = cfg.employed_bees
        n_onlookers = cfg.onlooker_bees
        # Step 2: initialize solutions (food sources) and calculate fitness
        foods = self._random_sources(sn)
        values = self.objective_fn(foods)
        trials = np.zeros(sn, dtype=int)

        best_position = foods[np.argmin(values)].copy()
        best_value = float(values.min())

        history_best, history_mean = [], []
        population_log, events_log, summary_log = [], [], []
        converged = False
        cycle = 0

        if log_details:
            self._log_population(population_log, 0, foods, values, trials)

        # Step 3: repeat until the stopping criteria is met
        for cycle in range(1, cfg.cycles + 1):
            events_before = len(events_log)

            # Employed bee phase: each employed bee evolves its food source
            for i in range(sn):
                self._try_improve(
                    foods, values, trials, i, "employed", cycle, events_log, log_details
                )

            # Onlooker bee phase: P(x_i) = fit_i / sum(fit), roulette wheel selection
            probabilities = self._fitness(values)
            probabilities = probabilities / probabilities.sum()
            for _ in range(n_onlookers):
                i = int(self.rng.choice(sn, p=probabilities))
                self._try_improve(
                    foods, values, trials, i, "onlooker", cycle, events_log, log_details
                )

            # Scout bee phase: sources not improved for `limit` cycles are abandoned and
            # re-initialized with x_ij = x_min,j + rand(0,1)(x_max,j - x_min,j)
            n_scouts = 0
            for worst_trial in np.argsort(-trials)[: cfg.n_scouts]:
                worst_trial = int(worst_trial)
                if trials[worst_trial] <= cfg.limit:
                    break
                old = foods[worst_trial].copy()
                foods[worst_trial] = self._random_sources(1)[0]
                values[worst_trial] = float(
                    self.objective_fn(foods[worst_trial][None, :])[0]
                )
                trials[worst_trial] = 0
                n_scouts += 1
                if log_details:
                    events_log.append(
                        {
                            "cycle": cycle,
                            "phase": "scout",
                            "source": worst_trial + 1,
                            "old_x1": float(old[0]),
                            "old_x2": float(old[1]),
                            "new_x1": float(foods[worst_trial, 0]),
                            "new_x2": float(foods[worst_trial, 1]),
                            "new_value": float(values[worst_trial]),
                            "improved": None,
                        }
                    )

            # Memorize the best solution
            idx = int(np.argmin(values))
            if values[idx] < best_value:
                best_value = float(values[idx])
                best_position = foods[idx].copy()

            history_best.append(best_value)
            history_mean.append(float(values.mean()))

            if log_details:
                self._log_population(population_log, cycle, foods, values, trials)
                new_events = events_log[events_before:]
                summary_log.append(
                    {
                        "cycle": cycle,
                        "best_value": float(values.min()),
                        "mean_value": float(values.mean()),
                        "worst_value": float(values.max()),
                        "std_value": float(values.std()),
                        "best_x1": float(foods[np.argmin(values)][0]),
                        "best_x2": float(foods[np.argmin(values)][1]),
                        "employed_improvements": sum(
                            1
                            for e in new_events
                            if e["phase"] == "employed" and e["improved"]
                        ),
                        "onlooker_improvements": sum(
                            1
                            for e in new_events
                            if e["phase"] == "onlooker" and e["improved"]
                        ),
                        "scouts": n_scouts,
                        "best_value_so_far": best_value,
                    }
                )

            if verbose and (cycle % 10 == 0 or cycle == 1):
                print(
                    f"Cycle {cycle:3d} | best f = {best_value:.6f} | "
                    f"x* = ({best_position[0]: .4f}, {best_position[1]: .4f}) | "
                    f"mean f = {values.mean():.4f}"
                )

            # Stopping criteria: close enough to the global minimum (0)
            if best_value <= cfg.convergence_tol:
                converged = True
                break

        # Step 4: stop
        return ABCResult(
            best_position=best_position,
            best_value=best_value,
            history_best=history_best,
            history_mean=history_mean,
            cycles_run=cycle,
            converged=converged,
            population_log=population_log,
            events_log=events_log,
            summary_log=summary_log,
        )
