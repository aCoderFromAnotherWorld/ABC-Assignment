"""
Small sanity-check test suite.

Run with:  python -m pytest test_abc.py -v
"""

import numpy as np
import pytest
from ackley import ackley, ackley_vec
from artificial_bee_colony import ABCConfig, ArtificialBeeColony


def test_ackley_global_minimum_is_zero_at_origin():
    assert ackley(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_ackley_vec_matches_scalar():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [-2.5, 4.0]])
    assert np.allclose(ackley_vec(pts), [ackley(x, y) for x, y in pts])


def test_abc_best_value_is_non_increasing():
    """Greedy selection + memorized best means the best-so-far never gets worse."""
    result = ArtificialBeeColony(config=ABCConfig(cycles=30, seed=1)).run()
    h = result.history_best
    assert all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1))


def test_abc_converges_close_to_zero():
    result = ArtificialBeeColony(config=ABCConfig(cycles=100, seed=7)).run()
    assert result.best_value < 0.1


def test_food_sources_stay_within_bounds():
    cfg = ABCConfig(cycles=50, seed=3)
    result = ArtificialBeeColony(config=cfg).run(log_details=True)
    low, high = cfg.bounds
    for row in result.population_log:
        assert low <= row["x1"] <= high and low <= row["x2"] <= high


def test_scout_replaces_exhausted_sources():
    """With limit=0 every source that fails once is abandoned, so scouts must fire."""
    result = ArtificialBeeColony(config=ABCConfig(cycles=10, limit=0, seed=2)).run(
        log_details=True
    )
    assert any(e["phase"] == "scout" for e in result.events_log)


def test_reproducible_with_same_seed():
    a = ArtificialBeeColony(config=ABCConfig(seed=5)).run()
    b = ArtificialBeeColony(config=ABCConfig(seed=5)).run()
    assert a.best_value == b.best_value


def test_swarm_split_follows_lecture_parameters():
    cfg = ABCConfig()
    assert cfg.employed_bees == cfg.onlooker_bees == cfg.swarm_size // 2 == 50


def test_onlooker_probabilities_sum_to_one_and_favour_low_values():
    abc = ArtificialBeeColony(config=ABCConfig(seed=0))
    fit = abc._fitness(np.array([0.1, 5.0, 12.0]))
    p = fit / fit.sum()
    assert p.sum() == pytest.approx(1.0) and p[0] > p[1] > p[2]
