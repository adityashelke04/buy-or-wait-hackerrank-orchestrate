from pathlib import Path

from evaluation.calibrate import sweep

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"


def test_sweep_returns_one_result_per_grid_point_sorted_best_first():
    results = sweep(DATASET, {"estimator": ["mean", "p75", "max"]})
    assert len(results) == 3
    scores = [r[1].overall for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all("estimator" in r[0] for r in results)


def test_sweep_covers_the_full_cartesian_product():
    results = sweep(DATASET, {"estimator": ["mean", "max"],
                              "min_observations": [2, 3]})
    assert len(results) == 4
    combos = {(r[0]["estimator"], r[0]["min_observations"]) for r in results}
    assert combos == {("mean", 2), ("mean", 3), ("max", 2), ("max", 3)}


def test_sweep_is_deterministic():
    grid = {"estimator": ["mean", "max"]}
    a = sweep(DATASET, grid)
    b = sweep(DATASET, grid)
    assert [r[0] for r in a] == [r[0] for r in b]
    assert [r[1].overall for r in a] == [r[1].overall for r in b]


def test_sweep_scores_all_25_samples():
    results = sweep(DATASET, {"estimator": ["p75"]})
    assert results[0][1].rows == 25
