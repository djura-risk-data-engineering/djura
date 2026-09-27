import numpy as np
import pytest

from djura.record_selection.numba_utils import greedy_algorithm, \
    greedy_algorithm_fast


def make_case(seed, num_rec=50, n_im=25, n_db=300):
    """Random greedy-step inputs shaped as ``_gcim_select`` passes them"""
    rng = np.random.default_rng(seed)
    mu = rng.normal(-2.0, 1.0, n_im)
    sigma = rng.uniform(0.4, 0.8, n_im)
    # A wider database than the target, so that some values fall outside
    # the +/-3 sigma bounds
    scaled_imi = mu + 1.5 * sigma * rng.normal(size=(num_rec - 1, n_im))
    ln_imi_db = mu + 1.5 * sigma * rng.normal(size=(n_db, n_im))
    sf = rng.uniform(0.5, 2.0, n_db)
    db_idxs = np.arange(n_db)
    rec_id = rng.choice(n_db, num_rec - 1, replace=False)
    alpha = rng.integers(1, 3, n_im)
    return dict(
        scaled_imi=scaled_imi, sf=sf, mu_imi=mu, sigma_imi=sigma,
        rec_id=rec_id, ln_imi_db=ln_imi_db, num_rec=num_rec,
        error_weights=np.array([1.0, 2.0, 0.0]), db_idxs=db_idxs,
        alpha=alpha, im_weights=rng.uniform(0.5, 1.0, n_im),
        selected_record_id=-1,
    )


def call(func, case, penalty, dev_min):
    c = case
    return func(
        c["scaled_imi"], c["sf"], c["mu_imi"], c["sigma_imi"], c["rec_id"],
        c["ln_imi_db"], c["num_rec"], c["error_weights"], penalty,
        c["db_idxs"], c["alpha"], c["im_weights"], c["selected_record_id"],
        dev_min)


def reference(case, penalty, dev_min):
    """Plain NumPy greedy step, penalising values outside +/-3 sigma"""
    c = case
    upper = c["mu_imi"] + 3.0 * c["sigma_imi"]
    lower = c["mu_imi"] - 3.0 * c["sigma_imi"]
    ew = c["error_weights"]
    selected = c["selected_record_id"]
    for imi, sf, db_idx in zip(c["ln_imi_db"], c["sf"], c["db_idxs"]):
        trial = np.vstack((c["scaled_imi"], imi + np.log(sf ** c["alpha"])))
        dev_mean = (trial.mean(axis=0) - c["mu_imi"]) * c["im_weights"]
        dev_std = (trial.std(axis=0) - c["sigma_imi"]) * c["im_weights"]
        dev = ew[0] * np.sum(dev_mean ** 2) + ew[1] * np.sum(dev_std ** 2)
        if np.any(c["rec_id"] == db_idx):
            dev += dev_min + 1e8
        elif penalty > 0:
            rows = trial[:c["num_rec"]]
            dev += np.sum((rows > upper) | (rows < lower)) * penalty
        if dev < dev_min:
            selected, dev_min = db_idx, dev
    return selected, dev_min


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("dev_min", [1e-3, 1e5])
def test_matches_original_without_penalty(seed, dev_min):
    case = make_case(seed)
    expected = call(greedy_algorithm, case, 0, dev_min)
    result = call(greedy_algorithm_fast, case, 0, dev_min)

    assert result[0] == expected[0]
    assert result[1] == pytest.approx(expected[1], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("dev_min", [1e-3, 1e5])
def test_matches_reference_with_penalty(seed, dev_min):
    case = make_case(seed)
    expected = reference(case, 2, dev_min)
    result = call(greedy_algorithm_fast, case, 2, dev_min)

    assert result[0] == expected[0]
    assert result[1] == pytest.approx(expected[1], rel=1e-9, abs=1e-12)
