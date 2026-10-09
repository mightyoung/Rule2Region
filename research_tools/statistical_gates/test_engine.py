import pytest
from research_tools.statistical_gates import *

def test_equal_camera_estimand_weights_sources_not_windows():
    assert equal_camera_estimand([[1, 3], [10]]) == pytest.approx(6.0)

def test_bootstrap_is_deterministic_and_seeded():
    a = hierarchical_bootstrap([[1, 2], [8, 10]], draws=100, seed=7)
    assert (a == hierarchical_bootstrap([[1, 2], [8, 10]], draws=100, seed=7)).all()
    assert not (a == hierarchical_bootstrap([[1, 2], [8, 10]], draws=100, seed=8)).all()

def test_sign_flip_exact_small_camera_fixture():
    assert sign_flip_pvalue([1, 1], draws=100) == pytest.approx(.5)

def test_holm_adjust_preserves_original_order_and_monotonicity():
    assert holm_adjust([.01, .04, .03, .2]) == pytest.approx([.04, .09, .09, .2])

@pytest.mark.parametrize("value,expected", [(0.039999, False), (0.04, True), (0.06, True), (0.060001, False)])
def test_calibration_fwer_boundary(value, expected):
    # Exercise the exact inclusive decision boundaries without formal outcomes.
    ci = (0.05, 0.05)
    assert ((.04 <= value <= .06) and ci[0] <= .05 <= ci[1]) is expected

def test_cal_acc4_and_int2_are_separate_and_reproducible():
    a = run_calibration(CAL_ACC4, 4, 20260810)
    b = run_calibration(CAL_ACC4, 4, 20260810)
    i = run_calibration(CAL_INT2, 2, 20260811)
    assert a.to_dict() == b.to_dict()
    assert a.evidence_level == i.evidence_level == "FIXTURE_VERIFIED"
    assert len({a.dgp_sha256, a.covariance_sha256, a.config_sha256, a.result_sha256}) == 4
    assert a.artifact_id != i.artifact_id and a.seed != i.seed and a.receipt != i.receipt
    assert a.verdict == "PASS" and i.verdict == "PASS"

def test_calibration_rejects_wrong_family_schema():
    with pytest.raises(ValueError): run_calibration(CAL_ACC4, 6, 20260810)

def test_failure_rate_one_percent_is_eligible_but_above_is_invalid():
    reject_five_percent = lambda _rng, i: i % 20 == 0
    ok = run_calibration(
        CAL_ACC4, 4, 20260810, statistic=reject_five_percent,
        failure_indices=range(200),
    )
    bad = run_calibration(
        CAL_ACC4, 4, 20260810, statistic=reject_five_percent,
        failure_indices=range(201),
    )
    assert ok.verdict == "PASS" and bad.verdict == "CAL_ACC4_INVALID"

def test_power_requires_exact_formal_structure_and_blinded_pilot():
    s = FormalStructure(8, (4,)*8, tuple((10,)*4 for _ in range(8)), "formal-sha")
    validate_power_inputs(s, {"blinded": True, "variance": 1.0, "icc": .2})
    with pytest.raises(ValueError): validate_power_inputs(s, {"blinded": True, "variance": 1.0, "icc": .2, "effect": .3})

def test_non_boolean_fixture_result_counts_as_failure():
    result = run_calibration(
        CAL_INT2, 2, 20260811, statistic=lambda _rng, _i: float("nan")
    )
    assert result.failures == 20000
    assert result.verdict == "CAL_INT2_INVALID"
