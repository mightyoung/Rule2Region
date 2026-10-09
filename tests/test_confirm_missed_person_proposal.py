import numpy as np

from research_tools.cascade_experiment import Box
from research_tools.confirm_missed_person_proposal import run_frozen_confirmation
from research_tools.run_missed_person_proposal_baseline import FrameRecord, _camera


def _record(sequence: str, frame: int, confidence: float) -> FrameRecord:
    rgb = np.zeros((72, 128, 3), dtype=np.uint8)
    prediction = (Box(16, 16, 32, 48), confidence)
    truth = (Box(80, 20, 96, 60),)
    return FrameRecord(
        sequence,
        _camera(sequence),
        frame,
        128,
        72,
        (prediction,),
        truth,
        rgb,
        rgb,
    )


def test_frozen_confirmation_does_not_search_mode_or_configuration() -> None:
    fit = [_record("VIRAT_S_010004_01_x", index, .2 + index / 100) for index in range(10)]
    confirmation = [_record("VIRAT_S_010111_02_x", index, .3 + index / 100) for index in range(10)]
    payload = run_frozen_confirmation(fit, confirmation, l2=.001, peak_count=1)
    assert payload["method"] == "uncertainty_only_pointwise_ridge"
    assert payload["frozen_configuration"]["l2"] == .001
    assert payload["frozen_configuration"]["peak_count"] == 1
    assert payload["configuration_source"].startswith("PREREGISTERED")
    assert payload["gt_used_online"] is False
    assert payload["strong_model_called"] is False
    assert payload["sequence_split_leakage"] is False
