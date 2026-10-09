import numpy as np

from research_tools.cascade_experiment import Box
from research_tools.run_dense_residual_proposal import mask_existing_predictions, run
from research_tools.run_missed_person_proposal_baseline import FrameRecord, _camera


def _record(frame: int) -> FrameRecord:
    rgb = np.zeros((72, 128, 3), dtype=np.uint8)
    return FrameRecord(
        "VIRAT_S_010111_02_x",
        _camera("VIRAT_S_010111_02_x"),
        frame,
        128,
        72,
        ((Box(0, 0, 32, 36), .4),),
        (Box(80, 20, 96, 60),),
        rgb,
        rgb,
    )


def test_mask_existing_predictions_preserves_uncovered_dense_region() -> None:
    record = _record(1)
    score = np.ones((23, 40), dtype=np.float32)
    masked = mask_existing_predictions(score, record)
    assert masked[2, 2] == 0
    assert masked[15, 30] == 1


def test_dense_residual_run_keeps_fixed_budget_and_no_strong_model() -> None:
    records = [_record(frame) for frame in range(5)]
    scores = np.zeros((5, 23, 40), dtype=np.float32)
    scores[:, 13, 27] = np.arange(1, 6)
    payload = run(records, scores)
    assert payload["strong_model_called"] is False
    assert payload["gt_used_online"] is False
    assert payload["result"]["selected_frames"] == 1
