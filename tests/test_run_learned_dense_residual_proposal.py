import numpy as np

from research_tools.cascade_experiment import Box
from research_tools.run_learned_dense_residual_proposal import assemble_features, run_fixed_gate
from research_tools.run_missed_person_proposal_baseline import FrameRecord, _camera


def _records(camera: str) -> list[FrameRecord]:
    rgb = np.zeros((72, 128, 3), dtype=np.uint8)
    sequence = f"{camera}_01_x"
    return [
        FrameRecord(sequence, _camera(sequence), frame, 128, 72, (), (Box(80, 20, 96, 60),), rgb, rgb)
        for frame in range(5)
    ]


def test_assemble_features_adds_detection_occupancy_channel() -> None:
    records = _records("VIRAT_S_010004")
    features = assemble_features(records, np.zeros((5, 3, 23, 40), dtype=np.float32))
    assert features.shape == (5, 4, 23, 40)


def test_fixed_gate_has_no_search_and_keeps_camera_isolation() -> None:
    fit = _records("VIRAT_S_010004")
    evaluation = _records("VIRAT_S_010111")
    fit_scores = np.zeros((5, 3, 23, 40), dtype=np.float32)
    evaluation_scores = np.zeros((5, 3, 23, 40), dtype=np.float32)
    fit_scores[:, :, 13, 27] = .2
    evaluation_scores[:, :, 13, 27] = .3
    payload = run_fixed_gate(fit, evaluation, fit_scores, evaluation_scores)
    assert payload["fixed_configuration"]["l2"] == .001
    assert payload["strong_model_called"] is False
    assert payload["gt_used_online"] is False
    assert payload["sequence_split_leakage"] is False
