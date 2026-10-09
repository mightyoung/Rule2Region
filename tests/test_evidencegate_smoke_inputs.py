import numpy as np

from codecref_lite.schemas import CandidateFeature
from evidencegate.smoke_inputs import prepare_video_input


def _features(count):
    return tuple(
        CandidateFeature(index, index / 12, 0.1 + index, 0.1, 0.2, 0.3 + index, "I" if index == 0 else "P", key_frame=index == 0, mv_missing=index == 0)
        for index in range(count)
    )


def test_video_smoke_conditions_return_actual_model_frames():
    source = tuple(np.full((8, 8, 3), index, np.uint8) for index in range(8))
    reduced = tuple(np.full((8, 8, 3), index + 1, np.uint8) for index in range(8))
    guided = prepare_video_input(source, "codec_guided50", source_sha256="a" * 64, codec_features=_features(8))
    assert len(guided.model_frames) == 4
    bitrate = prepare_video_input(
        source,
        "bitrate_only",
        source_sha256="a" * 64,
        bitrate_frames=reduced,
        original_encoded_bytes=1000,
        reduced_encoded_bytes=400,
    )
    assert np.array_equal(bitrate.model_frames[0], reduced[0])
    assert bitrate.encoded_bytes == 400
