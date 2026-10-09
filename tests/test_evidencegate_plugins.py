import numpy as np
import pytest

from evidencegate import (
    image_jpeg,
    image_native,
    image_query_aware_crop50,
    image_center_crop50,
    image_random_crop50,
    image_resize50,
    video_bitrate_only,
    video_resize50,
    video_select,
)
from codecref_lite.schemas import CandidateFeature


def encoder(frame, quality=75):
    return b"jpeg:" + bytes([quality]) + np.ascontiguousarray(frame).tobytes()[:8]


def frames():
    return tuple(np.full((10, 12, 3), i, dtype=np.uint8) for i in range(8))


def test_image_transforms_are_deterministic_and_bind_source_hash():
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    native = image_native(image)
    reduced = image_resize50(image)
    repeated = image_resize50(image)
    cropped = image_random_crop50(image, seed=17)
    assert native.source_sha256 == reduced.source_sha256 == cropped.source_sha256
    assert reduced.derived_sha256 == repeated.derived_sha256
    assert reduced.model_pixels / native.model_pixels == pytest.approx(0.5, abs=0.02)
    assert reduced.decoded_pixels == native.decoded_pixels
    assert cropped.decoded_pixels == native.decoded_pixels
    assert cropped.region == image_random_crop50(image, seed=17).region


def test_jpeg_model_input_is_the_decoded_compressed_image():
    source = np.zeros((8, 8, 3), np.uint8)
    source[0, 0] = (255, 20, 30)
    result = image_jpeg(source, quality=61)
    assert result.condition == "jpeg" and result.metadata["quality"] == 61
    assert result.metadata["model_input"] == "decoded_jpeg"
    assert result.model_frames[0].shape == source.shape
    assert not np.array_equal(result.model_frames[0], source)
    with pytest.raises(ValueError, match="cannot_be_decoded"):
        image_jpeg(source, encoder=encoder)


def test_query_crop_requires_non_gt_provenance_and_bounded_region():
    image = np.zeros((10, 12, 3), np.uint8)
    with pytest.raises(ValueError, match="query_provider"):
        image_query_aware_crop50(image, region=(0, 0, 5, 5), provider="ground_truth", provenance_hash="a" * 64)
    with pytest.raises(ValueError, match="provenance"):
        image_query_aware_crop50(image, region=(0, 0, 5, 5), provider="detector-v1", provenance_hash=None)
    with pytest.raises(ValueError, match="out_of_bounds"):
        image_query_aware_crop50(image, region=(9, 0, 5, 5), provider="detector-v1", provenance_hash="a" * 64)
    result = image_query_aware_crop50(image, region=(1, 2, 5, 5), provider="detector-v1", provenance_hash="a" * 64)
    assert result.metadata["provider_region"] == [1, 2, 5, 5]
    assert result.region is not None
    assert result.model_pixels / result.decoded_pixels == pytest.approx(0.5, abs=0.05)
    crop_x, crop_y, crop_width, crop_height = result.region
    assert crop_x <= 1 and crop_y <= 2
    assert crop_x + crop_width >= 6 and crop_y + crop_height >= 7
    with pytest.raises(ValueError, match="exceeds_crop50"):
        image_query_aware_crop50(image, region=(0, 0, 10, 10), provider="detector-v1", provenance_hash="a" * 64)


def test_center_crop_fallback_is_dimension_only_and_near_half_area():
    image = np.zeros((10, 12, 3), np.uint8)
    result = image_center_crop50(image)
    assert result.condition == "center_crop50"
    assert result.metadata["query_aware"] is False
    assert result.metadata["input_dependence"] == "image_dimensions_only"
    assert result.model_pixels / result.decoded_pixels == pytest.approx(0.5, abs=0.05)


def test_video_selectors_and_resize_metadata():
    values = frames()
    dense = video_select(values, condition="dense")
    uniform = video_select(values, condition="uniform", budget=4)
    random1 = video_select(values, condition="random", budget=4, global_seed=4, window_id="w")
    random2 = video_select(values, condition="random", budget=4, global_seed=4, window_id="w")
    rgb = video_select(values, condition="rgb_difference", budget=4)
    assert dense.selected_frame_indices == tuple(range(8))
    assert len(uniform.selected_frame_indices) == len(random1.selected_frame_indices) == len(rgb.selected_frame_indices) == 4
    assert random1.selected_frame_indices == random2.selected_frame_indices
    assert random1.decoded_pixels == 8 * 10 * 12
    assert random1.model_pixels == 4 * 10 * 12
    resized = video_resize50(values)
    assert resized.metadata["transform"] == "resize"
    assert resized.decoded_pixels == 8 * 10 * 12
    assert resized.model_pixels < resized.decoded_pixels


def test_video_random_selection_handles_nonzero_first_source_index():
    values = frames()
    for seed in range(100):
        result = video_select(values, condition="random50", budget=2, global_seed=seed)
        if result.selected_frame_indices[0] > 1:
            assert result.evidence.frame_index == result.selected_frame_indices[0]
            assert result.evidence.width == 12
            return
    raise AssertionError("test did not find a nonzero-first deterministic sample")


def test_codec_selector_and_dimension_fail_closed():
    values = frames()
    features = tuple(CandidateFeature(i, float(i), 1.0 + i, 0.0, 0.0, float(i), "P", pts_supported=True, packet_attribution_supported=True, mv_side_data_valid=True, mv_missing=False) for i in range(8))
    result = video_select(values, condition="codec", budget=3, codec_features=features)
    assert len(result.selected_frame_indices) == 3
    with pytest.raises(ValueError, match="dimensions"):
        video_select(values[:-1] + (np.zeros((11, 12, 3), np.uint8),), condition="dense")


def test_bitrate_only_separates_bytes_from_pixels_and_tokens():
    source = frames()
    reduced = tuple(np.clip(frame + 1, 0, 255).astype(np.uint8) for frame in source)
    result = video_bitrate_only(source, reduced_frames=reduced, original_encoded_bytes=1000, reduced_encoded_bytes=400)
    assert result.encoded_bytes == 400
    assert result.model_pixels == result.decoded_pixels
    assert result.metadata["encoded_byte_reduction"] == 600
    assert result.metadata["token_reduction_claim"] is False
    assert result.metadata["model_input"] == "decoded_reduced_bitrate_video"
    assert np.array_equal(result.model_frames[0], reduced[0])
    with pytest.raises(ValueError, match="encoded_byte"):
        video_bitrate_only(source, reduced_frames=reduced, original_encoded_bytes=100, reduced_encoded_bytes=200)
    with pytest.raises(ValueError, match="reduced_frames_required"):
        video_bitrate_only(source, reduced_frames=None, original_encoded_bytes=1000, reduced_encoded_bytes=400)
