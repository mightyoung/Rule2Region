import numpy as np
from evidencegate.cascade import Box, ROI, BenefitRouter, boundary_aware_wbf, map_local_box_to_global, select_rois_by_budget, calibration_metrics, primary_ownership


def test_mapping_uses_expanded_crop_origin():
    roi = ROI(100, 50, 200, 150)
    box = map_local_box_to_global(Box(10, 20, 30, 40, source="local"), roi)
    assert box.coords == (102.5, 62.5, 122.5, 82.5)
    assert box.roi_id == roi.roi_id


def test_same_roi_people_are_never_merged():
    roi = ROI(0, 0, 100, 100, roi_id="a")
    boxes = [Box(10, 10, 50, 80, .9, "local", "a"), Box(12, 12, 52, 82, .8, "local", "a")]
    assert len(boundary_aware_wbf(boxes, [roi], .3)) == 2


def test_primary_ownership_controls_context_only_counting():
    roi = ROI(0, 0, 100, 100, "a")
    context = Box(-20, 20, -5, 60, .8, "local", "a")
    owned = primary_ownership([context], [roi])[0]
    assert not owned.primary
    assert boundary_aware_wbf([owned], [roi]) == []


def test_cross_source_duplicate_is_fused():
    roi = ROI(0, 0, 100, 100, roi_id="a")
    boxes = [Box(10, 10, 50, 80, .9, "global"), Box(11, 11, 51, 81, .8, "local", "a")]
    fused = boundary_aware_wbf(boxes, [roi], .5)
    assert len(fused) == 1 and fused[0].source == "fused"


def test_matching_guards_keep_distinct_overlapping_people():
    # IoU is high, but centers are too far apart for one person.
    a = Box(0, 0, 100, 200, .9, "global")
    b = Box(45, 0, 145, 200, .8, "local", "r")
    assert len(boundary_aware_wbf([a, b], [ROI(0, 0, 200, 200, "r")], .2, .2)) == 2


def test_router_state_and_calibration_metrics_round_trip():
    router = BenefitRouter().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    restored = BenefitRouter.from_dict(router.to_dict())
    assert np.allclose(router.predict([[.5], [2.5]]), restored.predict([[.5], [2.5]]))
    assert set(calibration_metrics([.1, .9], [0, 1])) == {"brier", "ece"}


def test_unmatched_context_box_does_not_count_independently():
    roi = ROI(20, 20, 80, 80, roi_id="a")
    context = primary_ownership([Box(10, 10, 18, 18, .8, "local", "a")], [roi])
    assert not context[0].primary
    assert boundary_aware_wbf(context, [roi]) == []


def test_pixel_budget_and_router_monotonicity():
    rois = [ROI(0, 0, 10, 10, "a"), ROI(0, 0, 20, 20, "b")]
    chosen = select_rois_by_budget(rois, [1, 2], 140)
    assert [r.roi_id for r in chosen] == ["a"]
    router = BenefitRouter().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    pred = router.predict([[0], [1], [2], [3]])
    assert np.all(np.diff(pred) > 0)
