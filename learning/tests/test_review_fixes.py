"""Regression tests for post-e626 independent review; standard library only."""
import unittest
from learning import evaluate
from learning.metrics import selective_metrics

TASKS = ("relations", "matches", "sufficiency")


class ReviewFixTests(unittest.TestCase):
    def examples(self):
        probs = {k: [[.1, .8, .1]] for k in TASKS}
        truth = {k: [1] for k in TASKS}
        probs["matches"], truth["matches"] = [], []
        return probs, truth

    def test_relation_only_calibration_and_test_report(self):
        probs, truth = self.examples()
        thresholds, status = evaluate.fit_task_thresholds(probs, truth, .1)
        self.assertIsNone(thresholds["matches"])
        self.assertEqual(status["matches"], "not_applicable_no_candidates")
        self.assertEqual(status["relations"], "calibrated")
        report = evaluate.task_report([], [], thresholds["matches"])
        self.assertEqual(report["status"], "not_applicable_no_candidates")
        self.assertEqual(report["n"], 0)
        self.assertIsNone(report["accepted"])
        self.assertIsNone(report["risk"])
        policy = {"fit_split": "calibration", "manifest_sha256": "m", "checkpoint_sha256": "c", "thresholds": thresholds, "task_statuses": status, "max_risk": .1}
        self.assertEqual(evaluate.policy_for_test(policy, "m", "c"), thresholds)
        # A task appearing only at test cannot use a fabricated threshold.
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            evaluate.task_report([[.1, .8, .1]], [1], None)

    def test_present_task_without_labels_or_feasible_coverage_fails(self):
        probs, truth = self.examples()
        probs["matches"] = [[.1, .8, .1]]
        with self.assertRaisesRegex(ValueError, "length"):
            evaluate.fit_task_thresholds(probs, truth, .1)
        truth["matches"] = [2]
        with self.assertRaisesRegex(ValueError, "no feasible calibration.*matches"):
            evaluate.fit_task_thresholds(probs, truth, 0)
        with self.assertRaisesRegex(ValueError, "no applicable"):
            evaluate.fit_task_thresholds({k: [] for k in TASKS}, {k: [] for k in TASKS}, 0)

    def test_policy_cannot_hide_failed_task_as_not_applicable(self):
        probs, truth = self.examples()
        thresholds, status = evaluate.fit_task_thresholds(probs, truth, .1)
        policy = {"fit_split": "calibration", "manifest_sha256": "m", "checkpoint_sha256": "c", "thresholds": thresholds, "task_statuses": status, "max_risk": .1}
        policy["task_statuses"]["relations"] = "not_applicable_no_candidates"
        with self.assertRaisesRegex(ValueError, "status/threshold"):
            evaluate.policy_for_test(policy, "m", "c")

    def test_unknown_top_ties_abstain_in_both_known_classes(self):
        for probs in ([.5, 0, .5], [0, .5, .5], [1/3, 1/3, 1/3]):
            with self.subTest(probs=probs):
                report = selective_metrics([probs], [2], max(probs))
                self.assertEqual(report["accepted"], 0)
                self.assertEqual(report["predicted_unknown"], 1)
                self.assertEqual(report["unknown_accepted"], 0)
        self.assertEqual(selective_metrics([[.5, .5, 0]], [0], .5)["accepted"], 0)
        self.assertEqual(selective_metrics([[.51, 0, .49]], [0], .5)["accepted"], 1)
