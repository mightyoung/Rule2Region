import copy
import json
import unittest
from pathlib import Path
from learning.contract import fingerprint, validate_manifest
from learning.features import validate_features
from learning.metrics import choose_threshold, selective_metrics
from learning.evaluate import policy_for_test
from learning.review import review_key, review_queue

DATA = Path(__file__).resolve().parents[1] / "data"


def load(name):
    return json.loads((DATA / name).read_text())


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.m = load("software_manifest.json")
        self.f = load("software_features.json")

    def test_fixture_and_empty_real(self):
        validate_manifest(self.m)
        validate_features(self.m, self.f)
        self.assertEqual(validate_manifest(load("real_manifest.json"))["samples"], [])

    def test_no_input_truth(self):
        for field in ("truth", "physical_door_ids", "review", "labels"):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.m)
                bad["samples"][0]["input"][field] = "leak"
                with self.assertRaisesRegex(ValueError, "truth must stay separate"):
                    validate_manifest(bad)

    def test_identity_leakage_each_group(self):
        for key in ("building_id", "layout_family_id", "physical_door_ids", "source_plan_sha256", "variant_family_id"):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.m)
                value = "a" * 64 if key == "source_plan_sha256" else bad["samples"][0]["identity"][key]
                bad["samples"][0]["identity"][key] = value
                bad["samples"][1]["identity"][key] = value
                with self.assertRaisesRegex(ValueError, "cross-split"):
                    validate_manifest(bad)

    def test_same_media_cannot_cross_split(self):
        self.m["samples"][1]["input"]["views"][0]["sha256"] = self.m["samples"][0]["input"]["views"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "media leakage"):
            validate_manifest(self.m)

    def test_preannotation_is_not_truth(self):
        row = self.m["samples"][0]
        labels = row["truth"]
        row["ai_preannotation"] = {"model_revision": "test-only", "status": "default_adopted_pending_review", "risk": "normal", "labels": labels, "versions": row["versions"]}
        row["review"] = {"status": "pending", "reviewer": None, "evidence_refs": [], "versions": None}
        with self.assertRaisesRegex(ValueError, "pending is not truth"):
            validate_manifest(self.m)
        row["truth"] = None
        validate_manifest(self.m)
        self.assertIn(row["id"], review_queue(self.m["samples"]))

    def test_human_review_requires_evidence_and_versions(self):
        for mutation in ("evidence", "version", "reviewer", "permission"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.m)
                row = bad["samples"][0]
                if mutation == "evidence": row["review"]["evidence_refs"] = []
                if mutation == "version": row["review"]["versions"] = {**row["versions"], "rules": "new"}
                if mutation == "reviewer": row["review"]["reviewer"] = None
                if mutation == "permission": row["source"]["permission_status"] = "pending"
                with self.assertRaises(ValueError): validate_manifest(bad)

    def test_missing_or_wrong_label_and_reference(self):
        for mutation in ("missing", "invalid", "reference", "box", "purpose"):
            bad = copy.deepcopy(self.m)
            row = bad["samples"][0]
            if mutation == "missing": del row["truth"]["relations"]["r0"]
            if mutation == "invalid": row["truth"]["relations"]["r0"] = "probably"
            if mutation == "reference": row["input"]["region_pairs"][0]["region_a"] = "absent"
            if mutation == "box": row["input"]["door_candidates"][0]["box"] = [0.5, 0.1, 0.4, 0.9]
            if mutation == "purpose": bad["purpose"] = "real_research"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_manifest(bad)

    def test_cache_provenance_finite_shape(self):
        for mutation in ("hash", "nan", "shape", "truth", "frozen", "purpose"):
            bad = copy.deepcopy(self.f)
            feature = bad["samples"]["software-0"]
            if mutation == "hash": feature["input_sha256"] = "0" * 64
            if mutation == "nan": feature["doors"]["d0"][0] = float("nan")
            if mutation == "shape": feature["doors"]["d0"] = [0]
            if mutation == "truth": feature["truth"] = "leak"
            if mutation == "frozen": bad["encoder"]["frozen"] = False
            if mutation == "purpose": bad["purpose"] = "real_research"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_features(self.m, bad)

    def test_queue_history_and_version(self):
        self.assertEqual(review_queue(self.m["samples"]), [])
        row = self.m["samples"][0]
        old = review_key(row)
        row["versions"] = {**row["versions"], "evidence": "new"}
        self.assertNotEqual(old, review_key(row))
        self.assertEqual(review_queue(self.m["samples"]), [row["id"]])

    def test_unknown_distinct_from_low_confidence(self):
        m = selective_metrics([[.01, .01, .98], [.45, .5, .05], [.01, .98, .01]], [2, 1, 2], .9)
        self.assertEqual((m["predicted_unknown"], m["rejected"], m["unknown_accepted"]), (1, 2, 1))
        self.assertEqual(m["risk"], 1)
        self.assertIsNone(selective_metrics([], [], .9)["risk"])
        self.assertEqual(selective_metrics([[.5, .5, 0]], [0], 0)["accepted"], 0)

    def test_threshold_test_forbidden_and_policy_frozen(self):
        with self.assertRaisesRegex(ValueError, "test forbidden"):
            choose_threshold("test", [[.1, .8, .1]], [1], .1)
        self.assertIsNotNone(choose_threshold("calibration", [[.1, .8, .1]], [1], .1))
        self.assertIsNone(choose_threshold("calibration", [[.1, .8, .1]], [2], 0))
        policy = {"fit_split": "calibration", "manifest_sha256": fingerprint(self.m), "checkpoint_sha256": "a" * 64, "thresholds": {k: .9 for k in ("relations", "matches", "sufficiency")}, "max_risk": .1}
        policy_for_test(policy, fingerprint(self.m), "a" * 64)
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            policy_for_test(policy, fingerprint(self.m), "b" * 64)

    def test_candidate_duplicates_and_queue_priority(self):
        bad = copy.deepcopy(self.m)
        rp = bad["samples"][0]["input"]["region_pairs"]
        rp[2] = dict(rp[0], id="r2")
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            validate_manifest(bad)
        for row in self.m["samples"]:
            row["review"] = {"status": "pending", "reviewer": None, "evidence_refs": [], "versions": None}
            row["ai_preannotation"] = {"model_revision": "test-only", "status": "proposed", "risk": "normal", "labels": row["truth"], "versions": row["versions"]}
            row["truth"] = None
            row["ai_preannotation"]["labels"]["relations"]["r2"] = "connected"
        self.m["samples"][2]["ai_preannotation"]["risk"] = "high"
        self.assertEqual(review_queue(self.m["samples"])[0], "software-2")
        validate_manifest(self.m)
        self.m["samples"][0]["ai_preannotation"]["versions"] = {**self.m["samples"][0]["versions"], "labels": "stale"}
        with self.assertRaisesRegex(ValueError, "stale ai"):
            validate_manifest(self.m)
