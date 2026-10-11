import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from learning.tests.test_contract import load


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch absent; shape/gradient/resume NOT RUN")
class TorchSoftwareTests(unittest.TestCase):
    def test_gradient_freeze_and_pair_symmetry(self):
        import torch
        from learning.model import FrozenEncoder, RelationHead
        from learning.dataset import RelationDataset
        torch.manual_seed(20261011)
        ds = RelationDataset(load("software_manifest.json"), load("software_features.json"), "train", True)
        x, targets = ds[0]
        self.assertEqual(set(x), {"door", "region_a", "region_b", "view_a", "view_b"})
        head = RelationHead(8)
        y = head(**x)
        self.assertEqual(y["relations"].shape, (3, 3))
        self.assertEqual(y["matches"].shape, (1, 3))
        swap = dict(x, region_a=x["region_b"], region_b=x["region_a"], view_a=x["view_b"], view_b=x["view_a"])
        for key in y: self.assertTrue(torch.equal(y[key], head(**swap)[key]))
        loss = sum(torch.nn.functional.cross_entropy(y[k], targets[k]) for k in y)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        frozen = FrozenEncoder(torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(.5)))
        frozen.train()
        encoded = frozen(x["door"].requires_grad_())
        self.assertFalse(encoded.requires_grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in frozen.parameters()))
        self.assertFalse(frozen.encoder.training)

    def test_software_cannot_enter_real_training(self):
        from learning.dataset import RelationDataset
        with self.assertRaisesRegex(ValueError, "explicit software-test"):
            RelationDataset(load("software_manifest.json"), load("software_features.json"), "train")

    def test_resume_matches_uninterrupted_and_rejects_mismatch(self):
        import torch
        from learning.train import run
        data = Path(__file__).resolve().parents[1] / "data"
        with tempfile.TemporaryDirectory() as directory:
            def args(name, steps, resume=False):
                return SimpleNamespace(manifest=data / "software_manifest.json", features=data / "software_features.json", checkpoint=Path(directory) / name, resume=resume, steps=steps, seed=20261011, hidden=16, lr=.001, device="cpu", software_test=True)
            run(args("whole.pt", 4))
            run(args("resumed.pt", 2))
            run(args("resumed.pt", 4, True))
            whole = torch.load(Path(directory) / "whole.pt", weights_only=True)
            resumed = torch.load(Path(directory) / "resumed.pt", weights_only=True)
            for key in whole["model"]: self.assertTrue(torch.equal(whole["model"][key], resumed["model"][key]))
            bad = args("resumed.pt", 5, True)
            bad.seed = 1
            with self.assertRaisesRegex(ValueError, "resume config"):
                run(bad)

    def test_relation_only_evaluation_cli(self):
        import subprocess
        import sys
        import torch
        from learning.contract import fingerprint
        from learning.train import run
        manifest, features = load("software_manifest.json"), load("software_features.json")
        for row in manifest["samples"]:
            row["input"]["view_pairs"] = []
            row["truth"]["matches"] = {}
            features["samples"][row["id"]]["input_sha256"] = fingerprint(row["input"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp, fp, cp, pp = (root / name for name in ("manifest.json", "features.json", "head.pt", "policy.json"))
            mp.write_text(json.dumps(manifest))
            fp.write_text(json.dumps(features))
            run(SimpleNamespace(manifest=mp, features=fp, checkpoint=cp, resume=False, steps=1, seed=20261011, hidden=16, lr=.001, device="cpu", software_test=True))
            # Force a known class on this software-only head so CLI calibration
            # has coverage. This is a regression fixture, never model evidence.
            state = torch.load(cp, weights_only=True)
            for name, tensor in state["model"].items():
                tensor.zero_()
                if name.endswith("2.bias"):
                    tensor[1] = 2
            torch.save(state, cp)
            base = [sys.executable, "-m", "learning.evaluate", str(mp), str(fp), str(cp), "--software-test", "--policy", str(pp)]
            for split, extra in (("calibration", ["--max-risk", "1"]), ("test", [])):
                result = subprocess.run(base + ["--split", split] + extra, check=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
                report = json.loads(result.stdout)
                self.assertEqual(report["kind"], "software_test_only")
                self.assertIsNone(report["thresholds"]["matches"])
                self.assertEqual(report["metrics"]["matches"]["status"], "not_applicable_no_candidates")
                self.assertEqual(report["descriptive_risk_coverage"]["matches"], [])
                self.assertEqual(report["metrics"]["relations"]["status"], "evaluated")
            policy = json.loads(pp.read_text())
            self.assertEqual(policy["task_statuses"]["matches"], "not_applicable_no_candidates")
