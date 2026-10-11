"""Dataset selects reviewed targets separately from allowed input features."""
import torch
from torch.utils.data import Dataset
from .contract import RELATIONS, MATCHES, SUFFICIENCY, require, validate_manifest
from .features import validate_features

CLASSES = {"relations": RELATIONS, "matches": MATCHES, "sufficiency": SUFFICIENCY}


class RelationDataset(Dataset):
    def __init__(self, manifest, features, split, software_test=False):
        validate_manifest(manifest)
        validate_features(manifest, features)
        require(split in ("train", "calibration", "test"), "split")
        require((manifest["purpose"] == "software_test_only") == software_test, "software tensors require explicit software-test mode")
        self.rows = [row for row in manifest["samples"] if row["split"] == split]
        require(self.rows, "empty split; do not fabricate real data")
        require(all(row["review"]["status"] in ("confirmed", "corrected") for row in self.rows), "unreviewed samples cannot supervise/evaluate")
        self.features = features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        inp = row["input"]
        feature = self.features["samples"][row["id"]]
        dim = self.features["dimension"]
        def stack(values):
            return torch.tensor(values, dtype=torch.float32).reshape(-1, dim)
        rp, vp = inp["region_pairs"], inp["view_pairs"]
        inputs = {
            "door": stack([feature["doors"][p["door_id"]] for p in rp]),
            "region_a": stack([feature["regions"][p["region_a"]] for p in rp]),
            "region_b": stack([feature["regions"][p["region_b"]] for p in rp]),
            "view_a": stack([feature["doors"][p["door_a"]] for p in vp]),
            "view_b": stack([feature["doors"][p["door_b"]] for p in vp]),
        }
        targets = {task: torch.tensor([classes.index(row["truth"][task][p["id"]]) for p in (vp if task == "matches" else rp)], dtype=torch.long) for task, classes in CLASSES.items()}
        return inputs, targets
