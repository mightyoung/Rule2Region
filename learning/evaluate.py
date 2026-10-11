"""Task-level evaluation; does not claim complete topology or counter accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
from .contract import exact, fingerprint, require
from .metrics import choose_threshold, selective_metrics


def policy_for_test(policy, manifest_sha, checkpoint_sha):
    exact(policy, ("fit_split", "manifest_sha256", "checkpoint_sha256", "thresholds", "max_risk"), "policy")
    require(policy["fit_split"] == "calibration", "test requires calibration policy")
    require(policy["manifest_sha256"] == manifest_sha and policy["checkpoint_sha256"] == checkpoint_sha, "policy artifact mismatch")
    exact(policy["thresholds"], ("relations", "matches", "sufficiency"), "thresholds")
    require(all(type(x) in (int, float) and 0 <= x <= 1 for x in policy["thresholds"].values()), "thresholds invalid")
    require(type(policy["max_risk"]) in (int, float) and 0 <= policy["max_risk"] <= 1, "policy risk")
    return policy["thresholds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--policy", type=Path, required=True, help="calibration writes; test reads")
    parser.add_argument("--max-risk", type=float, help="only calibration can fit thresholds")
    parser.add_argument("--software-test", action="store_true")
    args = parser.parse_args()
    require((args.split == "calibration") == (args.max_risk is not None), "calibration needs max-risk; test may not tune")
    try:
        import torch
        from .dataset import RelationDataset
        from .model import RelationHead
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("PyTorch unavailable: evaluation NOT RUN") from exc
        raise
    manifest = json.loads(args.manifest.read_text())
    features = json.loads(args.features.read_text())
    dataset = RelationDataset(manifest, features, args.split, args.software_test)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(state["config"]["manifest_sha256"] == fingerprint(manifest) and state["config"]["features_sha256"] == fingerprint(features), "checkpoint data mismatch")
    model = RelationHead(features["dimension"], state["config"]["hidden"])
    model.load_state_dict(state["model"])
    model.eval()
    probabilities, targets = {k: [] for k in ("relations", "matches", "sufficiency")}, {k: [] for k in ("relations", "matches", "sufficiency")}
    with torch.no_grad():
        for inputs, truth in dataset:
            for task, value in model(**inputs).items():
                probabilities[task].extend(value.softmax(-1).tolist())
                targets[task].extend(truth[task].tolist())
    ck_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.split == "calibration":
        thresholds = {k: choose_threshold(args.split, probabilities[k], targets[k], args.max_risk) for k in targets}
        require(all(v is not None for v in thresholds.values()), "no feasible calibration policy; do not evaluate test")
        require(not args.policy.exists(), "policy already exists: use new version path")
        args.policy.parent.mkdir(parents=True, exist_ok=True)
        args.policy.write_text(json.dumps({"fit_split": "calibration", "manifest_sha256": fingerprint(manifest), "checkpoint_sha256": ck_sha, "thresholds": thresholds, "max_risk": args.max_risk}, indent=2) + "\n")
    else:
        thresholds = policy_for_test(json.loads(args.policy.read_text()), fingerprint(manifest), ck_sha)
    result = {k: selective_metrics(probabilities[k], targets[k], thresholds[k]) for k in targets}
    curves = {k: [{"threshold": t, **selective_metrics(probabilities[k], targets[k], t)} for t in (0, 0.5, 0.75, 0.9, 1)] for k in targets}
    print(json.dumps({"kind": "software_test_only" if args.software_test else "task_evaluation", "split": args.split, "thresholds": thresholds, "metrics": result, "descriptive_risk_coverage": curves, "warning": "task marginals only; no complete-boundary or occupancy claim; test curves must not tune policy"}, allow_nan=False))


if __name__ == "__main__":
    main()
