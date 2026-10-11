"""Task-level evaluation; does not claim complete topology or counter accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
from .contract import exact, fingerprint, require
from .metrics import choose_threshold, selective_metrics


TASKS = ("relations", "matches", "sufficiency")


def fit_task_thresholds(probabilities, targets, max_risk):
    exact(probabilities, TASKS, "task probabilities")
    exact(targets, TASKS, "task targets")
    require(0 <= max_risk <= 1, "risk criteria")
    require(any(targets.values()), "no applicable calibration tasks")
    thresholds, statuses = {}, {}
    for task in TASKS:
        require(len(probabilities[task]) == len(targets[task]), "metric length: " + task)
        if not targets[task]:
            thresholds[task] = None
            statuses[task] = "not_applicable_no_candidates"
            continue
        threshold = choose_threshold("calibration", probabilities[task], targets[task], max_risk)
        require(threshold is not None, "no feasible calibration policy: " + task + "; do not evaluate test")
        thresholds[task] = threshold
        statuses[task] = "calibrated"
    return thresholds, statuses


def task_report(probabilities, targets, threshold):
    require(len(probabilities) == len(targets), "metric length")
    if not targets:
        return {"status": "not_applicable_no_candidates", "n": 0, "accepted": None, "coverage": None, "risk": None}
    require(threshold is not None, "task not calibrated; do not score test candidates")
    return {"status": "evaluated", **selective_metrics(probabilities, targets, threshold)}


def policy_for_test(policy, manifest_sha, checkpoint_sha):
    exact(policy, ("fit_split", "manifest_sha256", "checkpoint_sha256", "thresholds", "task_statuses", "max_risk"), "policy")
    require(policy["fit_split"] == "calibration", "test requires calibration policy")
    require(policy["manifest_sha256"] == manifest_sha and policy["checkpoint_sha256"] == checkpoint_sha, "policy artifact mismatch")
    exact(policy["thresholds"], ("relations", "matches", "sufficiency"), "thresholds")
    exact(policy["task_statuses"], TASKS, "task statuses")
    for task in TASKS:
        threshold, status = policy["thresholds"][task], policy["task_statuses"][task]
        require((status == "not_applicable_no_candidates" and threshold is None) or
                (status == "calibrated" and type(threshold) in (int, float) and 0 <= threshold <= 1),
                "policy status/threshold mismatch")
    require(any(s == "calibrated" for s in policy["task_statuses"].values()), "no calibrated tasks")
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
        thresholds, statuses = fit_task_thresholds(probabilities, targets, args.max_risk)
        require(not args.policy.exists(), "policy already exists: use new version path")
        args.policy.parent.mkdir(parents=True, exist_ok=True)
        args.policy.write_text(json.dumps({"fit_split": "calibration", "manifest_sha256": fingerprint(manifest), "checkpoint_sha256": ck_sha, "thresholds": thresholds, "task_statuses": statuses, "max_risk": args.max_risk}, indent=2) + "\n")
    else:
        thresholds = policy_for_test(json.loads(args.policy.read_text()), fingerprint(manifest), ck_sha)
    result = {k: task_report(probabilities[k], targets[k], thresholds[k]) for k in targets}
    curves = {k: [{"threshold": t, **selective_metrics(probabilities[k], targets[k], t)} for t in (0, 0.5, 0.75, 0.9, 1)] if targets[k] else [] for k in targets}
    print(json.dumps({"kind": "software_test_only" if args.software_test else "task_evaluation", "split": args.split, "thresholds": thresholds, "metrics": result, "descriptive_risk_coverage": curves, "warning": "task marginals only; no complete-boundary or occupancy claim; test curves must not tune policy"}, allow_nan=False))


if __name__ == "__main__":
    main()
