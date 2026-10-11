"""Selective classification: semantic unknown (class 2) differs from rejection."""
import math
from .contract import require


def selective_metrics(probabilities, targets, threshold):
    require(0 <= threshold <= 1, "threshold range")
    require(len(probabilities) == len(targets), "metric length")
    n, accepted, errors, unknown, unsafe = len(targets), 0, 0, 0, 0
    semantic_unknown = 0
    for probs, target in zip(probabilities, targets):
        require(len(probs) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in probs) and abs(sum(probs) - 1) < 1e-5, "probabilities")
        require(target in (0, 1, 2), "target")
        predicted = max(range(3), key=lambda i: probs[i])
        semantic_unknown += predicted == 2
        unknown += target == 2
        # Predicted unknown, low confidence, or a known-class tie abstains.
        act = predicted != 2 and probs[predicted] >= threshold and probs[0] != probs[1]
        if act:
            accepted += 1
            errors += predicted != target
            unsafe += target == 2
    return {"n": n, "accepted": accepted, "coverage": accepted / n if n else None,
            "accepted_errors": errors, "risk": errors / accepted if accepted else None,
            "truth_unknown": unknown, "predicted_unknown": semantic_unknown,
            "unknown_accepted": unsafe, "rejected": n - accepted}


def choose_threshold(split, probabilities, targets, max_risk, min_accepted=1):
    require(split == "calibration", "threshold fitting is calibration-only; test forbidden")
    require(0 <= max_risk <= 1 and min_accepted >= 1, "risk criteria")
    choices = []
    for threshold in sorted({0.0, 1.0, *(max(p) for p in probabilities)}):
        result = selective_metrics(probabilities, targets, threshold)
        if result["accepted"] >= min_accepted and result["risk"] <= max_risk:
            choices.append((result["coverage"], threshold))
    return max(choices, default=(None, None))[1]
