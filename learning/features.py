"""Frozen feature cache contract; producers must separately audit encoder provenance."""
import math
from .contract import exact, fingerprint, nonempty, require


def validate_features(manifest, cache):
    exact(cache, ("purpose", "encoder", "dimension", "samples"), "feature cache")
    require(cache["purpose"] == manifest["purpose"], "cache purpose")
    exact(cache["encoder"], ("revision", "preprocessing", "frozen", "input_only"), "encoder")
    require(all(nonempty(cache["encoder"][k]) for k in ("revision", "preprocessing")), "encoder provenance")
    require(cache["encoder"]["frozen"] is True and cache["encoder"]["input_only"] is True, "encoder must be frozen/input-only")
    dim = cache["dimension"]
    require(type(dim) is int and dim > 0, "feature dimension")
    require(isinstance(cache["samples"], dict), "cache samples")
    require(set(cache["samples"]) == {row["id"] for row in manifest["samples"]}, "cache sample mismatch")
    for row in manifest["samples"]:
        feature = cache["samples"][row["id"]]
        exact(feature, ("input_sha256", "doors", "regions"), "feature sample")
        require(feature["input_sha256"] == fingerprint(row["input"]), "cache input hash mismatch")
        for key, source in (("doors", "door_candidates"), ("regions", "regions")):
            require(isinstance(feature[key], dict) and set(feature[key]) == {p["id"] for p in row["input"][source]}, "feature candidate mismatch")
            for vec in feature[key].values():
                require(isinstance(vec, list) and len(vec) == dim and all(type(x) in (int, float) and math.isfinite(x) for x in vec), "finite feature shape")
    return cache
