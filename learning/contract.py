"""Dependency-free, fail-closed manifest contract. No media is read here."""
import hashlib
import json
import math

VERSION = "boundary-v1"
RELATIONS = ("not_connected", "connected", "unknown")
MATCHES = ("different", "same", "unknown")
SUFFICIENCY = ("insufficient", "sufficient", "unknown")
SPLITS = ("train", "calibration", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(obj, fields, name):
    require(isinstance(obj, dict) and set(obj) == set(fields), name + " fields invalid")


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_labels(labels, inputs):
    exact(labels, ("relations", "matches", "sufficiency"), "labels")
    by_task = {"relations": (inputs["region_pairs"], RELATIONS),
               "matches": (inputs["view_pairs"], MATCHES),
               "sufficiency": (inputs["region_pairs"], SUFFICIENCY)}
    for task, (pairs, classes) in by_task.items():
        require(isinstance(labels[task], dict), "labels must be maps")
        require(set(labels[task]) == {p["id"] for p in pairs}, "label candidate mismatch")
        require(all(v in classes for v in labels[task].values()), "invalid label")


def validate_manifest(payload):
    exact(payload, ("schema_version", "purpose", "samples"), "manifest")
    require(payload["schema_version"] == VERSION, "schema version")
    require(payload["purpose"] in ("real_research", "software_test_only"), "purpose")
    require(isinstance(payload["samples"], list), "samples list")
    owners, seen = {}, set()
    for row in payload["samples"]:
        exact(row, ("id", "kind", "split", "identity", "source", "versions", "input", "ai_preannotation", "review", "truth"), "sample")
        require(nonempty(row["id"]) and row["id"] not in seen, "sample id duplicate/empty")
        seen.add(row["id"])
        require(row["split"] in SPLITS, "split")
        require(row["kind"] in ("real", "software_tensor"), "sample kind")
        require((row["kind"] == "real") == (payload["purpose"] == "real_research"), "synthetic/real purpose mismatch")
        identity = row["identity"]
        exact(identity, ("building_id", "layout_family_id", "physical_door_ids", "source_plan_sha256", "variant_family_id"), "identity")
        require(all(nonempty(identity[k]) for k in ("building_id", "layout_family_id", "variant_family_id")), "missing grouping identity")
        require(isinstance(identity["physical_door_ids"], list) and all(nonempty(x) for x in identity["physical_door_ids"]), "door grouping")
        require(identity["source_plan_sha256"] is None or sha(identity["source_plan_sha256"]), "plan hash")
        for key, values in identity.items():
            for value in (values if isinstance(values, list) else [values]):
                if value is None:
                    continue
                token = (key, value)
                require(token not in owners or owners[token] == row["split"], "cross-split identity leakage: " + key)
                owners[token] = row["split"]
        source = row["source"]
        exact(source, ("url", "license", "access_date", "permission_status"), "source")
        require(all(nonempty(source[k]) for k in ("url", "license", "access_date")), "source provenance")
        require(source["permission_status"] in ("verified", "pending"), "permission status")
        versions = row["versions"]
        exact(versions, ("evidence", "labels", "rules"), "versions")
        require(all(nonempty(v) for v in versions.values()), "versions missing")
        inp = row["input"]
        exact(inp, ("views", "allowed_context", "regions", "door_candidates", "region_pairs", "view_pairs"), "input: truth must stay separate")
        require(isinstance(inp["allowed_context"], str), "allowed context")
        catalogs = {}
        for key in ("regions", "door_candidates", "views"):
            require(isinstance(inp[key], list), "candidate list")
            ids = []
            for item in inp[key]:
                fields = ("id", "description") if key == "regions" else (("id", "view_id", "box") if key == "door_candidates" else ("id", "asset_ref", "sha256"))
                exact(item, fields, key)
                require(nonempty(item["id"]), "local id")
                ids.append(item["id"])
                if key == "regions":
                    require(isinstance(item["description"], str), "description")
                elif key == "views":
                    require(nonempty(item["asset_ref"]) and sha(item["sha256"]), "asset reference/hash")
                    token = ("media_sha256", item["sha256"])
                    require(token not in owners or owners[token] == row["split"], "cross-split media leakage")
                    owners[token] = row["split"]
                else:
                    box = item["box"]
                    require(isinstance(box, list) and len(box) == 4 and all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1 for x in box), "normalized box")
                    require(box[0] < box[2] and box[1] < box[3], "box area")
            require(len(set(ids)) == len(ids), "duplicate local id")
            catalogs[key] = set(ids)
        require(all(d["view_id"] in catalogs["views"] for d in inp["door_candidates"]), "door view missing")
        for key in ("region_pairs", "view_pairs"):
            require(isinstance(inp[key], list), "pair list")
            ids, instances = [], set()
            for pair in inp[key]:
                exact(pair, ("id", "door_id", "region_a", "region_b") if key == "region_pairs" else ("id", "door_a", "door_b"), key)
                require(nonempty(pair["id"]), "pair id")
                ids.append(pair["id"])
                token = ((pair["door_id"], *sorted((pair["region_a"], pair["region_b"]))) if key == "region_pairs" else tuple(sorted((pair["door_a"], pair["door_b"]))))
                require(token not in instances, "duplicate candidate pair")
                instances.add(token)
                if key == "region_pairs":
                    require(pair["door_id"] in catalogs["door_candidates"] and pair["region_a"] in catalogs["regions"] and pair["region_b"] in catalogs["regions"] and pair["region_a"] != pair["region_b"], "region pair references")
                else:
                    require(pair["door_a"] in catalogs["door_candidates"] and pair["door_b"] in catalogs["door_candidates"] and pair["door_a"] != pair["door_b"], "view pair references")
            require(len(set(ids)) == len(ids), "duplicate pair")
        ai = row["ai_preannotation"]
        if ai is not None:
            exact(ai, ("model_revision", "status", "risk", "labels", "versions"), "ai")
            require(nonempty(ai["model_revision"]) and ai["status"] in ("proposed", "default_adopted_pending_review") and ai["risk"] in ("high", "normal"), "ai status")
            require(ai["versions"] == versions, "stale ai annotation")
            validate_labels(ai["labels"], inp)
        review = row["review"]
        exact(review, ("status", "reviewer", "evidence_refs", "versions"), "review")
        require(review["status"] in ("pending", "confirmed", "corrected"), "review status")
        require(isinstance(review["evidence_refs"], list) and all(nonempty(x) for x in review["evidence_refs"]), "evidence refs")
        if review["status"] == "pending":
            require(row["truth"] is None and review["reviewer"] is None, "pending is not truth")
            require(review["versions"] is None, "pending review versions")
        else:
            require(nonempty(review["reviewer"]) and review["evidence_refs"] and review["versions"] == versions, "review evidence/version required")
            require(source["permission_status"] == "verified", "truth needs permission")
            validate_labels(row["truth"], inp)
    return payload
