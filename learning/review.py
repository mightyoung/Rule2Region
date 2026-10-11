"""Pure station handoff helper; no station writes or publication."""
from .contract import fingerprint


def review_queue(samples):
    def pending(row):
        review = row["review"]
        return review["status"] == "pending" or review["versions"] != row["versions"]
    def priority(row):
        ai = row["ai_preannotation"]
        high = ai is None or ai["risk"] == "high" or any("unknown" in labels.values() for labels in ai["labels"].values())
        return (not high, row["id"])
    return [row["id"] for row in sorted(samples, key=priority) if pending(row)]


def review_key(row):
    return fingerprint({"id": row["id"], "versions": row["versions"]})
