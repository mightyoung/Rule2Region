from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from research_tools.qwen_business_profile_mapper import load_blind_cases, load_cards, rule_map


ABSTAIN_LABEL = "ABSTAIN"


def _label(profile_code: str | None) -> str:
    return profile_code if profile_code is not None else ABSTAIN_LABEL


def _text(card: dict[str, Any]) -> str:
    return "".join(row["text"] for row in card["spans"])


def evaluate_baselines(
    *,
    development_cards_path: Path,
    blind_cards_path: Path,
    gold_path: Path,
    qwen_result_path: Path,
) -> dict[str, Any]:
    train = load_cards(development_cards_path)
    test = load_blind_cases(blind_cards_path, gold_path)
    train_text = [_text(row) for row in train]
    train_y = [_label(row["gold_profile_code"]) for row in train]
    test_text = [_text(row) for row in test]
    test_y = [_label(row["gold_profile_code"]) for row in test]
    labels = sorted(set(train_y) | set(test_y))

    linear = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, norm="l2")),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    linear.fit(train_text, train_y)
    linear_pred = linear.predict(test_text).tolist()

    retrieval_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, norm="l2")
    train_matrix = retrieval_vectorizer.fit_transform(train_text)
    test_matrix = retrieval_vectorizer.transform(test_text)
    similarities = (test_matrix @ train_matrix.T).toarray()
    retrieval_pred = [train_y[int(index)] for index in np.argmax(similarities, axis=1)]
    retrieval_max_similarity = [float(value) for value in np.max(similarities, axis=1)]

    rule_pred = [_label(rule_map(row)["profile_code"]) for row in test]
    qwen_result = json.loads(qwen_result_path.read_text(encoding="utf-8"))
    qwen_by_card = {}
    for row in qwen_result["rows"]:
        qwen_by_card.setdefault(row["card_id"], row["predicted_label"])
    qwen_pred = [qwen_by_card[row["card_id"]] for row in test]

    def metrics(pred: list[str]) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(test_y, pred)),
            "macro_f1": float(f1_score(test_y, pred, labels=labels, average="macro", zero_division=0)),
        }

    rows = []
    for index, card in enumerate(test):
        rows.append({
            "card_id": card["card_id"],
            "gold_label": test_y[index],
            "qwen_label": qwen_pred[index],
            "rule_label": rule_pred[index],
            "char_linear_label": linear_pred[index],
            "char_retrieval_label": retrieval_pred[index],
            "char_retrieval_max_similarity": retrieval_max_similarity[index],
        })
    result = {
        "schema_version": "sceneconfig.business-profile-postblind-baseline-diagnostic.v1",
        "evidence_status": "DEVELOPMENT_ONLY_POST_BLIND_DIAGNOSTIC",
        "holdout_opened": False,
        "training_card_count": len(train),
        "blind_card_count": len(test),
        "warning": "Baseline family was implemented after blind-set outcomes were observed; results are diagnostic, not confirmatory.",
        "metrics": {
            "qwen": metrics(qwen_pred),
            "keyword_rule": metrics(rule_pred),
            "char_ngram_logistic_regression": metrics(linear_pred),
            "char_ngram_nearest_development_card": metrics(retrieval_pred),
        },
        "rows": rows,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-cards", type=Path, required=True)
    parser.add_argument("--blind-cards", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    result = evaluate_baselines(
        development_cards_path=args.development_cards,
        blind_cards_path=args.blind_cards,
        gold_path=args.gold,
        qwen_result_path=args.qwen_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
