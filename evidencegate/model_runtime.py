from __future__ import annotations

import re
from typing import Any


_JSON_COUNT = re.compile(r'\{\s*"count"\s*:\s*(0|[1-9][0-9]*)\s*\}')


def render_prompt(prompt_config: dict[str, Any], modality: str, *, question: str | None = None) -> tuple[str, str]:
    contract = prompt_config[modality]
    if modality == "image":
        if not question:
            raise ValueError("image_question_required")
        prompt = str(contract["template"]).format(question=question)
    elif modality == "video":
        prompt = str(contract["template"])
    else:
        raise ValueError("unsupported_prompt_modality")
    return prompt, str(contract.get("assistant_prefix", ""))


def parse_model_output(value: str, parser: str) -> bool | int:
    text = value.strip()
    if parser == "exact_case_insensitive_yes_or_no":
        if text.lower() not in {"yes", "no"}:
            raise ValueError("value_must_be_exactly_yes_or_no")
        return text.lower() == "yes"
    if parser == "exact_json_single_count_0_99":
        match = _JSON_COUNT.fullmatch(text)
        if not match:
            raise ValueError("value_must_be_exact_json_count")
        count = int(match.group(1))
        if count > 99:
            raise ValueError("json_count_outside_0_99")
        return count
    raise ValueError("unsupported_output_parser")
