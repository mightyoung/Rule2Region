import json

from research_tools.fullframe_agent_review import N_PREDICT, parse_frame_review
from research_tools.parallel_agent_review import receipt_name


def test_frame_review_rejects_unknown_box_and_keeps_three_input_roles() -> None:
    allowed = {"f249b01", "f249b02"}
    parsed = parse_frame_review(
        json.dumps(
            {
                "abstain": False,
                "cells": [{"cell": "t3", "type": "EXCLUDE"}],
                "suspect_box_ids": ["f249b02"],
                "rationale": "far clutter",
            }
        ),
        allowed,
    )
    assert parsed["cells"][0]["type"] == "EXCLUDE"
    try:
        parse_frame_review(
            json.dumps(
                {
                    "abstain": False,
                    "cells": [],
                    "suspect_box_ids": ["f999b01"],
                    "rationale": "leak",
                }
            ),
            allowed,
        )
    except ValueError as exc:
        assert "suspect_box_id_not_in_frame_boxes" in str(exc)
    else:
        raise AssertionError("expected leaked box id to fail")
    assert receipt_name("A03", "f878").startswith("A03_f878_")
    assert N_PREDICT >= 768
