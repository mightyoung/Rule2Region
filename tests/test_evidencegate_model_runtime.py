import pytest

from evidencegate.model_runtime import parse_model_output, render_prompt


def test_prompts_and_strict_parsers_are_deterministic():
    config = {
        "image": {"template": "{question}\nAnswer exactly yes or no."},
        "video": {"template": "Count people.", "assistant_prefix": '{"count": '},
    }
    assert render_prompt(config, "image", question="Is there a person?")[0].startswith("Is there a person?")
    assert render_prompt(config, "video") == ("Count people.", '{"count": ')
    assert parse_model_output(" YES ", "exact_case_insensitive_yes_or_no") is True
    assert parse_model_output('{"count": 12}', "exact_json_single_count_0_99") == 12
    with pytest.raises(ValueError, match="exact_json"):
        parse_model_output("count: 12", "exact_json_single_count_0_99")
    with pytest.raises(ValueError, match="outside"):
        parse_model_output('{"count": 100}', "exact_json_single_count_0_99")
