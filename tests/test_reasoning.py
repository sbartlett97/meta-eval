"""Tests for harness.reasoning.split_reasoning (network-free)."""

from harness.reasoning import split_reasoning


def test_template_opened_think_block():
    # The common shape: the chat template opened <think>, the reply closes it.
    s = split_reasoning("plan the JSON\n</think>\n\n{\"a\": 1}", template_opens_think=True)
    assert s.reasoning == "plan the JSON"
    assert s.answer == '{"a": 1}'
    assert s.reasoning_complete is True


def test_model_opened_think_block():
    s = split_reasoning("<think>hmm</think>Paris.")
    assert (s.reasoning, s.answer, s.reasoning_complete) == ("hmm", "Paris.", True)


def test_thinking_tag_variant_and_case():
    s = split_reasoning("<Thinking>x</THINKING> done")
    assert (s.reasoning, s.answer) == ("x", "done")


def test_cut_off_before_closing_tag_with_template_flag_is_unfinished():
    # No tags at all: for a reasoning model this is a trace cut off by max_tokens.
    s = split_reasoning("Thinking Process:\n1. Analyze the request", template_opens_think=True)
    assert s.reasoning.startswith("Thinking Process")
    assert s.answer == ""
    assert s.reasoning_complete is False


def test_cut_off_after_explicit_open_tag_is_unfinished():
    s = split_reasoning("<think>still going")
    assert (s.reasoning, s.answer, s.reasoning_complete) == ("still going", "", False)


def test_plain_reply_from_a_non_reasoning_model_is_untouched():
    s = split_reasoning("Paris is the capital.")
    assert (s.reasoning, s.answer, s.reasoning_complete) == (None, "Paris is the capital.", None)


def test_last_closing_tag_wins():
    # A trace that mentions the tag literally must not cut the answer short.
    s = split_reasoning("I will write </think> later</think>Answer")
    assert s.answer == "Answer"
    assert s.reasoning == "I will write </think> later"


def test_empty_reply():
    s = split_reasoning("", template_opens_think=True)
    assert (s.reasoning, s.answer, s.reasoning_complete) == (None, "", None)
