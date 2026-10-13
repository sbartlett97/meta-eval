"""Tests for the deterministic format-adherence checks (judges/format_checks.py).

docs/critical-assessment.md #2.6: these back HeuristicJudge's format-adherence
dispatch and *are* the check for mechanically-verifiable criteria, not a proxy.
"""

from judges import format_checks as fc


def test_check_json_valid_with_required_keys():
    ok, note = fc.check_json('{"name": "Alice", "age": 30}', ["name", "age"])
    assert ok is True
    assert "all required keys" in note


def test_check_json_invalid_syntax():
    ok, note = fc.check_json("not json", [])
    assert ok is False
    assert "not valid JSON" in note


def test_check_json_missing_required_key():
    ok, note = fc.check_json('{"name": "Alice"}', ["name", "age"])
    assert ok is False
    assert "age" in note


def test_check_json_strips_code_fences():
    ok, _ = fc.check_json('```json\n{"a": 1}\n```', ["a"])
    assert ok is True


def test_check_json_rejects_non_object():
    ok, note = fc.check_json("[1, 2, 3]", [])
    assert ok is False
    assert "not a JSON object" in note


def test_check_sentence_count_exact():
    ok, note = fc.check_sentence_count("One. Two. Three.", 3)
    assert ok is True
    assert "found 3" in note


def test_check_sentence_count_mismatch():
    ok, note = fc.check_sentence_count("One. Two.", 3)
    assert ok is False
    assert "found 2" in note


def test_check_markdown_structure_valid():
    text = "# Title\n## Sub\n- one\n- two\n"
    ok, note = fc.check_markdown_structure(text, min_bullets=2)
    assert ok is True


def test_check_markdown_structure_missing_h2():
    text = "# Title\n- one\n- two\n"
    ok, note = fc.check_markdown_structure(text, min_bullets=2)
    assert ok is False
    assert "h2=False" in note


def test_check_csv_valid():
    text = "Product,Price,Stock\nApple,1,50\nBanana,0.5,100\n"
    ok, note = fc.check_csv(text, ["Product", "Price", "Stock"])
    assert ok is True
    assert "2 data row(s)" in note


def test_check_csv_wrong_headers():
    text = "Item,Cost\nApple,1\n"
    ok, note = fc.check_csv(text, ["Product", "Price", "Stock"])
    assert ok is False


def test_check_python_syntax_valid_with_signature():
    text = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
    ok, note = fc.check_python_syntax(text, must_contain="def factorial(n):")
    assert ok is True


def test_check_python_syntax_invalid():
    ok, note = fc.check_python_syntax("def broken(:\n    pass")
    assert ok is False
    assert "invalid Python syntax" in note


def test_check_python_syntax_missing_required_signature():
    ok, note = fc.check_python_syntax("def other():\n    pass", must_contain="def factorial(n):")
    assert ok is False
    assert "missing required text" in note


def test_check_sections_present_all_found():
    text = "## Parties\n...\n## Definitions\n...\n## Confidentiality\n...\n## Term\n..."
    ok, note = fc.check_sections_present(text, ["Parties", "Definitions", "Confidentiality", "Term"])
    assert ok is True


def test_check_sections_present_missing_some():
    text = "## Parties\n...\n## Term\n..."
    ok, note = fc.check_sections_present(text, ["Parties", "Definitions", "Term"])
    assert ok is False
    assert "Definitions" in note


def test_check_email_structure_valid():
    text = (
        "Subject: Hello\nTo: recipient@example.com\nBody:\n"
        "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nSignature: Alice"
    )
    ok, note = fc.check_email_structure(text, min_paragraphs=3)
    assert ok is True


def test_check_email_structure_missing_subject():
    text = "To: recipient@example.com\nBody:\nHello there."
    ok, note = fc.check_email_structure(text)
    assert ok is False
    assert "subject=False" in note


def test_checks_registry_covers_all_shipped_hints():
    # Every metadata.check value used in data/test_suite_v1.json must resolve.
    import json

    suite = json.load(open("data/test_suite_v1.json"))
    used = {
        e["metadata"]["check"]
        for e in suite["evals"]
        if e.get("metadata", {}).get("check")
    }
    assert used  # sanity: the suite actually uses this mechanism
    assert used <= set(fc.CHECKS)
