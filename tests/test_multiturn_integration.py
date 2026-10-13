"""End-to-end: a multi-turn test case's full exchange actually reaches the judge.

docs/critical-assessment.md #2.4 fixed multi-turn cases to be real
conversations (TestRunner -> model.generate_conversation -> transcript); this
proves the chain all the way through run_evaluation to what a judge sees --
not just that chat_turns()/generate_conversation() build the right messages
(covered in tests/test_llamacpp_engine.py) but that the *judge* actually gets
the transcript, not just the final reply, for the exact case shape used by
data/test_suite_v1.json's converted "[Follow-up]" cases.
"""

import json

from harness.evaluator import run_evaluation
from harness.test_runner import TestRunner as Runner
from judges.base import Judge, Verdict


class _ConversingModel:
    """A fake model that answers differently depending on prior turns, so a
    test that only looked at the final reply (not the full transcript) would
    be unable to tell these turns were ever connected.
    """

    id = "m"

    def generate_conversation(self, turns, system_prompt=None, **kwargs):
        replies = []
        for i, turn in enumerate(turns):
            replies.append(f"reply-{i}-to:{turn}")
        return replies


class _RecordingJudge(Judge):
    id = "recorder"

    def __init__(self):
        super().__init__()
        self.seen_outputs = []

    def _evaluate(self, item):
        test_id, model_output, criteria = item.test_id, item.model_output, item.criteria
        self.seen_outputs.append(model_output)
        return Verdict("PASS", 1.0, self.id, test_id)


class _FakeLoader:
    def load(self, model_id):
        return _ConversingModel()


def test_judge_receives_full_transcript_not_just_final_reply(tmp_path, monkeypatch):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps({"evals": [
        {
            "id": "consistency-01",
            "turns": ["What is the capital of France?", "Are you sure about that?"],
            "criteria": "expect:PASS Second answer must stay consistent with the first.",
            "category": "reasoning-coherence",
        },
    ]}))
    outputs_path = tmp_path / "outputs.json"

    runner = Runner(str(suite_path), _FakeLoader(), results_path=str(outputs_path))
    runner.run_tests(["m"])

    outputs = json.loads(outputs_path.read_text())["outputs"]
    (row,) = outputs
    assert row["output"] == "reply-1-to:Are you sure about that?"  # final reply only
    assert "reply-0-to:What is the capital of France?" in row["transcript"]
    assert "reply-1-to:Are you sure about that?" in row["transcript"]

    # Now judge it -- the judge must see BOTH turns, not just the final reply,
    # or a consistency criterion has nothing to actually check.
    recording_judge = _RecordingJudge()
    monkeypatch.setattr(
        "harness.evaluator.build_judges", lambda *a, **k: [recording_judge]
    )
    verdicts_path = tmp_path / "verdicts.json"
    run_evaluation(
        outputs_path=str(outputs_path),
        judges_config={"judges": []},  # ignored; build_judges is monkeypatched
        verdicts_path=str(verdicts_path),
        hardware_profile={},
        meta_eval_path=None,
    )

    (seen,) = recording_judge.seen_outputs
    assert "What is the capital of France?" in seen
    assert "Are you sure about that?" in seen
    assert "reply-0-to:What is the capital of France?" in seen
    assert "reply-1-to:Are you sure about that?" in seen
    # Critically, this is NOT just the final output string.
    assert seen != row["output"]
