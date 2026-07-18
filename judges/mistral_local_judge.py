"""Local Mistral 7B judge run in-process via vLLM. See PRD v3.1."""

from __future__ import annotations

from judges.local_vllm_judge import LocalVLLMJudge


class MistralLocalJudge(LocalVLLMJudge):
    """Judge that runs local Mistral 7B in-process via vLLM."""

    def __init__(
        self,
        model_id: str = "unsloth/Mistral-7B-v0.3-GGUF",
        judge_id: str = "mistral-7b-local",
        **kwargs,
    ) -> None:
        super().__init__(model_id=model_id, judge_id=judge_id, **kwargs)
