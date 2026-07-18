"""Local Llama 2 7B judge run in-process via vLLM. See PRD v3.1."""

from __future__ import annotations

from judges.local_vllm_judge import LocalVLLMJudge


class LlamaLocalJudge(LocalVLLMJudge):
    """Judge that runs local Llama 2 7B in-process via vLLM."""

    def __init__(
        self,
        model_id: str = "unsloth/Llama-2-7b-GGUF",
        judge_id: str = "llama-7b-local",
        **kwargs,
    ) -> None:
        super().__init__(model_id=model_id, judge_id=judge_id, **kwargs)
