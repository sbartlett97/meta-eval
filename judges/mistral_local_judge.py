"""Local Mistral 7B judge served via vLLM (default :8000). See PRD v3.1."""

from __future__ import annotations

from judges.local_vllm_judge import LocalVLLMJudge


class MistralLocalJudge(LocalVLLMJudge):
    """Judge that calls local Mistral 7B via the vLLM API on localhost:8000."""

    def __init__(
        self,
        vllm_port: int = 8000,
        model_id: str = "unsloth/Mistral-7B-v0.3-GGUF",
        judge_id: str = "mistral-7b-local",
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_port=vllm_port, model_id=model_id, judge_id=judge_id, **kwargs
        )
