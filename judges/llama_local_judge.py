"""Local Llama 2 7B judge served via vLLM (default :8001). See PRD v3.1."""

from __future__ import annotations

from judges.local_vllm_judge import LocalVLLMJudge


class LlamaLocalJudge(LocalVLLMJudge):
    """Judge that calls local Llama 2 7B via the vLLM API on localhost:8001."""

    def __init__(
        self,
        vllm_port: int = 8001,
        model_id: str = "unsloth/Llama-2-7b-GGUF",
        judge_id: str = "llama-7b-local",
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_port=vllm_port, model_id=model_id, judge_id=judge_id, **kwargs
        )
