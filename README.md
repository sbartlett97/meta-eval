# meta-eval

A meta safety-eval framework that assesses the effects of fine-tuning and
quantisation on model safety, and evaluates the evaluators to determine if any
common LLM-as-a-judge models are biased on safety-critical tasks.

This repository implements the **core scaffolding** for the *Behavioral
Invariance Eval Harness + Meta-Evaluation Infrastructure* (PRD v3.1). It targets
consumer hardware (MacBook Pro M5, 32GB) using **pre-quantized checkpoints**
served via **vLLM** (or Ollama), with **all judges called through a unified
HTTP/API interface**.

> **Status.** This is AI-generated scaffolding for review. Items marked
> **[Sam]** below are intentionally left as decisions / manual work (test-suite
> content, the substantive meta-evaluation methodology, and on-hardware
> verification of vLLM flags and memory budgets).

## Architecture

```
Pre-quantized checkpoints (HF/Unsloth GGUF)
        │
        ▼
vLLM servers (:8000 Mistral, :8001 Llama)  ── served by harness/vllm_server_manager.py
        │
        ▼
Test Runner ──> model outputs ──> Judge Panel ──> verdicts + consensus
                                    ├─ Claude   (Anthropic API)
                                    ├─ GPT-4o   (OpenAI API)
                                    ├─ Mistral  (local vLLM :8000)
                                    ├─ Llama    (local vLLM :8001)
                                    └─ Heuristic (deterministic, in-process)
```

Every judge — remote or local — implements the same
`Judge.evaluate(test_id, model_output, criteria) -> Verdict` contract, so the
panel calls Claude, a local vLLM model, or the heuristic baseline through
identical code.

## Layout

```
harness/
  vllm_server_manager.py   # start/stop/health-check local vLLM servers  [Sam review]
  model_loader.py          # load models UNDER TEST (Ollama / vLLM / cloud)
  test_suite.py            # test-suite schema + .jsonl loader
  test_runner.py           # run a suite against models, persist outputs
  evaluator.py             # CLI: generate outputs + run judge panel
judges/
  base.py                  # Judge ABC + Verdict types
  prompts.py               # shared judge prompt + JSON response parser
  claude_judge.py          # Anthropic API judge
  gpt4_judge.py            # OpenAI API judge
  local_vllm_judge.py      # base for local vLLM judges
  mistral_local_judge.py   # local Mistral judge (:8000)
  llama_local_judge.py     # local Llama judge (:8001)
  heuristic_judge.py       # deterministic baseline
  factory.py               # build judges from config/judges.yaml
  judge_panel.py           # orchestrate the panel + naive consensus
config/
  models.yaml              # local + remote model catalogue (checkpoint-driven)
  judges.yaml              # judge panel definition (priority/enabled)
  hardware_profile.yaml    # Mac/vLLM launch defaults + lifecycle timeouts
data/
  test_suite_v1.jsonl      # EXAMPLE schema only — full 15-test suite is [Sam]
tests/                     # unit tests for the pure-Python components
```

## Setup

```bash
pip install -r requirements.txt

# Local model serving — pick one (PRD "Decision 1"):
#   Ollama (recommended on Apple Silicon)
brew install ollama && ollama pull mistral:7b && ollama pull llama2:7b
#   or vLLM (more control)
pip install vllm

# API keys for remote judges
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

## Usage

```bash
# Start local vLLM judge servers (foreground; Ctrl-C to stop)
python harness/vllm_server_manager.py start

# Generate outputs for a model, then judge with the full panel
python harness/evaluator.py --models mistral-7b-4bit --judges all

# Or judge previously-generated outputs with a cheap (API-only) subset
python harness/evaluator.py --outputs results/model_outputs.jsonl --judges cheap
```

Results are written to `results/model_verdicts.jsonl` (one row per test/model
with each judge's verdict plus a placeholder consensus).

## Tests

```bash
pytest
```

The tests cover the deterministic, network-free parts (prompt formatting + JSON
parsing, verdict normalisation, base-class error handling, the heuristic judge,
the suite loader, the judge factory, and panel aggregation).

## What is intentionally NOT done here — **[Sam]**

- **The 15-test behavioral-invariance suite.** `data/test_suite_v1.jsonl` is a
  3-row schema example only.
- **The real meta-evaluation methodology.** `judges.judge_panel.aggregate_verdicts`
  is a naive majority vote so the pipeline runs end-to-end; inter-judge
  agreement, bias detection, and reliability weighting are yours to design.
- **On-hardware verification.** Confirm the vLLM CLI flags in
  `config/hardware_profile.yaml`, the 32GB memory budget, startup time, and
  graceful shutdown on the actual M5 (PRD "Sam Must Review" checklist).
- **Cloud fallback wiring.** `harness.model_loader.ReplicateModel.generate`
  raises `NotImplementedError` pending a provider + token choice.
