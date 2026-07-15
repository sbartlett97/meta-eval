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

Generation and judging are **two separate passes** writing to two separate
files (`results/model_outputs.jsonl` then `results/model_verdicts.jsonl`), so
expensive test-model inference is done once and can be re-judged — or skipped
entirely (see [Generate outputs only](#generate-outputs-only-no-judges)).

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
# 1. Core install (pure-Python wheels; installs cleanly on every platform).
pip install -r requirements.txt

# 2. Local model serving — pick ONE backend (PRD "Decision 1").
#    Neither is a hard dependency; the harness shells out to `ollama` / `vllm`.
#
#    Ollama (recommended on Apple Silicon; auto-managed by the harness):
brew install ollama
#    or vLLM (more control; you start the servers yourself — see Usage):
pip install "vllm>=0.3.0"

# 3. API keys for the remote judges (only needed if you run those judges).
export ANTHROPIC_API_KEY=...   # Claude judge
export OPENAI_API_KEY=...       # GPT-4o judge
```

You do **not** need to pre-`ollama pull` the models — when a test model uses the
Ollama backend, `harness/model_loader.py` starts the `ollama serve` daemon (if
it isn't already up) and pulls the model tag on demand.

## Usage

The entrypoint is `harness/evaluator.py`. Model ids come from
`config/models.yaml` (e.g. `mistral-7b-4bit`, `llama-2-7b-4bit`); judge subsets
come from `config/judges.yaml`.

### Full run: generate outputs, then judge

```bash
# Ollama-backed test model — no server to start by hand:
python harness/evaluator.py --models mistral-7b-4bit --judges all --prefer-engine ollama

# vLLM-backed test model — start the local server(s) FIRST (see below):
python harness/evaluator.py --models mistral-7b-4bit --judges all
```

`--judges` selects a preset: `all` (every enabled judge), `cheap` (priority ≤ 1,
i.e. the remote API judges only), or `pilot`.

### Judge previously-generated outputs

```bash
python harness/evaluator.py --outputs results/model_outputs.jsonl --judges cheap
```

### Generate outputs only (no judges)

To run the eval questions through a model **without** invoking the judge panel,
pass `--no-judge`:

```bash
python harness/evaluator.py --models mistral-7b-4bit --no-judge
```

This runs the suite against the model and writes only
`results/model_outputs.jsonl` (one row per test/model — prompt, category,
criteria, raw model output). No judge is built, so **no API keys and no judge
servers are required**. `--no-judge` requires `--models` (there is nothing to
generate otherwise).

Results of a full run are written to `results/model_verdicts.jsonl` (one row per
test/model with each judge's verdict plus a placeholder consensus).

## Choosing the serving backend: Ollama vs vLLM

The backend for each **model under test** is set per-model in
`config/models.yaml` (it is YAML, not JSON) under `serving.engine`. To load the
local models with **Ollama instead of vLLM**, edit that field for each
`local_models` entry:

```yaml
local_models:
  - id: "mistral-7b-4bit"
    checkpoint: "unsloth/Mistral-7B-v0.3-GGUF"
    serving:
      engine: "ollama"          # was: "vllm"
      vllm_port: 8000
      ollama_tag: "mistral:7b"  # tag Ollama pulls/serves
```

This **is manual** — you edit the config file. The `ollama_tag` is already
present for the shipped models, so switching `engine` is the only change needed.

> **Note on `--prefer-engine`.** The CLI accepts `--prefer-engine {ollama,vllm}`,
> but a model's `serving.engine` in `config/models.yaml` currently takes
> precedence, so the flag does not override a model that is pinned to a specific
> engine. Treat editing `serving.engine` as the source of truth.

### Do I have to start the servers manually?

It depends on the backend:

| Backend | Who starts the server? |
| --- | --- |
| **Ollama** | **Automatic.** `model_loader.py` starts the `ollama serve` daemon (if down) and pulls the tag on first use. Nothing to launch by hand. |
| **vLLM** | **Manual.** The harness does **not** start vLLM in-process. `VLLMModel.generate` just POSTs to `http://localhost:<port>/v1/completions` and assumes a server is already listening there. |

For the vLLM path, start the servers yourself before generating:

```bash
# Starts a vLLM server for every local_models entry whose serving.engine is
# "vllm", on its configured vllm_port. Runs in the FOREGROUND; Ctrl-C stops
# them (the servers live only as long as this process).
python harness/vllm_server_manager.py start

# In another terminal, check health / list managed ports:
python harness/vllm_server_manager.py status
```

Run `evaluator.py` from a second terminal while the servers stay up. The same
vLLM servers on `:8000` / `:8001` also back the local **judges**
(`config/judges.yaml`, `access: vllm`), so a full run with local judges needs
them running too.

> The local **judge** backend is vLLM-only (`judges.yaml` uses `access: vllm`);
> there is no Ollama judge backend yet. Switching to Ollama as described above
> applies to the models **under test**, not the judge panel.

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
