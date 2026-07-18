# meta-eval

A meta safety-eval framework that assesses the effects of fine-tuning and
quantisation on model safety, and evaluates the evaluators to determine if any
common LLM-as-a-judge models are biased on safety-critical tasks.

This repository implements the **core scaffolding** for the *Behavioral
Invariance Eval Harness + Meta-Evaluation Infrastructure* (PRD v3.1). It targets
consumer hardware (MacBook Pro M5, 32GB) using **pre-quantized checkpoints**.
Local models and judges are loaded **in-process via the vLLM Python API** (no
separate server, no HTTP hop); remote judges are called through their hosted
APIs. Every judge implements the same `Judge.evaluate(...)` contract.

> **Status.** This is AI-generated scaffolding for review. Items marked
> **[Sam]** below are intentionally left as decisions / manual work (test-suite
> content, the substantive meta-evaluation methodology, and on-hardware
> verification of vLLM flags and memory budgets).

## Architecture

```
Pre-quantized checkpoints (HF/Unsloth GGUF)
        │
        ▼   hydration: harness/hydrate.py downloads open weights from HF at startup
        │
        ▼   in-process vLLM engines (harness/vllm_engine.py) — loaded lazily, shared
        │
        ▼
Test Runner ──> model outputs ──> Judge Panel ──> verdicts + consensus
                                    ├─ Claude   (Anthropic API)
                                    ├─ GPT-4o   (OpenAI API)
                                    ├─ Mistral  (in-process vLLM)
                                    ├─ Llama    (in-process vLLM)
                                    └─ Heuristic (deterministic, in-process)
```

Generation and judging are **two separate passes** writing to two separate
files (`results/model_outputs.jsonl` then `results/model_verdicts.jsonl`), so
expensive test-model inference is done once and can be re-judged — or skipped
entirely (see [Generate outputs only](#generate-outputs-only-no-judges)).

Every judge — remote or local — implements the same
`Judge.evaluate(test_id, model_output, criteria) -> Verdict` contract, so the
panel calls Claude, a local in-process vLLM model, or the heuristic baseline
through identical code. Local models and judges that share a checkpoint reuse a
**single in-process engine** (weights are loaded once, process-wide).

## Layout

```
harness/
  vllm_engine.py           # in-process vLLM engines (lazy, process-wide cache)
  hydrate.py               # download open weights from HuggingFace at startup
  model_loader.py          # load models UNDER TEST (Ollama / in-process vLLM / cloud)
  test_suite.py            # test-suite schema + .jsonl loader
  test_runner.py           # run a suite against models, persist outputs
  evaluator.py             # CLI: hydrate + generate outputs + run judge panel
judges/
  base.py                  # Judge ABC + Verdict types
  prompts.py               # shared judge prompt + JSON response parser
  claude_judge.py          # Anthropic API judge
  gpt4_judge.py            # OpenAI API judge
  local_vllm_judge.py      # base for local in-process vLLM judges
  mistral_local_judge.py   # local Mistral judge (in-process)
  llama_local_judge.py     # local Llama judge (in-process)
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
#    Includes huggingface_hub, used by the startup weight-hydration step.
pip install -r requirements.txt

# 2. Local model serving — pick ONE backend (PRD "Decision 1").
#
#    vLLM (loaded IN-PROCESS; required for any `engine: vllm` / `access: vllm`
#    model or judge — the harness imports vllm directly, no server to start):
pip install "vllm>=0.3.0"
#    or Ollama (recommended on Apple Silicon; talked to over HTTP, auto-managed):
brew install ollama

# 3. API keys for the remote judges (only needed if you run those judges).
export ANTHROPIC_API_KEY=...   # Claude judge
export OPENAI_API_KEY=...       # GPT-4o judge

# 4. (optional) HuggingFace token for gated repos (e.g. Llama-2) that hydration
#    downloads. Picked up automatically by huggingface_hub.
export HF_TOKEN=...
```

You do **not** need to pre-`ollama pull` the models — when a test model uses the
Ollama backend, `harness/model_loader.py` starts the `ollama serve` daemon (if
it isn't already up) and pulls the model tag on demand. Likewise you do **not**
need to pre-download HF weights: the harness hydrates them at startup (below).

## Usage

The entrypoint is `harness/evaluator.py`. Model ids come from
`config/models.yaml` (e.g. `mistral-7b-4bit`, `llama-2-7b-4bit`); judge subsets
come from `config/judges.yaml`.

### Full run: generate outputs, then judge

```bash
# In-process vLLM test model — nothing to start by hand; weights are hydrated,
# then loaded in-process on first use:
python harness/evaluator.py --models mistral-7b-4bit --judges all

# Ollama-backed test model:
python harness/evaluator.py --models mistral-7b-4bit --judges all --prefer-engine ollama
```

`--judges` selects a preset: `all` (every enabled judge), `cheap` (priority ≤ 1,
i.e. the remote API judges only), or `pilot`.

On startup the harness **hydrates**: it downloads every open-weight checkpoint
referenced by `config/models.yaml` + `config/judges.yaml` from HuggingFace. Pass
`--no-hydrate` to skip it (e.g. a `--judges cheap` run that needs no local
weights).

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
criteria, raw model output). No judge is built, so **no API keys are required**.
`--no-judge` requires `--models` (there is nothing to generate otherwise).

Results of a full run are written to `results/model_verdicts.jsonl` (one row per
test/model with each judge's verdict plus a placeholder consensus).

## Weight hydration

At startup the harness downloads all **open weights** from HuggingFace so the
first in-process vLLM load doesn't block mid-run on a multi-gigabyte download.
"Open weights" = every `checkpoint` on a locally-served `local_models` /
`fine_tuned_models` entry plus every `access: vllm` judge's `model`; cloud/API
models (Anthropic, OpenAI, Replicate) are skipped.

`huggingface_hub.snapshot_download` is idempotent, so after the first run
hydration only verifies the cache. Skip it with `--no-hydrate`, or run it
standalone:

```bash
python harness/hydrate.py            # download everything now
python harness/hydrate.py --dry-run  # just list what would be downloaded
```

Gated repos (e.g. `unsloth/Llama-2-7b-GGUF`) need `HF_TOKEN` set. To restrict a
GGUF repo to a single quant file, add `hf_allow_patterns` to its config entry
(e.g. `hf_allow_patterns: ["*Q4_K_M.gguf"]`).

## Choosing the serving backend: in-process vLLM vs Ollama

The backend for each **model under test** is set per-model in
`config/models.yaml` (it is YAML, not JSON) under `serving.engine`:

- `vllm` (default) — the checkpoint is loaded **in-process** via the vLLM Python
  API (`harness/vllm_engine.py`). No server, no port, no HTTP. vLLM must be
  installed (`pip install "vllm>=0.3.0"`).
- `ollama` — generation goes to the local Ollama daemon over HTTP. To use it,
  edit the field per `local_models` entry:

```yaml
local_models:
  - id: "mistral-7b-4bit"
    checkpoint: "unsloth/Mistral-7B-v0.3-GGUF"
    serving:
      engine: "ollama"          # was: "vllm"
      ollama_tag: "mistral:7b"  # tag Ollama pulls/serves
```

The `ollama_tag` is already present for the shipped models, so switching
`engine` is the only change needed.

> **Note on `--prefer-engine`.** The CLI accepts `--prefer-engine {ollama,vllm}`,
> but a model's `serving.engine` in `config/models.yaml` currently takes
> precedence, so the flag does not override a model that is pinned to a specific
> engine. Treat editing `serving.engine` as the source of truth.

### Do I have to start anything manually?

No. Both local backends are self-managing:

| Backend | Startup |
| --- | --- |
| **in-process vLLM** | **Automatic.** `harness/vllm_engine.py` loads the checkpoint in-process on first `generate` and caches the engine process-wide. Nothing to launch. |
| **Ollama** | **Automatic.** `model_loader.py` starts the `ollama serve` daemon (if down) and pulls the tag on first use. |

The local **judge** backend is in-process vLLM only (`judges.yaml` uses
`access: vllm`); a judge and a test model that share a checkpoint reuse the same
in-process engine. There is no Ollama judge backend yet; switching to Ollama as
described above applies to the models **under test**, not the judge panel.

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
- **On-hardware verification.** Confirm the in-process vLLM engine kwargs in
  `config/hardware_profile.yaml` (`gpu_memory_utilization`, `max_model_len`,
  GGUF `quantization`/`tokenizer` handling), the 32GB memory budget, and
  load time on the actual M5 (PRD "Sam Must Review" checklist).
- **Cloud fallback wiring.** `harness.model_loader.ReplicateModel.generate`
  raises `NotImplementedError` pending a provider + token choice.
