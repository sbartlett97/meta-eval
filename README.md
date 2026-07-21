# meta-eval

A meta safety-eval framework that assesses the effects of fine-tuning and
quantisation on model safety, and evaluates the evaluators to determine if any
common LLM-as-a-judge models are biased on safety-critical tasks.

This repository implements the **core scaffolding** for the *Behavioral
Invariance Eval Harness + Meta-Evaluation Infrastructure* (PRD v3.1). It targets
consumer hardware (MacBook Pro M5, 32GB) using **pre-quantized GGUF checkpoints**.
Local models and judges are loaded **in-process via llama.cpp** (no separate
server, no HTTP hop) — pinned to a specific GGUF quant and loaded sequentially;
remote judges are called through their hosted APIs. Every judge implements the
same `Judge.evaluate(...)` contract.

> **Status.** This is AI-generated scaffolding for review. Items marked
> **[Sam]** below are intentionally left as decisions / manual work (test-suite
> content, the substantive meta-evaluation methodology, and on-hardware
> verification of the llama.cpp engine settings and memory budgets).

## Architecture

```
Pre-quantized checkpoints (HF/Unsloth GGUF)
        │
        ▼   hydration: harness/hydrate.py downloads open weights from HF at startup
        │
        ▼   in-process llama.cpp engines (harness/llamacpp_engine.py) — lazy, sequential
        │
        ▼
Test Runner ──> model outputs ──> Judge Panel ──> verdicts + consensus
                                    ├─ Claude   (Anthropic API)
                                    ├─ GPT-4o   (OpenAI API)
                                    ├─ Mistral  (in-process llama.cpp)
                                    ├─ Llama    (in-process llama.cpp)
                                    └─ Heuristic (deterministic, in-process)
```

Generation and judging are **two separate passes** writing to two separate
files (`results/model_outputs.jsonl` then `results/model_verdicts.jsonl`), so
expensive test-model inference is done once and can be re-judged — or skipped
entirely (see [Generate outputs only](#generate-outputs-only-no-judges)).

Every judge — remote or local — implements the same
`Judge.evaluate(test_id, model_output, criteria) -> Verdict` contract, so the
panel calls Claude, a local in-process llama.cpp model, or the heuristic baseline
through identical code. Local models and judges that reference the same GGUF
reuse a **single in-process engine**, and the engine cache loads models
**sequentially** (one resident at a time by default) so a test model and the
judges never all hold weights at once.

## Layout

```
harness/
  llamacpp_engine.py       # in-process llama.cpp GGUF engines (lazy, sequential)
  hydrate.py               # download open weights from HuggingFace at startup
  model_loader.py          # load models UNDER TEST (llama.cpp / Ollama / cloud)
  test_suite.py            # test-suite schema + .jsonl loader
  test_runner.py           # run a suite against models, persist outputs
  evaluator.py             # CLI: hydrate + generate outputs + run judge panel
judges/
  base.py                  # Judge ABC + Verdict types
  prompts.py               # shared judge prompt + JSON response parser
  claude_judge.py          # Anthropic API judge
  gpt4_judge.py            # OpenAI API judge
  local_llamacpp_judge.py  # local in-process llama.cpp (GGUF) judge
  heuristic_judge.py       # deterministic baseline
  factory.py               # build judges from config/judges.yaml
  judge_panel.py           # orchestrate the panel + naive consensus
config/
  models.yaml              # local + remote model catalogue (checkpoint-driven)
  judges.yaml              # judge panel definition (priority/enabled)
  hardware_profile.yaml    # llama.cpp engine defaults + resident-model cap
data/
  test_suite_v1.jsonl      # EXAMPLE schema only — full 15-test suite is [Sam]
tests/                     # unit tests for the pure-Python components
```

## Setup

```bash
# 1. Core install (pure-Python wheels; installs cleanly on every platform).
#    Includes huggingface_hub, used by the startup weight-hydration step.
pip install -r requirements.txt

# 2. Local model serving.
#
#    llama.cpp (loaded IN-PROCESS; required for any `engine: llamacpp` /
#    `access: llamacpp` model or judge — no server to start). It builds its own
#    bundled llama.cpp at install time; for a Metal build on Apple Silicon:
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
#    Ollama is also supported for models under test (over HTTP, auto-managed):
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
# In-process llama.cpp test model — nothing to start by hand; weights are
# hydrated, then loaded in-process (sequentially) on first use:
python harness/evaluator.py --models mistral-7b-4bit --judges all

# Ollama-backed test model (for a model whose serving.engine is `ollama`):
python harness/evaluator.py --models mistral-7b-4bit --judges all --prefer-engine ollama
```

`--judges` selects a preset: `all` (every enabled judge), `cheap` (priority ≤ 1,
i.e. the remote API judges only), or `pilot`.

On startup the harness **hydrates**: it downloads the open-weight checkpoints
this run needs from HuggingFace (the requested `--models` plus the local judges
actually selected — see [Weight hydration](#weight-hydration)). Pass
`--no-hydrate` to skip it entirely.

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

At startup the harness downloads the **open weights this run needs** from
HuggingFace so the first in-process llama.cpp load doesn't block mid-run on a
multi-gigabyte download. Hydration is scoped to the run:

- the models named in `--models` (nothing when you only re-judge), and
- the local (`access: llamacpp`) judges that will actually be built — i.e. after
  the `--judges` preset and `enabled` are applied, and skipped entirely under
  `--no-judge`.

So `--outputs ... --judges cheap` (remote judges only, no `--models`) downloads
**nothing**; `--models mistral-7b-4bit --judges cheap` pulls only the Mistral
checkpoint. Cloud/API models (Anthropic, OpenAI, Replicate) are always skipped.

`huggingface_hub.snapshot_download` is idempotent, so after the first run
hydration only verifies the cache. Skip it with `--no-hydrate`, or run it
standalone (no scoping filters → downloads everything in the configs):

```bash
python harness/hydrate.py                         # download all open weights now
python harness/hydrate.py --models mistral-7b-4bit # just this model's weights
python harness/hydrate.py --dry-run               # list what would be downloaded
```

Gated repos (e.g. `unsloth/Llama-2-7b-GGUF`) need `HF_TOKEN` set. A `llamacpp`
entry's `gguf_file` scopes the download to a single file. Prefer the **exact**
GGUF file name (as in HuggingFace's "Use this model" snippet), e.g.
`gguf_file: "Mistral-7B-v0.3.Q4_K_M.gguf"`: hydration then fetches exactly that
file with `hf_hub_download` and **fails loudly if the name is wrong**, instead of
a glob like `"*Q4_K_M.gguf"` that can silently match nothing. `additional_files`
fetches extra parts (split-GGUF shards); `hf_allow_patterns` overrides both.

## The serving backend: in-process llama.cpp

Local models and judges are served **in-process via llama.cpp**
(`harness/llamacpp_engine.py`). It is GGUF-native, so each engine is pinned to a
**specific GGUF file** in a HuggingFace repo via `serving.gguf_file` (models) /
`gguf_file` (judges), passed straight to `Llama.from_pretrained(repo_id,
filename=...)` — normally the exact file name (a glob is also accepted). Install
with `pip install llama-cpp-python` (it builds its own bundled llama.cpp; it does
**not** reuse a system install).

**Sequential loading.** The engine cache holds at most `llamacpp.max_resident`
models in memory at once (`config/hardware_profile.yaml`, default **1**). Loading
a new model frees the least-recently-used one (`Llama.close()`); an evicted
engine transparently reloads on next use. So a test model and the judges never
all hold weights simultaneously. Raise `max_resident` to keep both local judges
resident if the memory budget allows.

To keep this cheap, the judge panel evaluates **judge-outer** (`JudgePanel.
evaluate_batch`): every judge scores the whole batch of outputs before the next
judge runs, so each local judge loads once per run rather than reloading for each
row.

```yaml
# config/models.yaml
local_models:
  - id: "mistral-7b-4bit"
    checkpoint: "unsloth/Mistral-7B-v0.3-GGUF"   # HF repo
    serving:
      engine: "llamacpp"          # llamacpp (in-process GGUF) | ollama
      gguf_file: "*Q4_K_M.gguf"   # the exact quant to load
      ollama_tag: "mistral:7b"    # only used if engine: ollama
```

### Ollama (models under test only)

A model whose `serving.engine` is `ollama` is generated via the local Ollama
daemon over HTTP instead. `model_loader.py` starts `ollama serve` (if down) and
pulls the `ollama_tag` on first use, so there is nothing to launch by hand. The
`--prefer-engine {llamacpp,ollama}` flag only sets the fallback for entries that
don't name a `serving.engine`; a model that pins its engine wins. The judge panel
is llama.cpp only — there is no Ollama judge backend.

### Do I have to start anything manually?

No. Both local backends are self-managing:

| Backend | Startup |
| --- | --- |
| **in-process llama.cpp** | **Automatic.** `harness/llamacpp_engine.py` loads the GGUF in-process on first `generate` (and unloads it when another model needs the slot). Nothing to launch. |
| **Ollama** | **Automatic.** `model_loader.py` starts the `ollama serve` daemon (if down) and pulls the tag on first use. |

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
- **On-hardware verification.** Confirm the in-process llama.cpp engine settings
  in `config/hardware_profile.yaml` (`n_ctx`, `n_gpu_layers`, `n_batch`), the
  `llamacpp.max_resident` cap, the 32GB memory budget, and load time on the
  actual M5 (PRD "Sam Must Review" checklist).
- **Cloud fallback wiring.** `harness.model_loader.ReplicateModel.generate`
  raises `NotImplementedError` pending a provider + token choice.
