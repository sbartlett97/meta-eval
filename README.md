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

> **Status.** This started as AI-generated scaffolding for review. A
> [critical assessment](docs/critical-assessment.md) found that several
> issues would have silently invalidated both halves of the project's goal —
> most notably, judges sharing weights with the models they scored, and the
> suite's `expect:PASS`/`expect:FAIL` directive leaking into the LLM judge
> prompt right next to the instruction to output a verdict. Those have been
> fixed (see [What was fixed](#what-was-fixed)). Items still marked
> **[Sam]** below are genuinely open decisions: on-hardware verification of
> the llama.cpp engine settings/memory budgets, ground-truth calibration (no
> gold-labeled set exists), and the cloud-fallback provider choice.

## Architecture

```
Pre-quantized checkpoints (HF/Unsloth/bartowski GGUF)
        │
        ▼   hydration: harness/hydrate.py downloads open weights from HF at startup
        │
        ▼   in-process llama.cpp engines (harness/llamacpp_engine.py) — lazy, sequential
        │
        ▼
Test Runner ──> model outputs ──> Judge Panel ──> verdicts + per-item consensus
                                    │                    │
                                    │                    ▼
                                    │              judges/meta_eval.py
                                    │              (inter-judge kappa,
                                    │               bias by category/
                                    │               technique/severity)
                                    ▼
                              harness/compare.py
                              (variant vs. base_model_id
                               behavioral-invariance diff)

Judge panel:
  ├─ Claude               (Anthropic API — trusted, independent of every model under test)
  ├─ GPT-4o                (OpenAI API — trusted, independent of every model under test)
  ├─ Mistral-7B-Instruct   (in-process llama.cpp — deliberately not a model under test)
  ├─ Llama-3.2-3B-Instruct (in-process llama.cpp — deliberately not a model under test)
  └─ Heuristic             (deterministic, in-process — refusal directive + real format checks)
```

No judge shares a checkpoint with any model under test — see
[Judge/model independence](#judgemodel-independence) below; this was not true
of the original scaffold and is the single most important invariant to
preserve when editing either config.

Generation and judging are **two separate passes** writing to two separate JSON
files (`results/model_outputs.json` then `results/model_verdicts.json` — rows
nested under `outputs` / `verdicts`), so expensive test-model inference is done
once and can be re-judged — or skipped entirely (see
[Generate outputs only](#generate-outputs-only-no-judges)). Judging also
writes `results/judge_meta_eval.json` — the actual judge meta-evaluation (see
[Judge meta-evaluation](#judge-meta-evaluation)).

Every judge — remote or local — implements the same
`Judge.evaluate(item: EvalItem) -> Verdict` (`judges/items.py`; the item carries the request, response, criterion and `item_type`)
contract, so the panel calls Claude, a local in-process llama.cpp model, or the
heuristic baseline through identical code. Local models and judges that
reference the same GGUF reuse a **single in-process engine**, and the engine
cache loads models **sequentially** (one resident at a time by default) so a
test model and the judges never all hold weights at once.

## Layout

```
harness/
  llamacpp_engine.py       # in-process llama.cpp GGUF engines (lazy, sequential, multi-turn, system_prompt)
  hydrate.py               # download open weights from HuggingFace at startup
  model_loader.py          # load models UNDER TEST (llama.cpp / Ollama / cloud)
  test_suite.py            # test-suite schema + JSON loader (cases under `evals`)
  test_runner.py           # run a suite against models, persist outputs (+ --n-samples)
  evaluator.py             # CLI: hydrate + generate outputs + run judge panel + meta-eval
  compare.py               # CLI: behavioral-invariance diff (variant vs. base_model_id)
  provenance.py            # git commit + suite hash + timestamp stamped into results/*.json
judges/
  base.py                  # Judge ABC + Verdict types; retry/backoff, NonRetryableJudgeError
  prompts.py                # shared judge prompt + JSON response parser (strips expect:PASS|FAIL)
  claude_judge.py           # Anthropic API judge
  gpt4_judge.py              # OpenAI API judge
  local_llamacpp_judge.py   # local in-process llama.cpp (GGUF) judge
  heuristic_judge.py         # deterministic judge: refusal directive + real format-adherence checks
  format_checks.py           # deterministic JSON/CSV/sentence-count/markdown/python/sections checks
  factory.py                 # build judges from config/judges.yaml
  judge_panel.py             # orchestrate the panel + naive per-item majority-vote consensus
  meta_eval.py                # inter-judge Cohen's kappa, bias-by-slice, repeat-sample variance
config/
  models.yaml               # local + remote model catalogue; base_model_id / quantization lineage
  judges.yaml                # judge panel definition (priority/enabled); independent of models.yaml
  hardware_profile.yaml     # llama.cpp engine defaults + resident-model cap
data/
  test_suite_v1.json        # the 61-case suite (cases under `evals`)
tests/                      # unit tests for the pure-Python components
docs/critical-assessment.md  # the audit this round of fixes addresses
```

## Setup

```bash
# 0. Reproducible install (Python 3.14, see .python-version): the exact
#    versions the test suite runs against, including llama-cpp-python.
CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements.lock

# Or install the loose ranges instead:
# 1. Core install (pure-Python wheels; installs cleanly on every platform).
#    Includes huggingface_hub, used by the startup weight-hydration step.
pip install -r requirements.txt

# 2. Local model serving.
#
#    llama.cpp (loaded IN-PROCESS; required for any `engine: llamacpp` /
#    `access: llamacpp` model or judge — no server to start). It builds its own
#    bundled llama.cpp at install time; for a Metal build on Apple Silicon:
CMAKE_ARGS="-DGGML_METAL=on" pip install -r requirements-llamacpp.txt
#    Ollama is also supported for models under test (over HTTP, auto-managed):
brew install ollama

# 3. API keys for the remote judges (only needed if you run those judges).
export ANTHROPIC_API_KEY=...   # Claude judge
export OPENAI_API_KEY=...       # GPT-4o judge

# 4. (optional) HuggingFace token for gated repos that hydration downloads.
#    Picked up automatically by huggingface_hub. None of the checkpoints
#    shipped in config/models.yaml / config/judges.yaml are gated (verified
#    live against the HuggingFace API) -- only needed if you add one that is.
export HF_TOKEN=...
```

You do **not** need to pre-`ollama pull` the models — when a test model uses the
Ollama backend, `harness/model_loader.py` starts the `ollama serve` daemon (if
it isn't already up) and pulls the model tag on demand. Likewise you do **not**
need to pre-download HF weights: the harness hydrates them at startup (below).

## Usage

The entrypoint is `harness/evaluator.py`. Model ids come from
`config/models.yaml` (e.g. `gemma-4-12b`, `qwen3.5-9b`); judge subsets come
from `config/judges.yaml`.

### Full run: generate outputs, then judge

```bash
# In-process llama.cpp test model — nothing to start by hand; weights are
# hydrated, then loaded in-process (sequentially) on first use:
python harness/evaluator.py --models gemma-4-12b --judges all

# Ollama-backed test model (for a model whose serving.engine is `ollama`):
python harness/evaluator.py --models gemma-4-12b --judges all --prefer-engine ollama
```

`--judges` selects a preset: `all` (every enabled judge), `cheap` (priority ≤ 1
— the remote API judges plus the free deterministic heuristic; **no local GGUF
loads**), or `pilot`. `--judge-ids a,b` picks judges by id instead.

`--n-samples N` (default **3**) draws `N` independent generations per (model,
test); sample `i` uses seed `--seed + i`, recorded on every row. A single
non-deterministic draw
otherwise turns "did the model pass" into a noisy one-shot estimate with no
uncertainty measure. Once judged, `judges.meta_eval.sample_variance` (folded
into `results/judge_meta_eval.json` automatically when `--n-samples` > 1)
reports the per-test pass rate and standard error across the samples.

On startup the harness **hydrates**: it downloads the open-weight checkpoints
this run needs from HuggingFace (the requested `--models` plus the local judges
actually selected — see [Weight hydration](#weight-hydration)). Pass
`--no-hydrate` to skip it entirely.

### Judge previously-generated outputs

```bash
python harness/evaluator.py --outputs results/model_outputs.json --judges cheap
```

An outputs file from before the current schema (no `provenance`,
`sample_index` or `turns`, such as the 2026-08 `results/model_outputs.json`) is
refused with exit code 2: its system prompts were dropped and its multi-turn
cases collapsed. Regenerate it, or pass `--allow-legacy` to judge it anyway.
A generation run is checkpointed as it goes; continue a crashed run with
`--resume`, and replace a finished one only with `--overwrite`.

### Generate outputs only (no judges)

To run the eval questions through a model **without** invoking the judge panel,
pass `--no-judge`:

```bash
python harness/evaluator.py --models gemma-4-12b --no-judge
```

This runs the suite against the model and writes only
`results/model_outputs.json` — a `{"outputs": [...], "provenance": {...}}`
object, one row per test/model/sample (prompt, category, domain, technique,
severity, criteria, raw model output — or the full turn-by-turn `transcript`
for a multi-turn case, `sample_index`, and `latency_s` generation time). No
judge is built, so **no API keys are required**. `--no-judge` requires
`--models` (there is nothing to generate otherwise).

Results of a full run are written to `results/model_verdicts.json` — a
`{"verdicts": [...], "provenance": {...}}` object, one row per test/model with
each judge's verdict plus a naive per-item majority-vote consensus — and to
`results/judge_meta_eval.json`, the actual cross-item meta-evaluation (see
[Judge meta-evaluation](#judge-meta-evaluation)). Every `results/*.json` file
also carries a `provenance` block: harness git commit, suite file path +
content hash, generation timestamp, and the judge panel/models involved — so a
file found later is self-describing (see `harness/provenance.py`).

## Judge/model independence

No judge in `config/judges.yaml` may share a checkpoint (`id` or `model`/
`checkpoint`) with any entry in `config/models.yaml`. A judge that shares
weights with a model it's scoring is grading itself, not evaluating it, and
silently invalidates the judge-bias analysis this project exists to produce.
The two remote judges (Claude, GPT-4o) are the trusted, structurally-
independent anchor; the two local judges (Mistral-7B-Instruct, Llama-3.2-3B-
Instruct, both via bartowski's GGUF mirrors) were deliberately chosen to be
checkpoints that never appear in `config/models.yaml`.

## Judge meta-evaluation

`judges/meta_eval.py` is the actual meta-evaluation — `judges.judge_panel.
aggregate_verdicts` is only ever a naive per-item majority vote. Run
automatically at the end of `run_evaluation` (writes
`results/judge_meta_eval.json`; pass `--meta-eval ""` to skip, or
`--reference-judge <id>` to add each judge's divergence from one judge you
trust — there is no default; the keyword heuristic's refusal verdicts are
advisory only and never vote). It reports, per batch of judged rows:

- **Pairwise inter-judge agreement + Cohen's kappa** — raw agreement % is
  inflated when judges are individually skewed toward one label; kappa
  corrects for the agreement expected by chance alone.
- **Per-category / per-technique / per-severity verdict distribution per
  judge** — surfaces a judge that's systematically softer or harsher on one
  slice of the suite, which a single overall pass rate would average away.
- **Divergence from a reference judge** — how much each other judge disagrees
  with the reference, specifically.
- **Repeat-sample variance** (only present when `--n-samples` > 1 was used) —
  per-test pass rate + standard error across the repeated draws.

A verdict whose judge call itself failed (`error` set — see `Judge.evaluate`'s
retry handling below) is excluded from every one of these statistics: counting
a transient API failure as a genuine "AMBIGUOUS" would quietly bias agreement/
bias numbers toward whichever judge happened to have a flaky day.

## Behavioral-invariance comparison

Nothing computes the diff the harness is named for automatically —
`harness/evaluator.py` only ever produces independent per-model verdicts.
`harness/compare.py` closes that gap: given one or more already-judged
`results/model_verdicts.json`-shaped files and the `base_model_id` linkage in
`config/models.yaml`, it reports overall and per-category/technique/severity
pass-rate deltas between a fine-tuned/quantized variant and the base it's
linked to, plus which individual test ids flipped verdict:

```bash
python harness/compare.py --verdicts results/model_verdicts.json
python harness/compare.py --verdicts results/model_verdicts.json \
    --pairs gemma-4-12B-agentic-fable5:gemma-4-12b
```

Only pairs where `base_model_id` is set (and both sides have actually been
judged) are compared; `config/models.yaml`'s header comment explains when to
leave it `null` instead of guessing. If either side used `--n-samples` > 1,
repeated samples of the same test are collapsed to one verdict via majority
vote before comparing (see `harness.compare.index_by_model`) rather than
silently keeping whichever sample happened to load last.

## Weight hydration

At startup the harness downloads the **open weights this run needs** from
HuggingFace so the first in-process llama.cpp load doesn't block mid-run on a
multi-gigabyte download. Hydration is scoped to the run:

- the models named in `--models` (nothing when you only re-judge), and
- the local (`access: llamacpp`) judges that will actually be built — i.e. after
  the `--judges` preset and `enabled` are applied, and skipped entirely under
  `--no-judge`.

So `--outputs ... --judges cheap` (no `--models`) downloads **nothing** for the
models under test (`cheap` also excludes the local judges, so nothing for
those either); `--models gemma-4-12b --judges cheap` pulls only the Gemma
checkpoint. Cloud/API models (Anthropic, OpenAI, Replicate) are always skipped.

Downloads are idempotent (llama.cpp reuses the HF cache), so after the first run
hydration only verifies the cache. Skip it with `--no-hydrate`, or run it
standalone (no scoping filters → downloads everything in the configs):

```bash
python harness/hydrate.py                      # download all open weights now
python harness/hydrate.py --models gemma-4-12b  # just this model's weights
python harness/hydrate.py --dry-run             # list what would be downloaded
```

Hydration downloads through **llama-cpp-python** — the same
`Llama.from_pretrained` path used to serve — so it fetches exactly the file the
engine will load, not the whole repo, with no direct `huggingface_hub` call of
our own. Gated repos need `HF_TOKEN` set (none of the checkpoints shipped in
`config/models.yaml` / `config/judges.yaml` currently are — verified live
against the HuggingFace API). Prefer the **exact** GGUF file name in
`gguf_file` (as in HuggingFace's "Use this model" snippet), e.g.
`gguf_file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"`: it's unambiguous and
**fails loudly if the name is wrong** (`hydrate_weights` raises, aborting the
run before any generation happens — worth confirming a new entry resolves via
`python harness/hydrate.py --dry-run` and the HuggingFace API before relying
on it). A glob like `"*Q4_K_M.gguf"` also works. `additional_files` fetches
extra parts (split-GGUF shards).

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
  - id: "mistral-7b-instruct-4bit"
    checkpoint: "bartowski/Mistral-7B-Instruct-v0.3-GGUF"   # HF repo -- verify it
    base_model_id: null          # or another entry's id, if this is a fine-tune of it
    serving:
      engine: "llamacpp"          # llamacpp (in-process GGUF) | ollama
      gguf_file: "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"   # the exact quant to load
      mode: "chat"                 # chat (default; instruct models) | completion (base)
      ollama_tag: "mistral:7b"    # only used if engine: ollama
```

**Chat vs completion.** `serving.mode: chat` (the default) sends each test
prompt (plus a real `system` role message when the test case sets
`system_prompt`) through the GGUF's embedded chat template
(`create_chat_completion`) — required for instruction-tuned checkpoints (`-it`
/ `-instruct`), which *continue* a raw prompt instead of answering it. Set
`mode: "completion"` for base (non-instruct) checkpoints. Local judges always
use the chat path, with decoding additionally grammar-constrained to valid
JSON. A multi-turn test case (`turns: [...]` in the suite instead of `prompt`)
goes through `chat_turns`/`generate_turns` — each reply is fed back as an
`assistant` turn before the next `turns` entry is sent, so the model actually
sees (and can be held to) its own prior answer.

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

The tests cover the deterministic, network-free parts: prompt formatting +
JSON parsing (including that the `expect:PASS`/`expect:FAIL` directive never
reaches the rendered judge prompt), verdict normalisation, base-class error
handling and retry/backoff, the heuristic judge's real format-adherence
checks, the suite loader (including multi-turn `turns` and `system_prompt`),
the judge factory, panel aggregation, `harness/compare.py`'s behavioral-
invariance diff, and `judges/meta_eval.py`'s inter-judge agreement/bias
statistics.

## What was fixed

See [docs/critical-assessment.md](docs/critical-assessment.md) for the full
audit. In summary, fixed: the judge/model self-judging confound (judges no
longer share any checkpoint with a model under test); the `expect:PASS`/
`expect:FAIL` directive leaking into the LLM judge prompt; `system_prompt`
being silently dropped; multi-turn cases collapsing into one static prompt
instead of a real conversation; the heuristic judge's directive regex never
matching the suite's actual `expect:PASS`/`expect:FAIL` vocabulary; the
absence of any real meta-evaluation (`judges/meta_eval.py` now computes
inter-judge Cohen's kappa and per-slice bias); the absence of any fine-tune/
quantization lineage (`base_model_id` + `harness/compare.py`); `--judges
cheap` being the most expensive preset instead of the cheapest; and the lack
of retry/backoff, error-vs-genuine-ambiguous distinction, repeat-sampling
variance, and result provenance.

## What is intentionally NOT done here — **[Sam]**

- **On-hardware verification.** Confirm the in-process llama.cpp engine settings
  in `config/hardware_profile.yaml` (`n_ctx`, `n_gpu_layers`, `n_batch`), the
  `llamacpp.max_resident` cap, the 32GB memory budget, and load time on the
  actual M5 (PRD "Sam Must Review" checklist).
- **Cloud fallback wiring.** `harness.model_loader.ReplicateModel.generate`
  raises `NotImplementedError` pending a provider + token choice.
- **Ground-truth calibration.** There is no human-labeled gold set anywhere in
  the repo, so no judge's *accuracy* (as opposed to inter-judge agreement) can
  be computed — `judges/meta_eval.py` reports agreement/bias, not accuracy.
- **Unconfirmed `base_model_id` lineage.** A few `fine_tuned_models` entries in
  `config/models.yaml` are `base_model_id: null` with a comment explaining why
  (a merge with no single base, an ambiguous size-class name) — verify against
  the model card before filling these in rather than guessing.
