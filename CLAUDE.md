# CLAUDE.md

Guidance for working in this repo. Read this before making changes.

## What this is

`meta-eval` is a **behavioral-invariance safety-eval harness + meta-evaluation
infrastructure** (PRD v3.1). It measures how fine-tuning and quantization affect
model safety, and evaluates the evaluators — checking whether common
LLM-as-a-judge models are biased on safety-critical tasks.

The repo is **scaffolding for review**, not a finished system. Several pieces are
deliberately left as `[Sam]` decisions (the real 15-test suite, the substantive
meta-evaluation methodology, on-hardware verification). Don't silently "finish"
those; see [What's intentionally unfinished](#whats-intentionally-unfinished).

## Pipeline at a glance

Two separate passes, two files, so expensive generation is done once and can be
re-judged:

```
hydrate (download the open weights this run needs from HF)
  → generate: models under test → results/model_outputs.jsonl
  → judge:    judge panel over outputs → results/model_verdicts.jsonl
```

Hydration is scoped to the run: only the requested `--models` and the local
judges actually selected are downloaded (`harness/hydrate.py:collect_weights`
takes the filters; the evaluator passes them).

`harness/evaluator.py` is the CLI entrypoint that ties it together.

## Layout

```
harness/
  evaluator.py        # CLI: hydrate → generate → judge
  hydrate.py          # download open weights from HuggingFace at startup
  llamacpp_engine.py  # in-process llama.cpp GGUF engines (sequential loading)
  vllm_engine.py      # in-process vLLM engines (lazy load, process-wide cache)
  model_loader.py     # load models UNDER TEST (llamacpp / ollama / vllm / cloud)
  test_runner.py      # run a suite against models, persist raw outputs
  test_suite.py       # .jsonl suite schema + loader
judges/
  base.py             # Judge ABC + Verdict types (the shared contract)
  prompts.py          # shared judge prompt template + JSON response parser
  claude_judge.py     # Anthropic API judge
  gpt4_judge.py       # OpenAI API judge
  local_vllm_judge.py # base for local in-process vLLM judges
  mistral_local_judge.py / llama_local_judge.py  # id + default-checkpoint subclasses
  heuristic_judge.py  # deterministic, zero-network baseline
  factory.py          # build judges from config/judges.yaml
  judge_panel.py      # fan out to the panel + naive consensus (placeholder)
config/
  models.yaml           # models under test (local + cloud)
  judges.yaml           # judge panel (provider / access / priority / enabled)
  hardware_profile.yaml # in-process vLLM engine kwargs + budgets
data/test_suite_v1.jsonl # EXAMPLE 3-row schema only (real suite is [Sam])
tests/                   # unit tests for the deterministic, network-free parts
```

## Key architectural facts

- **llama.cpp is the recommended local backend (GGUF-native, sequential).**
  `harness/llamacpp_engine.py` loads a *specific* GGUF quant from an HF repo —
  `Llama.from_pretrained(repo_id, filename="*Q4_K_M.gguf")` — selected by
  `serving.gguf_file` (models) / `gguf_file` (judges), which also scopes what
  hydration downloads. Its cache holds at most `max_resident` models at once
  (`config/hardware_profile.yaml → llamacpp.max_resident`, default 1), so loading
  a new model frees the previous one (`Llama.close()`) — unlike the vLLM cache,
  which keeps every checkpoint resident. Construct engines via
  `llamacpp_engine.get_engine(...)`; wire via `engine: llamacpp` / `access:
  llamacpp`. Needs `pip install llama-cpp-python`.
- **Local models and judges can also run in-process via vLLM.** There is no separate
  `vllm serve` process and no HTTP hop. `harness/vllm_engine.py` loads a
  checkpoint with `vllm.LLM(...)` the first time `generate` is called and caches
  the engine **process-wide, keyed by checkpoint** — so a test model and a judge
  that share a checkpoint reuse one loaded copy of the weights. Construct engines
  via `get_engine(model, engine_kwargs)`, never `InProcessVLLM(...)` directly.
- **Loading is lazy and import-light.** Constructing a `VLLMModel`, a judge, or an
  `InProcessVLLM` must NOT import vLLM or load weights — that only happens on the
  first `generate`. Keep it that way so the config/factory paths and the whole
  test suite run without vLLM installed.
- **Every judge implements the same contract:**
  `Judge.evaluate(test_id, model_output, criteria) -> Verdict`. The base class
  wraps `_evaluate`, so a judge that raises becomes an `AMBIGUOUS` verdict with
  `error` set — a single judge failure never aborts a run. Add new judges by
  subclassing `Judge`, implementing `_evaluate`, and wiring them into
  `judges/factory.py`.
- **Heavy deps are imported lazily** inside the method that needs them (`anthropic`,
  `openai`, `vllm`, `huggingface_hub`), each with a clear install message on
  `ImportError`. Preserve this pattern — the core `pip install -r requirements.txt`
  must stay pure-Python and cross-platform.
- **Config drives everything.** Models, judges, and hardware knobs live in
  `config/*.yaml`; loaders accept either a parsed dict or a path. `vllm.LLM`
  kwargs come from `hardware_profile.yaml → vllm_defaults` via
  `engine_kwargs_from_profile` (which filters to valid constructor keys).

## Common commands

```bash
pip install -r requirements.txt          # core deps (pure-Python; incl. huggingface_hub)
pip install llama-cpp-python             # recommended local GGUF backend (engine/access: llamacpp)
pip install "vllm>=0.3.0"                # alternative local backend (engine/access: vllm)

pytest                                    # run the (network-free) unit tests

# Full run: hydrate → generate → judge
python harness/evaluator.py --models mistral-7b-4bit --judges all
python harness/evaluator.py --outputs results/model_outputs.jsonl --judges cheap  # re-judge
python harness/evaluator.py --models mistral-7b-4bit --no-judge                    # generate only
python harness/evaluator.py --models mistral-7b-4bit --judges cheap --no-hydrate   # skip HF download

python harness/hydrate.py --dry-run       # list open weights that would be downloaded
```

`--judges` presets: `all` (every enabled judge), `cheap` (priority ≤ 1, remote
API judges only), `pilot`.

## Environment

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — only for the respective remote judges.
- `HF_TOKEN` — for gated HF repos that hydration downloads (e.g. Llama-2).
- Target hardware is a MacBook Pro M5 (32GB unified memory, Apple Metal). The
  scaffold's memory/latency assumptions are unverified on real hardware.

## Testing conventions

- Tests live in `tests/` and cover only the **deterministic, network-free** parts.
  `pytest.ini` sets `pythonpath = .` so `import harness` / `import judges` work.
- Never hit the network or load real weights in a test. To exercise vLLM /
  HuggingFace wiring, inject a fake module into `sys.modules` (see
  `tests/test_vllm_engine_and_hydrate.py`) and call `vllm_engine.reset_engines()`
  to clear the process-wide cache between tests.
- Any change to `harness/`, `judges/`, or the config schema should keep `pytest`
  green and, where it adds logic, come with a network-free test.

## What's intentionally unfinished (`[Sam]`)

Don't quietly implement these; they're deliberate decisions:

- **The real 15-test suite** — `data/test_suite_v1.jsonl` is a 3-row schema example.
- **The meta-evaluation methodology** — `judges.judge_panel.aggregate_verdicts` is
  a naive majority vote placeholder; inter-judge agreement, bias detection, and
  reliability weighting are unbuilt.
- **On-hardware verification** — vLLM engine kwargs, the 32GB budget, and GGUF
  handling in `config/hardware_profile.yaml` are unverified on the M5.
- **Cloud fallback** — `harness.model_loader.ReplicateModel.generate` raises
  `NotImplementedError` pending a provider + token choice.

## Git

- Work on the branch specified by the task; commit with clear messages; push with
  `git push -u origin <branch>`. Do not open a PR unless explicitly asked.
- `results/` holds run artifacts (only `.gitkeep` is tracked); don't commit
  generated `*.jsonl` outputs.
