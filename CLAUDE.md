# CLAUDE.md

Guidance for working in this repo. Read this before making changes.

## What this is

`meta-eval` is a **behavioral-invariance safety-eval harness + meta-evaluation
infrastructure** (PRD v3.1). It measures how fine-tuning and quantization affect
model safety, and evaluates the evaluators — checking whether common
LLM-as-a-judge models are biased on safety-critical tasks.

A [critical assessment](docs/critical-assessment.md) of the original scaffold
identified several issues that would have silently invalidated both halves of
that goal (judges sharing weights with models under test, a criterion-leaking
judge prompt, a dropped `system_prompt`, no lineage between fine-tuned variants
and their base). Those have since been fixed; see
[What was fixed](#what-was-fixed-see-docscritical-assessmentmd) below. What
remains genuinely open is on-hardware verification and the cloud-fallback
provider choice — both still deliberate `[Sam]` decisions.

## Pipeline at a glance

Three passes, three files, so expensive generation is done once, judging can be
redone freely, and the resulting numbers can actually answer the two questions
this harness exists for:

```
hydrate   (download the open weights this run needs from HF)
  → generate:  models under test → results/model_outputs.json    (rows under `outputs`)
  → judge:     judge panel over outputs → results/model_verdicts.json (rows under `verdicts`)
             + results/judge_meta_eval.json (inter-judge agreement/bias — judges/meta_eval.py)
  → compare:   harness/compare.py results/model_verdicts.json  (behavioral-invariance
               diff between a fine-tuned/quantized variant and its base — separate,
               on-demand step, not run automatically by evaluator.py)
```

Hydration is scoped to the run: only the requested `--models` and the local
judges actually selected are downloaded (`harness/hydrate.py:collect_weights`
takes the filters; the evaluator passes them).

`harness/evaluator.py` is the CLI entrypoint that ties generate+judge together.
`harness/compare.py` is a separate CLI, run against already-written verdict files.

## Layout

```
harness/
  evaluator.py        # CLI: hydrate → generate → judge (+ writes judge_meta_eval.json)
  compare.py           # CLI: behavioral-invariance diff between a variant and its base_model_id
  hydrate.py           # download open weights from HuggingFace at startup
  llamacpp_engine.py   # in-process llama.cpp GGUF engines (sequential loading, multi-turn, system_prompt)
  model_loader.py      # load models UNDER TEST (llamacpp / ollama / cloud)
  test_runner.py       # run a suite against models, persist raw outputs (+ --n-samples, provenance)
  test_suite.py        # JSON suite schema + loader (cases under `evals`; turns/system_prompt/domain/...)
  provenance.py         # git commit + suite hash + timestamp stamped into results/*.json
  gold.py              # task 2.12: gold sample, local labelling page (gold_ui.html), labels-vs-judges compare
judges/
  base.py              # Judge ABC + Verdict types (the shared contract); retry/backoff, NonRetryableJudgeError
  items.py             # EvalItem (the unit a judge grades) + item types
  rubric.py            # rubric v2: prompts, JSON schemas, StrongREJECT scoring, verdict derivation
  prompts.py           # strip_expect_directive + JSON response parser
  claude_judge.py      # Anthropic API judge
  gpt4_judge.py        # OpenAI API judge
  local_llamacpp_judge.py # local in-process llama.cpp (GGUF) judge
  classifier_judges.py # HarmBench / WildGuard reference classifiers (own prompts, raw completion)
  declines.py          # decline memory: never re-send a judge content it declined
  heuristic_judge.py   # deterministic judge: refusal directive + real format-adherence checks
  format_checks.py     # deterministic JSON/CSV/sentence-count/markdown/python/sections checks
  factory.py           # build judges from config/judges.yaml
  judge_panel.py        # fan out to the panel + naive per-item majority-vote consensus
  meta_eval.py          # the REAL meta-evaluation: inter-judge kappa, bias-by-slice, sample variance
config/
  models.yaml            # models under test (local + cloud); base_model_id / quantization lineage
  judges.yaml             # judge panel (provider / access / priority / enabled); independent of models.yaml
  hardware_profile.yaml   # in-process llama.cpp engine kwargs + budgets
data/test_suite_v1.json  # the 61-case suite, cases under `evals`
tests/                   # unit tests for the deterministic, network-free parts
docs/critical-assessment.md  # the audit this round of fixes addresses
```

## Key architectural facts

- **llama.cpp is the local serving backend (GGUF-native, sequential).**
  `harness/llamacpp_engine.py` loads a *specific* GGUF quant from an HF repo —
  `Llama.from_pretrained(repo_id, filename="*Q4_K_M.gguf")` — selected by
  `serving.gguf_file` (models) / `gguf_file` (judges), which also scopes what
  hydration downloads. Its cache holds at most `max_resident` models at once
  (`config/hardware_profile.yaml → llamacpp.max_resident`, default 1), so loading
  a new model frees the previous one (`Llama.close()`); an evicted engine reloads
  on next use. Construct engines via `llamacpp_engine.get_engine(...)`, never
  `LlamaCppEngine(...)` directly; wire via `engine: llamacpp` / `access:
  llamacpp`. Needs `pip install llama-cpp-python`. (Ollama remains an alternative
  backend for models under test, over HTTP.)
- **Judge/model independence is load-bearing, not incidental.** No `id` or
  `checkpoint` in `config/judges.yaml` may match an entry in
  `config/models.yaml` — a judge that shares weights with a model it's scoring
  is self-grading, and silently invalidates any judge-bias analysis. The two
  local judges are deliberately different checkpoints (verified live against
  the HuggingFace API — real, ungated, instruct-tuned) from every model under
  test. Keep this invariant when editing either config;
  `tests/test_judge_independence.py` enforces it. Every model and judge also
  declares a `family` (architecture), and models whose name claims a teacher
  (e.g. "Claude-Mythos") carry `name_claims_teacher`, so the meta-evaluation
  can report same-family judge/model pairs separately.
- **`system_prompt` and multi-turn (`turns`) are real, not suite metadata that
  gets silently dropped.** `TestCase` (`harness/test_suite.py`) carries both;
  `TestRunner` sends `system_prompt` through `generate()`/`generate_conversation()`
  down to a real `system` role message (`LlamaCppEngine.chat`/`chat_turns`) or
  Ollama's `system`/`/api/chat`. A multi-turn case's judge input is the full
  transcript (`row["transcript"]`), not just the final reply — see
  `tests/test_multiturn_integration.py` for why that distinction matters.
- **LLM judges grade with rubric v2 (`judges/rubric.py`), not PASS/FAIL.**
  The judge reports dimensions for the item's rubric (`harm`: StrongREJECT's
  refused / convincingness / specificity; `harm_refusal_only` for tier 3;
  `benign`; `factual`; `criterion`), under a strict JSON schema. Code derives
  the verdict and `harm_score` (`rubric.derive`), stored in `Verdict.scores`.
  The prompt shows the system prompt, request and response, but never the
  test id or the `expect:` directive (`prompts.strip_expect_directive`), and
  tags inside graded text are neutralised. The Claude judge
  (`claude-sonnet-5-5`) sends no `temperature`, sets `effort`, and has refusal
  fallbacks off (D4): a decline is `abstained` with
  `abstain_reason="judge_declined:<category>"`. The OpenAI judge is pinned to
  `gpt-4o-2024-11-20`. Verdict provenance records `rubric_version` and
  `rubric_sha256`. Rubric 2.1.0 is **frozen** (task 2.12):
  `tests/test_rubric.py::test_rubric_is_frozen` pins its version and hash.
  Any rubric edit needs a version bump and a new gold check
  (`harness/gold.py`). `judges/heuristic_judge.py` reads the raw criteria for its
  own deterministic dispatch.
- **Loading is lazy and import-light.** Constructing a `LlamaCppModel`, a judge,
  or a `LlamaCppEngine` must NOT import `llama_cpp` or load weights — that only
  happens on the first `generate`. Keep it that way so the config/factory paths
  and the whole test suite run without llama-cpp-python installed.
- **Every judge implements the same contract:**
  `Judge.evaluate(item: EvalItem) -> Verdict` (`judges/items.py`; the item carries the request, response, criterion and `item_type`).
  The base class wraps `_evaluate`, retries up to `max_retries` times (remote
  judges default to 2; a `NonRetryableJudgeError` — missing API key,
  uninstalled SDK — skips retrying entirely) and, on final failure, produces an
  `AMBIGUOUS` verdict with `error` set — a judge failure never aborts a run.
  Add new judges by subclassing `Judge`, implementing `_evaluate`, and wiring
  them into `judges/factory.py`.
- **`error` vs `abstained` are both "not a real verdict," excluded the same
  way.** `error` means the judge *call* failed (network, auth, malformed
  JSON). `abstained` means the call succeeded but the judge is *structurally
  incapable* of assessing this criterion — `HeuristicJudge` sets it on the
  AMBIGUOUS it returns for factual/reasoning criteria a keyword scan can't
  verify, and by default on **all** its refusal verdicts (`refusal_votes:
  false`): those are recorded as advisory but never vote, because keyword
  refusal detection is too unreliable to count. Only its deterministic format
  checks vote. There is no default meta-eval reference judge
  (`--reference-judge` is optional). Conflating
  either with a genuine verdict would count it as real agreement/disagreement.
  `aggregate_verdicts` (`judges/judge_panel.py`), `judges.meta_eval._ok_verdicts`,
  and `harness.compare._resolve_verdict` all exclude verdicts with `error` OR
  `abstained` set — extend all three together if you add a fourth "not a real
  verdict" case, don't patch just one. When no real verdict is left, the
  result is `None` (missing data), never `"AMBIGUOUS"`: `aggregate_verdicts`
  returns `consensus: None`, a failed generation gets `consensus: None` plus
  `generation_error`, and `compare_pair` / `sample_variance` drop `None` and
  report how many they dropped. A genuine AMBIGUOUS (the judge read the
  response and could not decide) stays in the pass-rate denominator as a
  non-pass, and `compare_pair` reports its count.
- **Heavy deps are imported lazily** inside the method that needs them (`anthropic`,
  `openai`, `llama_cpp`, `huggingface_hub`), each with a clear install message on
  `ImportError`. Preserve this pattern — the core `pip install -r requirements.txt`
  must stay pure-Python and cross-platform.
- **Config drives everything.** Models, judges, and hardware knobs live in
  `config/*.yaml`; loaders accept either a parsed dict or a path. `llama_cpp.Llama`
  kwargs come from `hardware_profile.yaml → llamacpp_defaults` via
  `llamacpp_engine.engine_kwargs_from_profile` (which filters to valid constructor
  keys); the resident-model cap comes from `llamacpp.max_resident`.
  Each model and local judge is served at `min(context_tokens,
  llamacpp.max_n_ctx)` (`llamacpp_engine.n_ctx_for`); `llamacpp_defaults.n_ctx`
  is only the fallback for entries without `context_tokens`.
- **Reasoning traces are split from the answer** (`harness/reasoning.py`,
  decision D2). Models with `serving.reasoning: true` get their chat template
  to open `<think>`, so replies arrive as `<trace></think><answer>`, and a reply
  with no closing tag is an unfinished trace. Output rows carry `output` (the
  final answer, which judges score), plus `reasoning`, `raw_replies` (one per
  turn, like `turns`) and `reasoning_complete`. A row whose trace was cut off
  with no answer is not judged; it gets `generation_error` like a failed
  generation. Give these models a larger `serving.max_tokens`.
- **Truncation is recorded, not guessed.** llama.cpp (`finish_reason`) and
  Ollama (`done_reason`) report why each turn ended. Models expose this as
  `last_finish_reasons`, and `TestRunner` writes `finish_reasons` and
  `truncated` (any turn ended with `"length"`) on every output row.
- **Each judge has a content policy (`skip:` in judges.yaml).** A judge is
  never sent harmful items of the listed `severity_tiers` or `domains`; a
  harmful item with no tier counts as tier 3. The skipped item gets an
  abstained verdict with `abstain_reason="skipped_by_policy:<why>"` and no
  call is made (`Judge.policy_skip_reason`, `JudgePanel._skip_verdict`). This
  protects the cloud API accounts: Claude skips tier 3, `cybersecurity` and
  `biology` (it declined all of them in the 2.12 gold check); the OpenAI
  judges skip tier 3. Local judges skip nothing, so the reference classifiers
  (`judges/classifier_judges.py`: HarmBench, WildGuard) grade those items.
  `skipped_by_policy` is not `judge_declined`: only a real decline counts in
  the decline rate. Do not send skipped content to a cloud judge in an ad-hoc
  test either.
- **Decline memory (`judges/declines.py`).** Suite labels miss some harmful
  content, so the policy above is not enough on its own. When a judge
  declines an item, the evaluator records `(judge_id, sha256 of the request,
  system prompt and response)` in `results/judge_declines.jsonl` (`--declines`;
  hashes only, no content, local to this machine). Later runs skip that judge
  on the same content (`skipped_by_policy:previously_declined`). Backfill
  from old verdict files with `python judges/declines.py --outputs <outputs>
  <verdicts...>`. `run_evaluation` has no default path (tests stay out of
  `results/`); the CLI does.
- **Exact checks decide exactly-checkable items.** When the panel has a judge
  with `runs_exact_checks` (the heuristic), every other judge skips items
  where `heuristic_judge.exact_check_applies` is true (format items with a
  registered check and no "must refuse" directive): they record an abstained
  "Skipped" verdict without making a call. With no such judge in the panel,
  nothing is skipped.
- **Judge panel evaluates judge-outer, not row-outer.** `JudgePanel.evaluate_batch`
  runs one judge over the *whole* batch of outputs before moving to the next
  judge, so each local llama.cpp judge loads once per run instead of once per
  row. Preserve this order in any change to `judges/judge_panel.py`.
- **Chat vs completion.** `serving.mode: chat` (the default) sends prompts
  through the GGUF's embedded chat template (`create_chat_completion`) — needed
  for instruction-tuned checkpoints. Use `mode: "completion"` only for base
  (non-instruct) checkpoints that should continue a raw prompt. Local judges
  always use the chat path with grammar-constrained JSON decoding.
- **Ollama models are self-managing.** A model with `serving.engine: ollama`
  causes `harness/model_loader.py` to start the `ollama serve` daemon (if not
  already running) and `pull` the `ollama_tag` on first use — no manual
  `ollama pull` or server start needed. `--prefer-engine` only picks the
  fallback for entries that don't pin `serving.engine`.
- **`base_model_id` in `config/models.yaml` links a fine-tuned/quantized
  variant to the entry it was derived from** — `null` where the lineage isn't
  confidently known (don't guess; a wrong lineage claim is worse than an
  honestly-missing one). `harness/compare.py` reads it to compute the actual
  behavioral-invariance diff; nothing else in the codebase does this
  automatically.
- **Repeated sampling (`--n-samples`) tags rows with `sample_index`.**
  `harness.compare.index_by_model` first keeps only the latest row per
  (model, test_id, sample_index) (`judges.meta_eval.latest_per_sample`, so a
  re-judge file never counts as extra samples), then collapses distinct
  `sample_index` values via majority vote. Every tie for first place — among
  samples (`majority_label`) or among judges (`aggregate_verdicts`) — is
  `"AMBIGUOUS"`, never broken by order or by confidence. The collapse is done
  **per `verdict_source`**: `index_by_model(rows, verdict_source=...)` must be
  called with the same `verdict_source` you'll later pass to `compare_pair`/
  `compare_all` (the CLI does this correctly; a caller that indexes on
  `"consensus"` and then compares on a named judge id silently reads that
  judge's un-aggregated sample-0 verdict instead of the majority across
  samples).

## Common commands

```bash
pip install -r requirements.lock         # exact tested versions (Python 3.14; incl. llama-cpp-python)
pip install -r requirements.txt          # or: core deps as ranges (pure-Python; incl. huggingface_hub)
pip install -r requirements-llamacpp.txt # local GGUF backend, pinned (engine/access: llamacpp)
# Regenerate the lock after changing requirements*.txt, from the tested venv:
#   uv pip compile requirements.txt requirements-llamacpp.txt --python-version 3.14 \
#     -c <(.venv/bin/pip freeze) -o requirements.lock

pytest                                    # run the (network-free) unit tests
pytest tests/test_prompts.py              # run a single test file
pytest tests/test_prompts.py -k parse     # run a single test by name

# Full run: hydrate → generate → judge (+ judge_meta_eval.json)
python harness/evaluator.py --models gemma-4-12b --judges all
python harness/evaluator.py --outputs results/model_outputs.json --judges cheap   # re-judge
python harness/evaluator.py --outputs results/model_outputs.json --self-consistency 0.1   # re-grade a fixed 10% to measure judge self-agreement
python harness/evaluator.py --outputs results/model_outputs.json --batch        # Claude grades via the Message Batches API (~half price, up to 24 h)
python harness/evaluator.py --outputs old.json --allow-legacy   # judge a pre-fix outputs file anyway (refused by default)
python harness/evaluator.py --models gemma-4-12b --no-judge                    # generate only
python harness/evaluator.py --models gemma-4-12b --judges cheap --no-hydrate   # skip HF download
python harness/evaluator.py --models gemma-4-12b --n-samples 1 --seed 7        # one sample (default is 3); sample i uses seed 7+i
python harness/evaluator.py --models gemma-4-12b --resume      # continue a crashed run from its checkpoint
python harness/evaluator.py --models gemma-4-12b --overwrite   # replace an existing outputs file

# Gold labels (2.12). Local only: results/gold/ holds raw completions and is gitignored.
python harness/gold.py sample            # stratified 20-row sample -> results/gold/sample.json
python harness/gold.py serve             # label at http://127.0.0.1:8765 (model hidden)
python harness/evaluator.py --outputs results/gold/sample.json --verdicts results/gold/verdicts.json --meta-eval results/gold/meta_eval.json --judges cheap
python harness/gold.py compare           # human labels vs judge verdicts -> results/gold/compare.json

python harness/hydrate.py --dry-run       # list open weights that would be downloaded

# Behavioral-invariance diff (needs both variant and base already judged)
python harness/compare.py --verdicts results/model_verdicts.json
```

`--judges` presets: `all` (every enabled judge), `cheap` (priority ≤ 1 — the
remote API judges and the free deterministic heuristic; **no local GGUF
loads**), `pilot`.
`--judge-ids a,b` picks judges by id instead; it overrides the preset and the
`enabled` flag. Selection lives in one place, `judges.factory.select_judge_entries`,
which both the panel and hydration use.

## Environment

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — only for the respective remote judges.
  The CLIs load them from the gitignored `.env` (`harness/env.py`, python-dotenv;
  variables already set in the shell win). The evaluator exits with code 2
  before any generation or judging if an enabled judge has no key.
- `HF_TOKEN` — for gated HF repos. Hydration downloads no gated repo. The one
  gated checkpoint is WildGuard (`allenai/wildguard`): accept the AI2 terms on
  HF, then run `scripts/convert_wildguard.sh`, which builds the local GGUF at
  the judge's `model_path`. Without that file, a run that selects
  `wildguard-7b` stops before any work.
- Target hardware is a MacBook Pro M5 (32GB unified memory, Apple Metal). The
  scaffold's memory/latency assumptions are unverified on real hardware.

## Testing conventions

- Tests live in `tests/` and cover only the **deterministic, network-free** parts.
  `pytest.ini` sets `pythonpath = .` so `import harness` / `import judges` work.
- Never hit the network or load real weights in a test. To exercise llama.cpp /
  HuggingFace wiring, inject a fake module into `sys.modules` (see
  `tests/test_llamacpp_engine.py`) and call `llamacpp_engine.reset_engines()`
  to clear the process-wide cache between tests. (`git_commit()` in
  `tests/test_provenance.py` is the one exception — it's a local, network-free
  subprocess call against this actual checkout, not a network call.)
- Any change to `harness/`, `judges/`, or the config schema should keep `pytest`
  green and, where it adds logic, come with a network-free test.
- A test that writes JSON via `run_evaluation`/`TestRunner.run_tests` must pass
  explicit `tmp_path`-based paths for **every** output arg (including
  `meta_eval_path`) — a default left unset writes into the real `results/` dir.

## What was fixed (see docs/critical-assessment.md)

The audit found the original scaffold's numbers would not have supported
either half of the project's goal. Fixed:

- **Judge/model self-judging confound** — judges no longer share checkpoints
  with any model under test; remote judges (previously commented out) are the
  trusted anchor.
- **`expect:PASS`/`expect:FAIL` leaking into the LLM judge prompt** — stripped
  before rendering; was sitting directly next to the instruction to output a
  verdict.
- **`system_prompt` silently dropped** — now wired end-to-end through generation.
- **Multi-turn cases collapsed into one static prompt** — now real
  conversations; the judge sees the full exchange.
- **Heuristic judge's directive regex never matched the real suite** (`expect:
  refuse`/`expect:comply` vs. the suite's actual `expect:PASS`/`expect:FAIL`) —
  fixed, plus it now runs real deterministic checks for format-adherence.
- **No meta-evaluation** — `judges/meta_eval.py` computes real inter-judge
  Cohen's kappa and per-category/technique/severity bias, not just a per-item
  majority vote.
- **No fine-tune/quantization lineage** — `base_model_id` in
  `config/models.yaml` + `harness/compare.py` compute the actual
  behavioral-invariance diff.
- **`--judges cheap` was the most expensive preset** — local judges lacked an
  explicit `priority`, defaulting to the same tier as the (then-disabled)
  remote judges.
- **No retry/backoff, no error-vs-genuine-ambiguous distinction, no repeat-
  sampling variance estimate, no result provenance** — all added; see
  `judges/base.py`, `judges/meta_eval.py::sample_variance`, `harness/provenance.py`.

## What's still intentionally unfinished (`[Sam]`)

Don't quietly implement these; they're deliberate decisions:

- **On-hardware verification** — the llama.cpp engine kwargs (`n_ctx`,
  `n_gpu_layers`), the `max_resident` cap, and the 32GB budget in
  `config/hardware_profile.yaml` are unverified on the M5.
- **Cloud fallback** — `harness.model_loader.ReplicateModel.generate` raises
  `NotImplementedError` pending a provider + token choice.
- **Ground-truth calibration** — there is no human-labeled gold set anywhere
  in the repo, so no judge's *accuracy* (as opposed to inter-judge agreement)
  can be computed. `judges/meta_eval.py` reports agreement/bias, not accuracy.
- **Unconfirmed `base_model_id` lineage** — several `fine_tuned_models`
  entries in `config/models.yaml` have `base_model_id: null` with a comment
  explaining why (merge with no single base, ambiguous size-class naming,
  etc.) — verify against the model card before filling these in, don't guess.

## Git

- Work on the branch specified by the task; commit with clear messages; push with
  `git push -u origin <branch>`. Do not open a PR unless explicitly asked.
- `results/` holds run artifacts (only `.gitkeep` is tracked); don't commit
  generated `model_outputs.json` / `model_verdicts.json` / `judge_meta_eval.json`.
