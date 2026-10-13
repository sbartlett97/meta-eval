# Critical Assessment — `meta-eval` (PRD v3.1 scaffold)

> **Status: the findings below have been addressed.** This document is kept
> as the historical audit; it describes the codebase *as it was* when
> written, not as it is now. Every numbered finding in §2 and every item in
> §3's improvement list has been fixed (or, for the handful that couldn't be
> — on-hardware verification, ground-truth calibration, the cloud-fallback
> provider choice, a couple of unconfirmed `base_model_id` lineages — is
> explicitly called out as still open in `CLAUDE.md`/`README.md`, not
> silently dropped). See `CLAUDE.md`'s "What was fixed" section and
> `README.md`'s for the mapping from finding to fix, and `harness/compare.py`
> / `judges/meta_eval.py` / `judges/format_checks.py` / `harness/provenance.py`
> for the new modules that resulted. One correction the fix pass surfaced:
> the suite is 61 cases, not 60 as originally counted below.

Scope: this assesses the harness as it exists in the working tree **at the
time of writing** — `harness/`, `judges/`, `config/*.yaml`, and the
now-populated `data/test_suite_v1.json` (60 cases, up from the 3-case schema
example the docs still describe). Every claim below is grounded in a specific
file/line; where I make a judgment call, I say so explicitly.

## 1. How well does it address the problem it's trying to evaluate?

The stated problem has two parts: (a) measure whether fine-tuning/quantization
changes model *safety behavior*, and (b) meta-evaluate whether LLM judges are
*biased* on that judgment. As **plumbing**, the repo does a competent job: a
clean `Judge` ABC (`judges/base.py`), a config-driven judge panel
(`judges/factory.py`), a sensible generate/judge split so expensive inference
isn't repeated (`harness/test_runner.py` → `harness/evaluator.py`), and a
genuinely good engineering decision in the sequential-loading llama.cpp cache
(`harness/llamacpp_engine.py`) that makes a 32GB machine viable for a panel of
multi-GB local models. The prompt/parsing layer
(`judges/prompts.py::parse_json_from_response`) is defensive in the right ways
(code-fence stripping, balanced-brace scanning) for the class of malformed
output small local models actually produce.

As a **research instrument** for the two questions it claims to answer,
though, it currently cannot answer either one:

- **Fine-tuning/quantization invariance**: nothing in the schema or code links
  a fine-tuned model to the base checkpoint it was derived from, or one
  quantization of a checkpoint to another. `config/models.yaml`'s
  `fine_tuned_models` list is nine unrelated community checkpoints (Gemma,
  Qwen, Llama, mixed provenance) with no `base_model_id` /
  `quantization_variant_of` field, and there is no comparison/diff module
  anywhere in the repo (confirmed — no `analysis/`, no `compare*`, no
  `baseline*` in the codebase). The harness can only ever produce independent
  per-model pass rates; "did behavior change under fine-tuning" has to be
  computed by hand from two independently-run result files, keyed by nothing
  more than eyeballing model ids.
- **Judge bias meta-evaluation**: `judges.judge_panel.aggregate_verdicts`
  (`judges/judge_panel.py:91-116`) is a naive majority vote over verdict
  *labels*; there is no inter-judge agreement statistic (Cohen's κ or
  otherwise), no per-category/per-technique bias breakdown, no calibration
  check against ground truth, and — as detailed in §2 — currently no
  judge in the panel that is actually independent of the models under test.
  The infrastructure to *run* judges exists; the infrastructure to
  *evaluate the judges* does not.

So: solid scaffolding for a two-pass generate→judge pipeline, but neither of
the two research questions in the project's own name currently have a code
path that could produce an answer, even with unlimited compute.

## 2. Main failure modes

Ordered by how badly each one undermines the stated research goal, not by
where it sits in the code.

### 2.1 The judge panel currently judges itself
`config/judges.yaml`'s ten enabled local judges have **identical `id` and
`checkpoint` values** to ten of the entries in `config/models.yaml`'s
`local_models`/`fine_tuned_models` (verified by set-intersecting the two
configs — all ten overlap: `gemma-4-12b`, `Ornith-1.0-9B`, `qwen3.5-9b`,
`llama-3.1-8b`, `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive`, `qwythos-9b`,
`Qwen3.5-9B-The-Defiant-Fable-Uncensored`, `Dolphin3-Cyber`,
`gemma-4-12B-agentic-fable5`, `Qwopus3.5-9B-Coder`). Both remote judges
(Claude, GPT-4o in `config/judges.yaml:22-36`) are commented out. So a default
`--judges all` run today evaluates each model's output using — among others —
that same model, and because `llamacpp_engine.get_engine` keys its cache by
`(repo_id, filename)` (`harness/llamacpp_engine.py:224-233`), the model-under-
test and the "judge" for that row can literally be **the same resident
in-process engine object**. This isn't a subtle bias risk, it's self-grading.
No study of "are judges biased" is possible until at least one judge in the
enabled set is model-independent of every model under test — practically,
that means turning the remote judges back on, or reserving a disjoint subset
of local checkpoints as judges-only.

### 2.2 `system_prompt` is parsed into the data but never reaches the model
`data/test_suite_v1.json` was expanded (uncommitted in this working tree,
`+188/-10` per `git diff --stat`) from a 3-case example into a real 60-case
suite, and roughly 20% of it (`factual-context-02`, and the eleven
`system-*` cases: `system-permissive-01/02`, `system-restrictive-01/02`,
`system-role-researcher-01`, `system-role-engineer-01`, `system-conflicting-
01`, `system-utility-01`, `system-factual-permissive-01`, `system-factual-
conflicting-01`, `system-reasoning-permissive-01`) exists specifically to test
whether a system prompt can shift the model off a safety-consistent baseline.
Every one of those cases carries a `system_prompt` field. Confirmed by
`grep -rn system_prompt --include=*.py .`: the string appears **nowhere** in
`harness/` or `judges/`. `TestCase` (`harness/test_suite.py:36-42`) has no
`system_prompt` field and `from_dict` doesn't read it; `TestRunner._run_one`
(`harness/test_runner.py:70-94`) calls `model.generate(test.prompt, ...)`
with nothing else; and even if it did, `LlamaCppModel.generate`
(`harness/model_loader.py:104-116`) and `LlamaCppEngine.chat`
(`harness/llamacpp_engine.py:343-364`) hard-code a single `{"role": "user", ...}`
message with no system-role support in the plumbing either. The loss compounds
silently: `factual-context-02`'s `prompt` field is *byte-identical* to
`factual-baseline-01`'s, so after loading, the two test cases send the exact
same request to the model — the entire manipulation the test exists to apply
(a journalist-role system prompt) never happens, and nothing logs or errors to
say so. This is a silent-failure bug, not a missing feature: the suite author
wrote a test whose premise the runner cannot execute, and the run completes
"successfully."

### 2.3 The `cheap` judge preset is the most expensive one
The docs (README, CLAUDE.md) describe `--judges cheap` as "remote judges
only." The preset is implemented as `priority <= 1`
(`harness/evaluator.py:52`). But every local llama.cpp judge entry in
`config/judges.yaml` omits `priority`, and `build_judges`
(`judges/factory.py:82-112`) defaults missing priority to `1`
(`entry.get("priority", 1)`) — the same default the (currently disabled)
remote judges would also use. With the remote judges commented out and every
local judge implicitly `priority: 1`, `--judges cheap` today builds and
sequentially loads all ten multi-gigabyte local GGUF judges — the opposite of
"cheap." This is a config/default interaction that will silently do the wrong
thing the first time someone re-enables the remote judges without also
explicitly setting `priority: 2` on every local entry (the two commented-out
`priority: 2` examples in the file's history suggest that was the intent, but
none of the ten live entries carry it).

### 2.4 Multi-turn tests are single strings, not conversations
Several `reasoning-coherence`/`factual-stability` cases encode a "did the
model stay consistent under a follow-up" probe as one prompt string with an
inline `[Follow-up]` marker (e.g. `reasoning-consistency-06`,
`factual-contradiction-06`, `jailbreak-escalation-08`). Because
`TestRunner`/`ModelLoader` only ever issue one `generate()` call per test case
with no conversation state, the model never actually commits to an answer for
"part A" before being confronted with "part B" — it sees both halves
simultaneously as one static block of text. That measures something (how the
model handles a compound prompt), but not the thing the category name claims
to measure (whether the model's own prior answer holds up under a follow-up).
A genuine consistency test needs two real turns with the model's first
response fed back in.

### 2.5 No real ground truth, no disagreement analysis
`HeuristicJudge` (`judges/heuristic_judge.py`) is documented as "a stable
reference point," but it only has real signal for the `refusal-consistency`
category's `expect:refuse`/`expect:comply` directive
(`_extract_directive`, line 86-88) — for the other four categories
(`factual-stability`, `reasoning-coherence`, `format-adherence`,
`jailbreak-susceptibility`) it degrades to a coin-flip `AMBIGUOUS` at
confidence 0.2-0.4 (lines 67-74). So for ~two-thirds of the suite, the one
judge that's guaranteed independent of the models under test contributes
almost no information to the majority vote. There is no human-labeled gold
set anywhere in the repo to check any judge's accuracy against, and (per §1)
`aggregate_verdicts` computes agreement/vote-counts but no bias-by-category
breakdown, so even the data that *is* being collected isn't being turned into
the "is judge X systematically biased" answer the project is named for.

### 2.6 Deterministic categories are graded by a non-deterministic LLM
`format-json-01`, `format-csv-05`, `format-sentences-02` etc. have criteria
that are mechanically checkable ("must be valid JSON," "exactly 3 sentences,"
"valid CSV with these headers"). Nothing in the codebase runs `json.loads()`,
a sentence tokenizer, or a CSV parser — every one of these is graded by
sending the output through the same LLM-judge prompt template as the safety
categories (`judges/prompts.py:15-41`), at real API cost and with real
false-verdict risk, for a question that has a free, exact, zero-latency
answer.

### 2.7 Small, non-reproducible sample per (model, test)
Generation defaults to `temperature=0.7`
(`harness/model_loader.py:37,69-71,114-115`) with exactly one draw per
(model, test) pair and no seed control exposed at the CLI/config level (an
engine-wide `seed` key exists in `_ENGINE_KWARG_KEYS` but nothing repeats a
generation with a different seed or reports variance). A single non-
deterministic sample turns "model X fails test Y" into a noisy one-shot
Bernoulli draw with no confidence interval, which is a weak foundation for
comparing behavioral-invariance across fine-tunes where the true effect size
may be small.

### 2.8 Judge context window is tight relative to what gets stuffed into it
`config/hardware_profile.yaml` defaults `n_ctx: 2048`. A local judge's
prompt is `JUDGE_PROMPT_TEMPLATE` + the full `criteria` string + the full
`model_output` (`judges/prompts.py:44-50`), and `model_output` can be up to
`max_tokens=512` tokens of test-model generation (the default cap in
`harness/model_loader.py:36`) plus the format-adherence prompts that
explicitly ask for longer structured documents (NDAs, emails). It would not
take much for a verbose model response plus the judge template to approach or
exceed 2048 tokens, silently truncating the judge's view of exactly the part
of the output that matters.

### 2.9 No retry/backoff; a transient failure is indistinguishable from a genuine "ambiguous"
`Judge.evaluate` (`judges/base.py:85-114`) catches every exception and turns
it into a permanent `AMBIGUOUS, confidence=0.0` verdict with no retry. A rate
limit, a network blip, or a single malformed JSON response from a judge that
would have succeeded on a second attempt is recorded identically to a judge
genuinely uncertain about the content. Since outputs are expensive to
regenerate but judges are (relatively) cheap to re-run, this trades a small
amount of complexity for a real loss of data quality, especially for the
remote judges where transient failures are the norm rather than the
exception.

## 3. What could be improved

Roughly in priority order, matched to the failure modes above:

1. **Break the judge/model identity confound (2.1).** Either re-enable the
   remote judges (they're fully implemented, just commented out) as the
   trusted anchor, or partition local checkpoints into a "models under test"
   set and a disjoint "judges" set that never overlap by `(repo_id, filename)`.
   This is the single highest-leverage fix — nothing else in the meta-eval
   story works without it.
2. **Wire `system_prompt` end-to-end (2.2).** Add the field to `TestCase`,
   thread it through `TestRunner`/`ModelLoader`/`GenerativeModel.generate`,
   and add a system-role message in `LlamaCppEngine.chat` and the Ollama
   payload. This is a small, mechanical, high-value fix given it's silently
   defeating ~20% of the suite today. Cheap regression test: assert two
   `TestCase`s with the same `prompt` but different `system_prompt` produce
   different rendered chat messages.
3. **Fix the `cheap` preset (2.3).** Either give every local judge an explicit
   `priority: 2+` in `config/judges.yaml`, or invert the semantics so
   "cheap" is provider-based (`access == "api"`) rather than priority-based.
4. **Preserve the suite's rich metadata.** `domain`, `technique`, `severity`,
   `source` are present in every test case but silently dropped by
   `TestCase.from_dict` (only `id`/`prompt`/`criteria`/`category`/`metadata`
   are read). Either nest them under `metadata` in the data file or extend
   `TestCase` to capture them — without this, no per-technique or
   per-severity breakdown of results is possible even after the suite work is
   otherwise done.
5. **Add a deterministic pre-check for mechanically-verifiable criteria
   (2.6).** A `format-adherence` category with `is_json`/`sentence_count`/
   `is_valid_csv` metadata could be checked in Python before ever calling an
   LLM judge, saving cost and removing an unnecessary source of judge noise.
6. **Real multi-turn support for consistency tests (2.4).** Extend
   `GenerativeModel.generate` to accept a list of turns (or add a
   `generate_conversation`) so `[Follow-up]`-style cases actually see the
   model's own first answer before the second prompt.
7. **Turn `aggregate_verdicts` into the meta-evaluation it's named for.**
   At minimum: pairwise inter-judge agreement (κ) per category, judge-vs-
   heuristic-baseline divergence, and confidence calibration if any gold
   labels are ever added. This is explicitly called out as `[Sam]`'s work in
   the docs, so it's not a "bug," but it's the actual deliverable the repo
   is scaffolding toward and is currently a placeholder majority vote.
8. **Add retries with backoff for judge calls (2.9)**, and distinguish
   "judge call failed" from "judge returned AMBIGUOUS" in the persisted
   verdict (there's already an `error` field on `Verdict` — the aggregation
   layer should exclude error'd verdicts from the vote rather than counting
   them as a normal AMBIGUOUS).
9. **Repeat sampling** for at least a subset of (model, test) pairs (e.g. n=3
   at a fixed temperature, or n=1 at temperature 0 for a deterministic
   baseline run) so pass/fail rates carry an uncertainty estimate.
10. **Record provenance in the output files** — suite file path + a content
    hash, harness git commit, model checkpoint/quant, judge panel — in
    `model_outputs.json`/`model_verdicts.json` so a result file is
    self-describing months later. Currently only `test_id`/`model` are
    recorded per row.
11. **Add a `base_model_id` / `quantization_of` field to `models.yaml`** and
    a small comparison script that, given two result files and that linkage,
    reports the actual behavioral-invariance diff (pass-rate delta per
    category, flipped verdicts by test id). Without this the "invariance"
    half of the project's name has no code path at all.

## 4. How can this be scaled?

**Suite size.** The JSON-with-`evals`-array schema and the loader
(`harness/test_suite.py`) scale to hundreds of cases with no code changes —
it's already validating for duplicate ids and required fields. The real
scaling bottleneck is *judged* throughput, not suite size: with
`max_resident: 1` (strict sequential local-model loading,
`config/hardware_profile.yaml:34`), wall-clock time is roughly
`(#models + #local_judges) × suite_size × per_call_latency`, since every model
switch pays a full GGUF load. On a single 32GB Mac this is workable for
today's ~10 models × ~5 judges × 60 cases, but doesn't scale past that without
either (a) raising `max_resident` on a machine with more memory to keep
several judges resident simultaneously, or (b) moving to a job-queue model
where independent (model, judge) pairs run on separate machines/processes —
the current design is explicitly single-process (module-level
`_ENGINE_CACHE` in `harness/llamacpp_engine.py:64`), so horizontal scaling
would need that cache made external (e.g. a small local model-serving daemon
per worker, or switching heavier models to Ollama/vLLM behind HTTP) rather
than in-process.

**Model count.** Adding models is one YAML entry each — cheap. But every
model added to `local_models`/`fine_tuned_models` multiplies total run time
linearly with no parallelism across models today (the `for model_id in models`
loop in `TestRunner.run_tests` is strictly sequential,
`harness/test_runner.py:55-63`). Running test-model generation and judge
evaluation for genuinely independent models is embarrassingly parallel across
processes/machines — the two-pass output/verdict file design already supports
that (outputs from N separate runs can be concatenated before judging), but
nothing in the repo currently orchestrates it.

**Judge count.** Config-driven judge addition already works well (PRD's
"add/remove judges without code changes" goal is genuinely met). Scaling the
*meta-evaluation* side (more judges → more inter-judge comparisons) is purely
a data-analysis problem once §3.7 exists — the current per-row verdict list
already carries what's needed (`PanelResult.verdicts`).

**Cost.** Remote judges are metered per token; at suite-size 60 × 10 models ×
2 remote judges that's 1,200 calls per full run — cheap in absolute terms
(~256 max output tokens each) but worth tracking via
`JudgePanel.cost_summary()` (already implemented,
`judges/judge_panel.py:75-88`) as both suite and model count grow.

## 5. Viability for a larger research piece

The core idea — behavioral-invariance testing across quantization/fine-tuning
*and* meta-evaluating the judges used to score it — is a legitimate, useful
research question, and the two-pass generate/judge architecture is the right
shape for it (decoupling expensive generation from cheap re-judging is
exactly what lets you iterate on judge methodology without re-running every
model). The engineering foundation (config-driven judges, lazy imports, the
llama.cpp sequential-loading cache, the shared prompt/parsing contract) is
solid enough to build on.

But as it stands, this is **not yet a viable research artifact** — it's
scaffolding that would produce numbers, but numbers that don't support the
paper's central claims, for three compounding reasons:

- The self-judging confound (2.1) means any "judges are biased" finding
  produced today would be an artifact of models grading themselves, not a
  measurement of judge bias in general.
- The dropped `system_prompt` (2.2) means the specific hypothesis the suite
  was designed to test on ~20% of its cases — does framing/role-play move a
  model off its safety baseline — currently cannot produce a true positive or
  negative; every one of those rows degenerates to a duplicate of its
  baseline case.
- There is no linkage between fine-tuned/quantized variants and their base
  models (§1, §3.11), so the "invariance" half of the research question has
  no computable answer even with clean generation and judging.

None of these are hard to fix — §3 items 1-4 are a few days of focused work,
not a redesign — but until they land, any run of this harness would produce a
results file that looks complete and well-formed while silently not
measuring what the write-up would claim it measures. For a larger research
piece, I'd treat items 1, 2, and 11 as blocking (they each defeat a stated
research question outright), items 3, 4, 6, 7, 9 as needed before results are
trustworthy enough to publish, and the rest as hardening.

## 6. Additional considerations

- **Ethical/provenance surface.** The suite embeds real jailbreak techniques
  (DAN, policy-puppetry, token-economy, encoding obfuscation) sourced from
  HarmBench/APE. That's appropriate for a safety-eval harness, but if this
  suite or its outputs are ever published, the `source` field (currently
  dropped on load, see §2.4/§3.4) is exactly the kind of provenance a
  reviewer or downstream user would need preserved.
- **Test-suite versioning.** `data/test_suite_v1.json` is edited in place
  with no version pinning beyond the filename, and results files don't record
  which suite revision produced them (§3.10). For a longitudinal study
  ("does behavior drift as we update the suite"), that linkage needs to exist
  from day one, not retrofitted later.
- **`ReplicateModel.generate` raises `NotImplementedError`**
  (`harness/model_loader.py:130-134`) — any model needing cloud fallback
  (explicitly called out for the too-large-for-local-serving case, e.g. a
  70B fine-tune) is a hard blocker today, not a degraded path.
- **Test coverage of the harness itself is reasonable** (75 test functions
  across `tests/`, network-free per `CLAUDE.md`'s stated convention) and
  should make the fixes in §3 straightforward to land safely — the gaps here
  are in research design and data plumbing, not in engineering discipline.
