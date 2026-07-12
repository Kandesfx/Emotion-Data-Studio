# AI Coding Agent Workflow — Emotion Data Studio

> **Purpose:** This document describes the engineering workflow I used to build
> Emotion Data Studio (EDS) with AI coding agents as a force multiplier.
> It is the audit trail for how AI-generated code was specified, generated,
> reviewed, and shipped to production.

---

## 1. Philosophy

**AI agents amplify engineering judgment — they do not replace it.**

For this project I followed three invariants:

1. **Spec before code.** Every non-trivial change starts as a written spec
   (sprint plan, design doc, or PR description) that a human reviewer can
   accept or reject *before* any code is generated.
2. **AI-generated code is a first-class artifact.** It gets the same review
   rigor as human code: tests, edge cases, security, naming, and a post-mortem
   when it ships wrong.
3. **Sprint logs are living documentation.** Every sprint writes back into
   `docs/03_ai_autocut_optimization.md` what worked, what failed, and what to
   do differently. AI is great at generating code; humans are still better at
   noticing when a pattern stops working.

---

## 2. Workflow per Sprint

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. SPEC     │ ─► │  2. GENERATE │ ─► │  3. REVIEW   │ ─► │  4. SHIP     │
│              │    │              │    │              │    │              │
│ Human writes │    │ AI agent     │    │ Human + AI   │    │ CI + manual  │
│ sprint plan  │    │ drafts code, │    │ verify every │    │ E2E test on  │
│ + acceptance │    │ tests, and   │    │ diff against │    │ real video;  │
│ criteria     │    │ docs from    │    │ spec; reject │    │ rollback if  │
│              │    │ spec         │    │ anything off │    │ regression   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 2.1 Spec — What the human owns

Every sprint opens with a written plan that fixes:

- **Goal** (one sentence)
- **Done criteria** (checkbox list, must be observable)
- **File targets** (which modules will change)
- **Out of scope** (explicit list — prevents AI from drifting)

Example from Sprint 1:

```markdown
## Sprint 1 — Foundation
- Goal: replace regex parsing with structured Gemini JSON output.
- Done criteria:
  - [ ] Gemini returns JSON array, no regex fallback needed
  - [ ] Hard filter rejects 5 sample segments with correct reasons
  - [ ] All existing tests still pass
- Files: backend/services/ai_video_segmenter.py, backend/config.py
- Out of scope: changing Stage 3 (face/audio), training, MMSA exporter
```

**Why this matters for AI agents:** AI coding agents work much better when
given a concrete acceptance criterion than when given a vague goal like
"improve Gemini integration."

### 2.2 Generate — What the AI owns

Once the spec is fixed, the AI agent drafts:

- Production code following the file's existing style
- Unit tests with explicit assertions on edge cases
- A draft of the sprint log entry

I never ask the AI to "decide the architecture." That decision is locked in
the spec. The AI's job is mechanical: translate spec → code → tests → docs.

### 2.3 Review — The non-negotiable step

Every AI-generated change goes through this checklist before merge:

| # | Check | Why it matters |
|---|---|---|
| 1 | Does the diff match the spec exactly? | AI tends to add "helpful" features that drift scope |
| 2 | Are error paths handled? | AI often writes the happy path and forgets exceptions |
| 3 | Are defaults safe? | AI may pick `default=0` which silently rejects all clips (Bug #2 Sprint 3.8) |
| 4 | Does it break existing tests? | Run the full suite, not just the new test |
| 5 | Are secrets logged? | Look for `print(api_key)` or similar |
| 6 | Is the public API backward-compatible? | Existing callers (UI, API clients) must not break |
| 7 | Did the AI add new dependencies? | Each new dep needs justification |
| 8 | Is naming consistent with the codebase? | `clip_path` vs `clip_filepath` — pick one |
| 9 | Are TODOs explained, not just left? | AI loves `TODO: handle this later` |
| 10 | Did the AI actually test it? | Or did it just write a test that always passes? |

A "no" on any of these sends the change back for revision, not commit.

### 2.4 Ship — Tests, E2E, then production

Before a sprint is marked done:

1. **Unit tests pass** (133/133 in our case — see `tests/test_sprint*_autocut.py`)
2. **Integration tests pass** (`scripts/verify_integration_imports.py`)
3. **End-to-end test on real data** (`data/scripts/test_e2e_bongma.py`)
4. **Cost verified** (token usage logged, under budget)
5. **Sprint log written** (decisions, deviations, lessons learned)

If any step fails, the sprint is *not* done — regardless of how the code looks.

---

## 3. Real examples from this codebase

These are not hypotheticals. Every entry below is a real change shipped
through this workflow.

### 3.1 Sprint 1 — Hard filter & structured JSON

**Spec:** Gemini should return JSON conforming to a schema; raw output not
passing the hard filter must be rejected with a specific reason.

**AI-generated:** `_build_response_schema()`, `_call_gemini_with_json_enforced()`,
`_hard_filter_clip_quality()` with 11 checks.

**Human review caught:** None on first pass — but the test script was too
permissive. Tightened the assertions before merging.

**Result:** Yield rate became observable (`stage1_candidates` → `stage2_passed`),
replacing "0 or many segments" with a measurable funnel.

### 3.2 Sprint 2 — Two-pass verification

**Spec:** After Scan pass cuts clips, Verify pass independently re-checks
each clip's emotion + quality; merge rules defined per verdict enum.

**AI-generated:** `verify_clip()`, `_call_gemini_verify_clip()`,
`combine_verdicts()` with 5 branches.

**Human review caught:** Schema design issue — Scan pass uses ARRAY,
Verify pass uses OBJECT. AI initially tried to reuse one schema; I separated
them into two functions (`_build_response_schema()` vs
`_build_verify_response_schema()`).

**Lesson logged:** *"Tách prompt Scan vs Verify là bắt buộc — 2 task khác
hẳn nhau (detect vs verify). Schema OBJECT vs ARRAY: dễ sai nếu dùng
chung."*

### 3.3 Sprint 3.8 — Seven bugs found by one E2E run

This is the most important entry because it shows the workflow catching
what the AI (and the unit tests) missed.

**Trigger:** First E2E run on real Vietnamese video (`BÓNG MA HẠNH PHÚC`).

**Bugs found:**

| # | Bug | Why unit tests missed it | AI's role in fix |
|---|---|---|---|
| 1 | `NameError: name 'stage1' is not defined` | Branch was unreachable in synthetic tests | AI generated the fix in 30 sec |
| 2 | Hard filter rejected 100% of segments because Gemini did not return `frontal_ratio` | Unit test always set the field | AI wrote the defensive `if "frontal_ratio" in seg` check |
| 3 | Same root cause for `people_count` | Same | Same |
| 4 | Same root cause for `speech_quality` | Same | Same |
| 5 | `max_output_tokens=1024` truncated Verify response mid-JSON | Synthetic mock didn't hit token limit | AI suggested `4096`, correct |
| 6 | Test script `c.emotion` attribute doesn't exist on new `Clip` schema | Test was written for old schema | AI rewrote with `getattr()` fallback |
| 7 | Persist script printed "FAIL" despite success | Bug #6 → AttributeError swallowed by `except` | AI traced and fixed root cause |

**Outcome:** 4/4 clips confirmed on real video, $1.23 total cost, 0 regressions
in Sprint 1–3 tests. Sprint log section 7.6 documents all seven.

**Why this matters:** AI alone would not have caught Bug #2 — it required
running against a real Gemini response and noticing that the field was
missing. The workflow's job is to put the AI in front of reality.

### 3.4 Sprint 3.9 — UI/API/DB wiring

**Trigger:** After Sprint 1–3 backend was done, I noticed the `/cut-and-create`
API endpoint was still calling the old `cut_video()` directly, bypassing the
new Verify pass.

**Spec:** Endpoint must call `verify_clip()` per clip, merge verdicts,
populate `Clip.verify_*` columns, return new fields in response
(`verify_summary`, `stage_counters`, `cost_breakdown`).

**AI-generated:** ~50 lines in `gemini_api.py`, ~15 lines in `database/models.py`,
~15 lines in `ai_video_segmenter.py`, ~25 lines in `settings_page.py`, ~100
lines in `review_page.py`, plus a new idempotent migration script.

**Human review caught:** The migration script needed `PRAGMA table_info(clips)`
check to be idempotent (running it twice should not fail). AI's first draft
would have crashed on second run.

**Lesson logged:** *"Backend không đủ — UI cũng cần kết nối. Sprint tách
biệt cần phải có bước cuối cùng: wiring tất cả chỗ gọi cũ."*

---

## 4. Prompt patterns that worked

These are the prompt patterns I used repeatedly with AI coding agents.
Each is followed by a real example from EDS.

### 4.1 "Spec → Code → Test" three-shot prompt

```
Here is the spec for Sprint X: [paste spec]
Here is the file you will modify: [paste file]
Constraints:
  - Match existing style
  - Do NOT add features not in spec
  - Write tests with explicit assertions
  - Log changes in docs/03_ai_autocut_optimization.md §7.X
Generate the diff.
```

### 4.2 "Find the bug, do not write the fix yet" pattern

```
This E2E test failed: [paste output]
Hypothesize 3 possible root causes.
For each, point to the file:line that would confirm or rule it out.
Do not write any fix yet.
```

This separates *diagnosis* from *fix*, which catches wrong-direction fixes.

### 4.3 "Reverse review" pattern

```
Here is the PR I want to merge: [paste diff]
You are a hostile reviewer.
List every place this code could:
  - Crash on edge case
  - Leak resources
  - Break a contract documented elsewhere
  - Surprise the next person reading it
Be specific. No "looks good" comments.
```

This pattern found Bugs #2, #3, #4 in Sprint 3.8 that the AI's first drafts
did not catch.

---

## 5. Anti-patterns I avoid

These are the failure modes that AI coding agents fall into by default.
Each one cost me time at least once.

| Anti-pattern | What it looks like | Why it fails | Counter |
|---|---|---|---|
| **Happy-path only** | Code handles the success case, throws on error | Errors in production are not the success case | Force spec to list error paths explicitly |
| **Default-zero silent reject** | `seg.get("frontal_ratio", 0.0)` then check `< 0.75` | Missing field becomes 0, fails check, rejects valid data | Check field presence before applying filter |
| **Helpful drift** | AI adds "while I'm here, let me also refactor X" | Scope creep, harder to review | Lock the spec, reject out-of-scope changes |
| **Test theater** | Test asserts `result is not None` instead of `result == expected` | Test passes even when logic is broken | Write the assertion first, then make it pass |
| **Magic constants** | `if confidence > 0.55` with no name | Reader cannot tell why 0.55 | Always named: `settings.AI_AUTOCUT_INTENSITY_THRESHOLD` |
| **TODO without owner** | `# TODO: fix later` | "Later" never comes | Either fix now or write an issue, no in-between |

---

## 6. Measuring whether the workflow is working

A workflow is only useful if you can tell when it is failing. The signals
I track:

- **Yield rate** (segments passing hard filter / Gemini raw) — should be
  30–50%. If it drops below 20%, either Gemini is failing or the filter is
  too strict.
- **Verify agreement rate** (confirmed / total verified) — should be > 70%.
  Below 50% means Scan and Verify disagree on what "good" looks like.
- **Human override rate** (reviewer changes emotion / total reviewed) — should
  be < 30%. Above 50% means the AI is not learning what humans want.
- **Cost per approved clip** — target < $0.05. Above $0.10 means we are
  calling Gemini on clips that will be rejected anyway.
- **Time per 30-min video** — target < 5 min. Above 10 min means FFmpeg or
  network is the bottleneck, not the AI.

These are not vanity metrics — they tell me when the workflow is shipping
real value vs. when it is just shipping *something*.

---

## 7. What this workflow is NOT

To be precise about scope:

- **Not** a framework. I do not ask the AI to invent a new abstraction on
  every change. Existing patterns in `pipeline_orchestrator.py` are the
  patterns to follow.
- **Not** autonomous. I read every diff before merge. There is no "the AI
  shipped this while I slept" mode.
- **Not** a substitute for testing. AI writes tests, but I run them and read
  the failures.
- **Not** a substitute for domain knowledge. Knowing that COVAREP features
  are 74-dim and MOSEI expects `(50, 74)` is not something the AI knows
  by default — that knowledge came from the spec.

---

## 8. References

- `docs/03_ai_autocut_optimization.md` — full sprint log with all
  decisions, deviations, and lessons learned (Sprint 1 → 3.9).
- `docs/02_ai_autocut_design.md` — architecture and contracts for AI Auto-Cut.
- `docs/09_vertex_ai_integration.md` — Vertex AI + `google-genai` SDK setup.
- `scripts/verify_integration_imports.py` — 10 import-time assertions that
  catch regressions before runtime.
- `tests/test_sprint{1,2,3}_autocut.py` — 133 unit tests.
