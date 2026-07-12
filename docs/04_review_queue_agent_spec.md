# Sprint 4 — Self-tuning Review Queue Agent

> **Spec-first workflow:** This document is the source of truth for the
> Review Queue Agent. Any code change must reference a goal below.
> See `docs/AI_CODING_WORKFLOW.md §2.1` for why.

---

## Goal

Reduce reviewer time per clip by automatically routing clips into
"easy queue" (auto-approvable) or "needs review queue" based on signals
that already exist in the database. The agent does not decide labels —
it only prioritizes human attention.

## Why this is worth doing

Today the reviewer sees clips in raw order. When there are 200 clips
from a 30-min video, the reviewer must look at every one. But:

- Clips where Gemini Scan + Verify both agree AND quality_score > 0.85
  AND no incongruity → almost always approved by humans
- Clips with `verify_verdict='wrong_emotion'` or `rejected_by_verify=True`
  → almost always rejected by humans

If we surface these patterns to the reviewer first, they can clear the
easy 60% in minutes and focus the remaining 40% of their attention on
clips where humans disagree with AI.

## Done criteria

- [ ] `ReviewQueueAgent` class in `backend/services/review_queue_agent.py`
- [ ] Returns ordered list of clip IDs grouped into 3 buckets:
      `auto_approve_candidates`, `auto_reject_candidates`, `needs_human_review`
- [ ] Each clip has a `confidence: float` (0.0–1.0) and `reasons: list[str]`
      explaining the routing decision
- [ ] Agent is **read-only** — it never modifies the database
- [ ] All thresholds read from `settings.*` (configurable, not hardcoded)
- [ ] ≥ 15 unit tests covering each routing rule + edge cases
- [ ] One integration test that loads 30 synthetic clips and asserts the
      expected bucket distribution
- [ ] README updated with a "Review Queue Agent" section

## Files

**New:**
- `backend/services/review_queue_agent.py` — the agent
- `tests/test_review_queue_agent.py` — unit + integration tests

**Modified:**
- `backend/config.py` — add `REVIEW_QUEUE_*` settings
- `tools/emotion-data-studio/README.md` — link to spec + agent

**Out of scope (deliberately):**
- No UI changes — agent is a library, not a widget
- No DB schema changes — agent reads only
- No Gemini calls — agent is fully offline
- No auto-write of `status='approved'` — human approval stays human

## Routing rules

| Rule | Bucket | Why |
|---|---|---|
| Verify verdict `confirmed` AND `ai_confidence ≥ HIGH` AND `quality_score ≥ HIGH` | `auto_approve_candidates` | Two AI passes agree + high quality |
| `verify_verdict='wrong_emotion'` OR `rejected_by_verify=True` | `auto_reject_candidates` | Verify pass explicitly rejected |
| `ai_confidence < LOW` | `auto_reject_candidates` | AI unsure enough that human rarely approves |
| `has_incongruity=True` AND no `verify_status` | `needs_human_review` | Incongruity is the strongest signal for human override |
| Everything else | `needs_human_review` | Default — preserve human authority |

**Thresholds (config in `backend/config.py`):**
```
REVIEW_QUEUE_AUTO_APPROVE_CONFIDENCE = 0.85
REVIEW_QUEUE_AUTO_APPROVE_QUALITY = 0.85
REVIEW_QUEUE_AUTO_REJECT_CONFIDENCE = 0.40
REVIEW_QUEUE_BUCKET_AUTO_APPROVE_RATIO = 0.30  # soft target
REVIEW_QUEUE_BUCKET_AUTO_REJECT_RATIO = 0.20
```

The ratio targets are observed via log, not enforced. They tell us whether
the thresholds are too aggressive.

## Confidence scoring

Each clip gets a confidence score per bucket:

```
auto_approve_confidence = w1 * (1 - (HIGH - ai_confidence)/HIGH)
                       + w2 * (1 - (HIGH - quality_score)/HIGH)
                       + w3 * (verify_agreement == 'confirmed')
                       + w4 * (1 - has_incongruity)
```

where `w1 + w2 + w3 + w4 = 1.0`. Default: `w1=0.30, w2=0.20, w3=0.40, w4=0.10`.

The Gemini verify weight is highest because it's the most expensive and
most recent signal.

## Self-tuning (the "self-" part)

After the agent runs, it logs to `logs/review_queue_agent.jsonl`:

```json
{
  "ts": "2026-07-11T10:00:00",
  "n_clips": 30,
  "auto_approve": 12,
  "auto_reject": 5,
  "needs_review": 13,
  "actual_ratios": {"approve": 0.40, "reject": 0.60},
  "predicted_ratios": {"approve": 0.40, "reject": 0.17}
}
```

This lets us detect drift: if the agent predicts 30% auto-approve but
reviewers actually approve 70%, the threshold is too low.

**v1 only logs.** v2 (out of scope for this sprint) will adjust thresholds
automatically based on accumulated reviewer feedback.

## Public API

```python
from backend.services.review_queue_agent import ReviewQueueAgent

agent = ReviewQueueAgent(session)
result = agent.run()
# result.auto_approve_candidates: list[ClipBucket]
# result.auto_reject_candidates: list[ClipBucket]
# result.needs_human_review: list[ClipBucket]
# result.summary: dict (for logging)

for bucket_entry in result.auto_approve_candidates:
    print(bucket_entry.clip_id, bucket_entry.confidence, bucket_entry.reasons)
```

## Anti-goals

These are explicitly NOT in scope:

- ❌ Auto-approve clips (status='approved' requires human)
- ❌ Modify Gemini prompts or thresholds
- ❌ Call Gemini or any LLM (agent is offline)
- ❌ New DB columns (use existing `per_model_scores` JSON)
- ❌ UI integration (agent is a backend library)
- ❌ ML model training (no gradient-based learning)

The "agent" in the name refers to the routing decision-making, not to
LLM autonomy. This is a deterministic rule engine with logging.

## Open questions

- Q: Should we expose `confidence` to the UI?
  A: v1: no. v2: yes, as a small badge next to clip name.
- Q: Should the agent skip clips where `status='approved'` already?
  A: v1: no — it returns all clips so the UI can show distribution.
       The UI is responsible for filtering.
- Q: How do we handle clips with `verify_status='not_run'`?
  A: They go to `needs_human_review` with reason
       "verify_not_yet_run".

## Validation plan

1. **Unit tests** (`tests/test_review_queue_agent.py`):
   - 3 routing rules × 3 branches (rule applies, rule doesn't apply, edge case)
   - Empty input, single clip, 1000 clips
   - All thresholds from config (mocked)
   - Confidence score is in [0.0, 1.0]
   - Reasons list is non-empty for every clip

2. **Integration test**:
   - Generate 30 synthetic clips covering all combinations
   - Assert bucket distribution within ±10% of predicted ratios
   - Assert ordering within each bucket by confidence DESC

3. **Manual review**:
   - Run on real clips from `data/test_outputs/bongma_raw_gemini.json`
   - Verify auto_approve_candidates actually match what a human would approve

## Sprint log entry

To be filled in by the implementer after Done criteria are met.
See `docs/03_ai_autocut_optimization.md` §8 for the template.

---

*Last updated: 2026-07-11 — Sprint 4 spec.*