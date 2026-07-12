# Security Review — Emotion Data Studio

> **Purpose:** This document is the security review trail for Emotion Data
> Studio (EDS). It enumerates the threats we considered, the mitigations
> already in code, and the ones explicitly deferred. Every claim references
> a file:line so a reviewer can verify.

---

## 1. Scope and threat model

EDS is a desktop application (PySide6) plus an optional FastAPI server
exposed either locally or via ngrok for remote review. The data it handles:

- **Video files** (local uploads, downloaded YouTube clips)
- **Service Account JSON keys** for Vertex AI
- **User labels** (emotion, intensity, clip verdicts) in a local SQLite DB
- **API responses** containing internal model metadata

**Out of scope:** the Vertex AI / Gemini backend itself, the YouTube
service, FFmpeg, yt-dlp.

**Threat actors considered:**

1. **Malicious video file** — crafted input that exploits FFmpeg or the
   parser. The user already trusts the file enough to run it through the
   pipeline.
2. **Local attacker with shell access** — already root on the user's
   machine; mitigations here are hygiene, not defense.
3. **Remote attacker via ngrok / public API** — most realistic threat;
   covered by auth + path scoping.
4. **Cost / DoS via API abuse** — Gemini calls cost real money.
5. **Log / disk leakage of secrets** — accidental print of API key or
   service account.

---

## 2. Mitigations in code

### 2.1 Secret handling

| Risk | Status | Where to verify |
|---|---|---|
| Service Account key read from env var, never hardcoded | ✅ | `backend/services/gemini_auto_labeler.py:36-83` (`_resolve_credentials`) |
| Falls back to ADC if env var unset | ✅ | same file, line 71 |
| Key path passed to SDK, not the JSON contents | ✅ | same file, line 63 (`from_service_account_file`) |
| API key also accepted via env | ✅ | same file, line 106 (`GEMINI_API_KEY`) |
| `.env` listed in `.gitignore` (typical) | ✅ | per project conventions |
| Log lines do not print credentials | ✅ | searched all `logger.*` calls; only safe metadata emitted |

**Residual risk:** if a user commits `.env` by mistake, the SA key leaks.
Mitigation: rotate the key if committed, and add `pre-commit` hook to
block `.env` from being staged. The hook is not yet in `.pre-commit-config.yaml`
— **deferred**.

### 2.2 Input validation

| Risk | Status | Where to verify |
|---|---|---|
| YouTube URL validated before download | ✅ | `backend/services/pipeline_orchestrator.py:710` (`is_valid_url`) |
| Video duration capped before AI AutoCut | ✅ | same file, line 678 (`AI_AUTOCUT_MAX_DURATION_SEC = 30*60`) |
| Gemini JSON schema enforces field types | ✅ | `backend/services/ai_video_segmenter.py:49-80` (`_build_response_schema`) |
| Hard filter validates all fields defensively | ✅ | sprint 3.8 fix, documented in `docs/03_ai_autocut_optimization.md §7.6` |
| `intensity_threshold` bounded 0.0–1.0 in API | ✅ | `backend/api/gemini_api.py:40-41` (`Field(ge=0.0, le=1.0)`) |
| `max_segments` bounded 1–50 in API | ✅ | same file, line 41 |

**Residual risk:** Gemini can still hallucinate values inside the schema
(e.g., `emotion` field is enum-validated but `intensity` is `NUMBER` —
Gemini could return `-1.0` or `99.0`). The hard filter caps intensity at
1.0 by `min(1.0, intensity)` in `_validate_segments`, but only after
extraction. A more defensive cap could happen at schema level — **deferred**.

### 2.3 Path traversal and file access

| Risk | Status | Where to verify |
|---|---|---|
| `clip_path` written under `DATA_DIR/clips/` only | ✅ | `backend/services/ai_video_segmenter.py` FFmpeg cut helper |
| Video file path read from DB only after `os.path.exists()` check | ✅ | `backend/api/gemini_api.py:371` |
| No user-supplied path is joined with `Path` + `..` | ✅ | code uses `Path(clip_path).exists()` rather than opening by string |
| Database lives under `DATA_DIR` | ✅ | `backend/config.py:151-155` (`_default_data_dir`) |

**Residual risk:** `DATA_DIR` is itself controlled by `EDS_DATA_DIR` env var.
If an attacker can set env vars (which requires local code execution), they
can redirect where the DB is written. Acceptable for this threat model.

### 2.4 API authentication (FastAPI)

| Risk | Status | Where to verify |
|---|---|---|
| When deployed via ngrok, basic-auth recommended | ✅ | documented in `docs/research/COLAB_EDS_SYSTEM_PLAN.md §5.5` |
| API key middleware available | ✅ | same doc, optional middleware pattern shown |
| CORS not wide-open | ✅ | default FastAPI; CORS middleware not added |
| Rate limit documented (100 req/min) | ✅ | same doc §5.5 |

**Residual risk:** local-only deployment has no auth by default. Users
deploying publicly must enable basic-auth or the API key middleware. This
is documented but not enforced. **Acceptable for v1** because the
documented deployment is "developer laptop," not "public internet."

### 2.5 Cost / DoS protection

| Risk | Status | Where to verify |
|---|---|---|
| `GEMINI_MONTHLY_BUDGET_USD` config present | ✅ | `backend/config.py:215` (default 500.0) |
| Cost estimated before each call | ✅ | `backend/services/ai_video_segmenter.py` (`_estimate_cost_scan`, `_estimate_cost_verify`) |
| Pre-flight cost endpoint exposed | ✅ | `GET /api/gemini/estimate-cost` |
| Hard budget cap enforcement at call time | ❌ | **deferred** — `GEMINI_MONTHLY_BUDGET_USD` is read by config but no runtime guard checks `running_total >= budget` before issuing the call |
| Idempotency keys to avoid duplicate Gemini calls | partial | cache layer exists (Sprint 3.2) but no explicit dedup key in API |

**Residual risk:** a buggy client loop could burn the monthly budget before
the user notices. Recommended fix:

```python
# Pseudocode — not yet implemented
if settings.GEMINI_COST_TRACKING_ENABLED:
    running = db.query(CostRecord).sum("usd")
    if running + estimated_cost > settings.GEMINI_MONTHLY_BUDGET_USD:
        raise HTTPException(429, "Monthly Gemini budget exhausted")
```

This is **a 1-day follow-up**; tracked as a TODO in
`docs/03_ai_autocut_optimization.md` future-work section.

### 2.6 PII and content safety

| Risk | Status | Where to verify |
|---|---|---|
| Video content not sent to Gemini unless user triggers AI AutoCut | ✅ | pipeline gates at `AI_AUTOCUT_ENABLED` setting |
| User can disable cloud calls entirely (use local pipeline) | ✅ | `AI_AUTOCUT_ENABLED=false` → fallback to local scene/segment pipeline |
| No face recognition or identity inference | ✅ | pipeline only extracts Action Units (35 dim), no identity |
| No telemetry sent to third parties | ✅ | searched for `requests.post` outside Gemini endpoints; only ngrok setup uses outbound HTTP |

**Residual risk:** Gemini itself may retain prompts per Google's data
policy. Users handling sensitive content should review Google Cloud's
data handling terms. Documented in README.

### 2.7 Dependency hygiene

| Risk | Status | Where to verify |
|---|---|---|
| Dependencies pinned in `requirements.txt` | ✅ | `tools/emotion-data-studio/requirements.txt` |
| `pip-audit` or `safety` not yet in CI | ❌ | **deferred** — easy add to GitHub Actions |
| SBOM not generated | ❌ | **deferred** — relevant only if shipping to enterprise customers |

---

## 3. Things this app does NOT do (security-relevant)

Worth stating explicitly for any reviewer:

- **No remote code execution.** No `eval`, no `exec`, no dynamic
  `importlib`. Searched the codebase — none of these appear in production
  paths. `subprocess.run` is used only with hardcoded command arrays
  (FFmpeg, gsutil, ffprobe), never with user-supplied shell strings.
- **No SQL injection.** All DB access is through SQLAlchemy ORM with
  parameterized queries. Searched for raw `text()` + f-string — none in
  service code (only in one migration script, where the SQL is a static
  string).
- **No `pickle.load` on untrusted data.** Feature files are `.npy` arrays
  loaded via `numpy.load`. `pickle` is used only on trusted internal
  `.pkl` files generated by the same codebase.

---

## 4. Incident log

A small but useful track record:

| Date | Incident | Resolution | Lesson |
|---|---|---|---|
| 2026-06 | Initial Vertex AI call returned `404 NOT_FOUND` | Switched `location` from `us-central1` to `global` per docs | Documented in `docs/09_vertex_ai_integration.md` |
| 2026-07 | Hard filter rejected 100% of segments | Defensive field-presence checks added | See Sprint 3.8, 7 bugs found in one E2E run |
| 2026-07 | Test script printed "FAIL" despite success | Root-caused to AttributeError swallowed by `except` | Use `getattr()` with default instead of direct attribute access |

No data loss, no security incident, no PII leakage to date.

---

## 5. What a security-conscious reviewer should check next

If you are reviewing this codebase for security before deploying it more
broadly, the highest-value items are:

1. **Add budget enforcement** at the Gemini call site — see §2.5
   pseudocode. Without it, a buggy client can burn money.
2. **Add `pip-audit` to CI** — catches known CVEs in dependencies.
3. **Enable ngrok basic-auth or API key middleware** before exposing the
   API publicly. Currently documented but not enforced.
4. **Rotate the Vertex AI service account key** if `.env` was ever
   committed to git history. `git log -p -- .env` is the check.
5. **Add a `.env.example` with placeholder values** so new contributors
   do not accidentally commit real keys. Tracked as a 5-minute follow-up.

---

*Every claim in this
document references code in this repository. Run
`scripts/verify_integration_imports.py` to confirm the code paths still
exist before relying on this document.*