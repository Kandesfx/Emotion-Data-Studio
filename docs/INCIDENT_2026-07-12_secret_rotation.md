# Incident Log — Secret Rotation Required (2026-07-12)

## Status: BLOCKED — DO NOT PUSH until steps below complete

## What happened

Commit `748a8e1` ("feat: Emotion Data Studio v1.1 — Colab GPU Worker + Gemini
Auto-Label + Cloud Run") contains `service-account-key.json` which is a Google
Service Account credential. GitHub Push Protection correctly blocked the
push attempt.

**Severity: HIGH** — credentials may have been public on GitHub for an
unknown period before the push was blocked.

## Required actions (in order)

### 1. Rotate the leaked Service Account key

- [ ] Open https://console.cloud.google.com/iam-admin/serviceaccounts
- [ ] Find the project (project_id in the leaked JSON)
- [ ] Locate the SA with the leaked key
- [ ] Delete the leaked key (trash icon)
- [ ] Create new key (Add Key → Create new → JSON)
- [ ] Save new JSON to `tools/emotion-data-stemo/.env` or a path NOT in repo

### 2. Verify the new key works

- [ ] Update `GOOGLE_APPLICATION_CREDENTIALS` env var to new key path
- [ ] Run a small test (e.g., `gcloud auth activate-service-account --key-file=new.json`)
- [ ] Verify Vertex AI / Gemini API calls still succeed

### 3. Purge the leaked key from git history

After rotation, the leaked key is no longer useful, but it remains in git
history. Two options:

**Option A: Accept and move on** (if the leak was caught fast)
- Force-rewrite history using `git filter-repo`:
  ```
  pip install git-filter-repo
  git filter-repo --path service-account-key.json --invert-paths
  git push origin main --force
  ```
- This rewrites every commit; collaborators must re-clone.

**Option B: Just add to `.gitignore` and move on**
- Add `service-account-key.json` to `.gitignore` (already done — Sprint 5)
- The leaked key stays in history but is no longer useful (assuming rotate)
- Safer for any collaborators

### 4. Verify push protection passes

- [ ] `git push origin main`
- [ ] If still blocked, the leaked key is still in some commit on origin
- [ ] Check with: `git log --all --source --oneline -- service-account-key.json`

### 5. Document the incident

- [ ] Add entry to SECURITY_REVIEW.md §4 "Incident log"
- [ ] Note: time of detection, time of rotation, key fingerprint

## Local cleanup already done

- [x] `git rm --cached service-account-key.json aura-social-vn-e7a147284c33.json`
- [x] Local working tree cleaned (data/* deleted)
- [x] `.gitignore` updated to prevent future leaks
- [x] Commit `0e32307` (Sprint 4) is clean
- [x] `git pull --rebase` succeeded

## What remains on remote

- Commit `748a8e1` on `origin/main` still contains `service-account-key.json`
- This commit is older than our current HEAD
- GitHub Push Protection will block any push until resolved

## After incident

- Consider enabling `pre-commit` hook with `detect-secrets` to catch
  future leaks before commit
- Consider rotating all Service Account keys periodically (90 days)
- Add `SECURITY_REVIEW.md` mention of this incident as a lesson learned