# Blocked Gates Decision Brief

Date: 2026-06-26

## Purpose

Plan #1 safe local runtime work is complete. The remaining gates are not engineering ambiguity; they require Brian's direction on privacy, budget, or destructive-state risk. This brief records the decisions needed before any agent proceeds.

## Current Safe State

- Raw PhD/KGAS history remains preserved in `archive_full_record/`; do not mutate it in place.
- `/api/analyze` is live-proven for `.txt`, `.pdf`, `.md`, and `.docx`.
- `/api/batch/analyze` is live-proven for a TXT batch through the same single-document path.
- Legacy `.doc` remains explicit 501.
- `/api/recommend` is contract-wired but not live LLM-proven.
- Neo4j source-scoped cleanup has a dry-run-first helper, but no cleanup execution has been approved.

## Gate 1 - Public/Export Bundle

Decision needed: choose target audience and privacy posture.

Safe default: do not publish and do not create a shareable export from raw archive material.

If approved:

1. Create a derived export outside the raw preserved archive.
2. Include an export manifest listing included/excluded material.
3. Scan for `.env`, keys, tokens, logs, local paths, backups, and sensitive datasets.
4. Prefer private GitHub unless Brian explicitly approves public release.
5. Keep raw `archive_full_record/` unchanged.

## Gate 2 - Live LLM Recommendation Test

Decision needed: choose provider/model and budget cap.

Safe default: leave live recommendation proof deferred.

If approved:

1. Configure the selector with explicit credentials and a tiny budget.
2. Run one minimal `/api/recommend` live smoke.
3. Record model, cost, result, and any failure mode.
4. Do not expand into broad recommendation evals until the single smoke is understood.

## Gate 3 - Neo4j Cleanup Execution

Decision needed: identify exact `source_ref` values that are safe to remove.

Safe default: dry-run only; no deletion.

Existing safety checkpoint:

- Neo4j dump: `~/archive/phd_thesis_work/neo4j/20260626-075357/neo4j.dump`
- SHA-256: `553d57c74eb1ac3619755e3af41be81ebc1dd00fd52e2005b21f7d5cbbb630dc`

If approved:

1. Run dry-run for the exact source ref.
2. Review node and relationship candidate counts.
3. Execute only the exact source-scoped cleanup command.
4. Re-run source-scoped smoke tests afterward.

Do not run broad graph cleanup.

## Gate 4 - Legacy `.doc` Support

Decision needed: decide whether old binary Word format matters enough to support.

Safe default: keep `.doc` as explicit 501 because `.docx` is already proven.

If approved:

1. Identify a real legacy `.doc` fixture from non-sensitive material.
2. Choose a loader path that can run locally.
3. Add focused API coverage.
4. Run a live smoke only after the loader path is proven.

## Recommended Order

1. Public/export decision, if sharing or GitHub publication is near-term.
2. Live LLM recommendation, only if the recommendation endpoint matters for current use.
3. Neo4j cleanup execution, only if graph clutter is materially blocking work.
4. Legacy `.doc`, likely defer unless a specific historical document requires it.

## Stop Line

Agents should not proceed past any gate without Brian's explicit approval for that gate. Safe follow-up work is limited to documentation, read-only review, and non-destructive dry-runs.
