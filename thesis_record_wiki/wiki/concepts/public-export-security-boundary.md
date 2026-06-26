---
type: Concept
title: Public Export Security Boundary
description: Security boundary for sharing or exporting the preserved thesis/KGAS archive without modifying the raw record.
tags: [concept, security, public-export, credentials, preservation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../sources/digimon-lineage-old-backups-logs.md
  - ../sources/digimon-lineage-old-backups-backup-archives.md
  - ../sources/lit-review-root-files.md
  - ../sources/digimon-lineage-archived-uncertainty-datasets.md
confidence: high
---

# Summary

The preserved thesis/KGAS record should be treated as an internal source of truth, not a public-ready export bundle. It intentionally preserves logs, environment files, backup tarballs, old configs, generated outputs, and sensitive research datasets so the work history remains recoverable.

That preservation choice creates a security boundary: public sharing requires a separate reviewed derivative export. Do not edit, delete, or rewrite the raw archive to sanitize it in place.

The proposal's HSPC/data-governance boundary does not remove this export requirement. Even when proposal materials classify the dissertation design as public/de-identified and not human subjects research, raw archive export still needs a separate security/privacy review. See [HSPC Data Governance Boundary](/wiki/concepts/hspc-data-governance-boundary.md).

# Boundary

Before any public sharing, archive export, publication supplement, or handoff to a third party:

1. Treat credential-looking values in preserved logs, `.env` files, tarballs, and historical configs as compromised.
2. Rotate or revoke any credential that could still be live.
3. Build a separate export candidate rather than modifying `archive_full_record/`.
4. Scan the export candidate for secrets and sensitive identifiers.
5. Exclude unreviewed backup tarballs and raw logs unless they have a documented redaction plan.
6. Keep provenance: record what was excluded, what was transformed, and which internal source pages or hashes the export derives from.

# Known Risk Sources

- Old-backups logs include credential-looking API request URLs. The wiki records the risk but does not reproduce the values. [1]
- Old-backups tarballs were compressed but unencrypted and include config paths such as `.env`. They were listed during ingest, not extracted or modified. [2]
- The lit-review root contains a preserved `.env` file whose value is intentionally not reproduced in the wiki. [3]
- Archived uncertainty datasets include identifiers, handles, tweet IDs, timestamps, full tweet text, and psychological scores; they require an ethics/privacy review before export. [4]

# Operating Rule

The raw archive is for internal preservation and evidence recovery. A public bundle is a derived artifact with its own manifest, scans, exclusions, and rationale.

# Links

- [Digimon Lineage Old Backups Logs](/wiki/sources/digimon-lineage-old-backups-logs.md)
- [Digimon Lineage Old Backups Backup Archives](/wiki/sources/digimon-lineage-old-backups-backup-archives.md)
- [Lit Review Root Files](/wiki/sources/lit-review-root-files.md)
- [Digimon Lineage Archived Uncertainty Datasets](/wiki/sources/digimon-lineage-archived-uncertainty-datasets.md)
- [Archived Uncertainty Dataset Access Plan 2026 06 26](/wiki/sources/archived-uncertainty-dataset-access-plan-2026-06-26.md)
- [KGAS Evolution Checkpoint 2026-06-25](/wiki/concepts/kgas-evolution-checkpoint-2026-06-25.md)
- [HSPC Data Governance Boundary](/wiki/concepts/hspc-data-governance-boundary.md)

# Citations

[1] `../sources/digimon-lineage-old-backups-logs.md`  
[2] `../sources/digimon-lineage-old-backups-backup-archives.md`  
[3] `../sources/lit-review-root-files.md`  
[4] `../sources/digimon-lineage-archived-uncertainty-datasets.md`
