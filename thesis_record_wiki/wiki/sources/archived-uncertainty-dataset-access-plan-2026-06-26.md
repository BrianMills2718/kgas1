---
type: SourceSummary
title: Archived Uncertainty Dataset Access Plan 2026 06 26
description: Privacy-aware access plan for the archived Twitter-like psychological uncertainty datasets, defaulting to manifest-level use and requiring explicit review before raw reads, exports, or reruns.
tags: [source, privacy, datasets, uncertainty, access-plan, export-boundary]
created: 2026-06-26
updated: 2026-06-26
sources:
  - /wiki/sources/digimon-lineage-archived-uncertainty-datasets.md
  - /wiki/concepts/public-export-security-boundary.md
  - ../../docs/public_export/EXPORT_REVIEW_2026-06-26.md
confidence: high
---

# Summary

The archived uncertainty datasets are valuable internal thesis evidence and should be preserved, but they are not safe default inputs for casual reruns, public export, or broad agent search. The default access mode is manifest-level summary only. [1][2]

The raw files include Twitter-like identifiers, screen names, tweet IDs, timestamps, full tweet text, public handles/mentions, links, and psychological survey/ground-truth scores in the same records. Even if some records are synthetic, replicated, or transformed, the combined structure is sensitive research data. [1]

# Default Access Policy

| Use case | Default decision | Required before escalation |
| --- | --- | --- |
| Wiki/source summaries | Allowed at manifest level. | Do not quote tweet text or reproduce identifiers. |
| Hash/file inventory | Allowed. | Record file path, size, SHA-256, and structural fields only. |
| Aggregate statistics | Allowed if no raw examples are emitted. | Verify code does not print text, handles, IDs, or per-user rows. |
| Local model evaluation | Deferred. | Write a rerun plan, redaction plan, output path, and review criteria first. |
| Export candidate | Excluded by default. | Separate Brian approval plus ethics/privacy/security review. |
| Public repository | Prohibited by default. | Only derived, reviewed, non-identifying summaries may be considered. |
| Raw dataset mutation | Prohibited. | Raw archives remain immutable preservation evidence. |

# Safe Agent Procedure

For future agents:

1. Start from [Digimon Lineage Archived Uncertainty Datasets](/wiki/sources/digimon-lineage-archived-uncertainty-datasets.md), not the raw JSON files.
2. Use existing hashes, sizes, and structural counts before opening raw data.
3. If raw access is necessary, write a plan first that states the exact files, fields read, derived outputs, and redaction rules.
4. Run any raw-data script so stdout/stderr cannot print tweet text, handles, IDs, or per-user psychological rows.
5. Write derived outputs under an ignored/local path until reviewed.
6. Commit only manifest summaries or reviewed aggregate results.
7. Do not include these raw files in any export candidate unless Brian explicitly approves that specific inclusion.

# Redaction Requirements For Derived Outputs

Any derived output must remove or aggregate away:

- `twitter_id`
- `twitter_screen_name`
- tweet IDs
- timestamps at tweet/user level
- full tweet text
- handles/mentions
- URLs
- email-like strings
- per-user psychological score rows
- original source IDs

Allowed derived outputs by default:

- file-level hashes and byte sizes;
- row/user/tweet counts;
- field inventories;
- count inconsistency notes;
- aggregate distributions with bins large enough to avoid row reconstruction;
- high-level validation caveats.

# Rerun Gate

Before using the datasets for uncertainty or psychological-trait evaluation, require:

1. **Purpose:** the exact research/verification question.
2. **Input minimization:** which fields are needed and why.
3. **Output minimization:** what will be written and what will be suppressed.
4. **Storage:** local ignored path for intermediate outputs.
5. **Review:** human check before commit/export.
6. **Claim level:** whether the result is a smoke, exploratory analysis, historical reproduction, or validation evidence.

# Relationship To Export Boundary

The docs-only export candidate deliberately excluded high-risk raw file classes and remained unapproved for publication. These datasets are higher risk than the docs-only candidate because they include raw text plus identifiers and psychological scores. [2][3]

The public-export rule remains: derive a separate reviewed artifact; never sanitize the preserved raw archive in place. [2]

# Relationship To Wiki

- [Digimon Lineage Archived Uncertainty Datasets](/wiki/sources/digimon-lineage-archived-uncertainty-datasets.md): structural inventory and privacy review.
- [Public Export Security Boundary](/wiki/concepts/public-export-security-boundary.md): broader export rule for raw thesis/KGAS archives.
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md): uncertainty lineage map that treats the datasets as sensitive corpus/provenance evidence.

# Citations

[1] `/wiki/sources/digimon-lineage-archived-uncertainty-datasets.md`  
[2] `/wiki/concepts/public-export-security-boundary.md`  
[3] `../../docs/public_export/EXPORT_REVIEW_2026-06-26.md`
