---
type: SourceSummary
title: Digimon Lineage Generated Outputs 2025 08
description: Small generated-output archive containing repomix bundles, performance/SLA JSON artifacts, provenance and reasoning-trace SQLite databases, and a real-vector proof artifact.
tags: [source, digimon-lineage, archive, generated-outputs, evidence, provenance, reasoning-traces, vectors]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/
confidence: high
---

# Summary

`archive/generated_outputs_2025_08/` is a 10-file generated-output archive totaling about 1.9 MB. Its aggregate content-manifest hash is `eadcace777ff9fc19f26c2860fc87fe0358bb4283598981ab6f4550613aa4ceb`. [1]

The archive contains context bundles, performance/SLA JSON, two SQLite databases, and a small proof artifact showing real vector generation. It is more evidence-like than many planning archives, but most files are still supporting artifacts rather than full validation reports. [1]

# Inventory

| File | Bytes | Role |
| --- | ---: | --- |
| `clean-architecture-bundle.xml` | 914,285 | Large repomix/context bundle for clean architecture review. [1] |
| `repomix-phase21-analytics.xml` | 147,856 | Repomix bundle for Phase 2.1 analytics context. [1] |
| `repomix-phase21-real.xml` | 70,631 | Repomix bundle for real Phase 2.1 context. [1] |
| `repomix-t57.xml` | 93,873 | Repomix bundle for T57-related context. [1] |
| `test-repomix.xml` | 48,224 | Test repomix/context bundle. [1] |
| `provenance.db` | 45,056 | SQLite provenance database with `operations` and `lineage` tables. [2] |
| `reasoning_traces.db` | 561,152 | SQLite reasoning-trace database with `reasoning_traces`, `reasoning_steps`, and `query_metrics` tables. [2] |
| `performance_data.json` | 1,491 | Performance baseline JSON for fast/medium/variable/slow demo operations. [3] |
| `sla_config.json` | 1,644 | SLA threshold config and check/violation stats. [4] |
| `real_vectors_proof.json` | 838 | Small vector-generation proof using `text-embedding-3-small`. [5] |

# Structured Artifacts

`performance_data.json` records four operation baselines, each with p50/p75/p95/p99/mean/std-dev and 50 samples. It also records 393 total operations, 129 degraded operations, and 29 baseline updates. [3]

`sla_config.json` defines thresholds for tool execution, database query, API request, document processing, pipeline execution, and a demo operation. It records five total checks, one total violation, and zero critical violations. [4]

`real_vectors_proof.json` records a 2025-08-06 vector proof with 5 graph nodes, 3 graph edges, a 5-row table, `text-embedding-3-small` vectors of dimension 1536, generation time of about 7.19 seconds, and sample similarities among company-name entities. [5]

# SQLite Databases

`provenance.db` contains two substantive tables: `operations` with 19 rows and `lineage` with 1 row. Its schema tracks operation IDs, tool IDs, operation type, inputs, parameters, outputs, success, errors, metadata, timestamps, and duration. [2]

`reasoning_traces.db` contains `reasoning_traces` with 11 rows, `reasoning_steps` with 50 rows, and an empty `query_metrics` table. Its schema tracks trace IDs, operation metadata, root steps, step counts, confidence scores, decision levels, reasoning types, options considered, decisions, and final outputs. [2]

The database inspection was schema/metadata-only; the wiki does not copy full trace payloads.

# Interpretation

This slice preserves concrete generated artifacts for performance tracking, SLA configuration, provenance recording, reasoning trace recording, and real-vector generation. It is especially useful as a pointer to database schemas and artifact existence.

However, the repomix XML files are context bundles, not independent proof of behavior. The JSON and SQLite files are stronger evidence that some runtime-oriented instrumentation or demos existed, but they still need code-path and test-context verification before being cited as production capability.

# Credential Scan

A targeted scan of this generated-output archive found no literal OpenAI or Google API keys. [1]

# Relationship To Wiki

- [Digimon Lineage Archive Coverage Audit 2026-06-25](digimon-lineage-archive-coverage-audit-2026-06-25.md): queue-control page that identified this archive area.
- [Digimon Lineage Generated Reports](digimon-lineage-generated-reports.md): broader generated-report corpus.
- [Digimon Lineage Current Evidence Archive](digimon-lineage-current-evidence-archive.md): related evidence/current archive.
- [Digimon Lineage Gemini Review Tool Archive](digimon-lineage-gemini-review-tool-archive.md): related repomix/Gemini validation context.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): relevant because bundles and generated artifacts have different evidence levels.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): relevant guardrail for not treating archived generated artifacts as current runtime proof.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/provenance.db` and `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/reasoning_traces.db`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/performance_data.json`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/sla_config.json`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/generated_outputs_2025_08/real_vectors_proof.json`
