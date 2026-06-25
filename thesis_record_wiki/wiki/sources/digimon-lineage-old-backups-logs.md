---
type: Source
title: Digimon Lineage Old Backups Logs
description: Old-backups log slice with a 44,622-line super_digimon log, Neo4j/API failures, successful pipeline traces, and credential-leak caveat.
tags: [source, old-backups, logs, observability, security, credentials]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/outputs/logs/
confidence: high
---

# Summary

`archive/old_backups_2025_08/outputs/logs/` contains two files: a 9,061,113-byte `super_digimon.log` and an empty `super_digimon.rotating.log`. The aggregate SHA-256 for the directory is `fe0c3d19aec8b76afc027042026f5b164c8d6a9f0af28986431c269c264e6269`. [1][2]

The non-empty log has 44,622 lines and spans `2025-07-24 19:14:43` through `2025-08-01 20:09:42`. It is useful operational evidence because it records initialization, schema loading, tool registration, pipeline execution, Neo4j connection failures, API authentication failures, deprecation warnings, and some successful natural-language pipeline traces. [1]

# Security Caveat

The log contains credential-looking API request URLs, including Google API key query parameters. During this ingest, those values were not copied into the wiki. Treat any keys exposed in this preserved log as compromised before public sharing, export, or publication. [1]

The log also records OpenAI authentication failures involving test-looking or redacted keys. Those lines are useful as configuration evidence, but they should not be quoted verbatim in public-facing summaries. [1]

# Log Shape

Parsed log-level counts:

| Level | Count |
| --- | ---: |
| INFO | 26,476 |
| WARNING | 6,882 |
| ERROR | 10,620 |

Top modules by parsed log count:

| Module | Count |
| --- | ---: |
| `src.tools.phase1.t27_relationship_extractor_unified` | 7,785 |
| `src.core.memory_manager` | 6,113 |
| `src.core.quality_service` | 5,934 |
| `src.core.provenance_service` | 1,665 |
| `src.core.schema_manager` | 1,560 |
| `src.mcp.tool_registry` | 1,281 |
| `src.execution.mcp_executor` | 1,238 |

# Notable Patterns

Pattern counts from the log:

| Pattern | Count |
| --- | ---: |
| successful tool registration lines | 1,196 |
| deprecated provenance `tool_id` parameter warnings | 1,452 |
| Neo4j password-missing warnings | 365 |
| Neo4j unsupported-authentication-token warnings | 42 |
| successful pipeline completion lines | 110 |
| successful question-processing lines | 18 |
| Google API request URLs with key query parameters | 11 |
| OpenAI incorrect-key authentication warnings | 6 |

# Interpretation

This log strengthens the operational history in several directions:

- It confirms repeated tool registration and pipeline execution activity.
- It preserves successful natural-language interface and MCP executor traces near the tail of the log.
- It documents persistent Neo4j configuration/authentication problems across multiple dates.
- It documents provenance API drift through repeated deprecation warnings.
- It introduces a concrete public-export risk because secrets or secret-like query parameters appear in raw logs.

Use this page with [Old Backups Output Reports](/wiki/sources/digimon-lineage-old-backups-output-reports.md), [Old Backups Error Reports](/wiki/sources/digimon-lineage-old-backups-error-reports.md), and [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md).

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/outputs/logs/super_digimon.log`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/outputs/logs/super_digimon.rotating.log`
