---
type: SourceSummary
title: Digimon Lineage Old Claude Md Versions
description: Archive of historical CLAUDE.md variants showing KGAS agent-instruction evolution across contract-first migration, MVP setup, Phase 2.1 completion, evidence-first repair, and template iterations.
tags: [source, digimon-lineage, archive, claude-md, instructions, policy, agent-workflow]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_claude_md_versions/
confidence: high
---

# Summary

`archive/old_claude_md_versions/` is a seven-file, 140 KB archive of historical KGAS `CLAUDE.md` variants. Its aggregate content-manifest hash is `1de042339c7ac7753bc806c418e5a32a34a1b32a738f3f1652ffda5a76340f28`. [1]

These files preserve agent-instruction history, not current operating policy. They show how local KGAS instructions evolved through contract-first migration, MVP setup, Phase 2.1 completion, and evidence-first repair after Gemini validation. [1]

# Inventory

| File | Bytes | Role |
| --- | ---: | --- |
| `CLAUDE_OLD2.md` | 14,651 | Contract-first tool migration instructions from 2025-08-05. [2] |
| `CLAUDE_MVP_DAY1.md` | 12,496 | MVP Day 1 environment setup and basic document-to-entity pipeline guide. [3] |
| `CLAUDE_PHASE21_COMPLETION.md` | 5,800 | Phase 2.1 completion tasks for T59 scale-free analysis and T60 graph export. [4] |
| `CLAUDE_deferred.md` | 49,511 | Evidence-first Phase 5.3 implementation-fix instructions after Gemini validation. [5] |
| `CLAUDE_OLD.md` | 29,962 | Older broader KGAS instruction set. [1] |
| `claude_md_template_digimon_202507232000.md` | 7,490 | Earlier generated template. [1] |
| `claude_md_template_digimon_202507232000_improved.md` | 3,604 | Improved generated template. [1] |

# Contract-First Migration

`CLAUDE_OLD2.md` says KGAS was migrating from multiple competing tool interfaces to a single `KGASTool` contract defined in `src/core/tool_contract.py`. It explicitly states no backwards compatibility was needed because the project was a single-developer environment without production data. [2]

It instructs agents to fix a `ToolValidationResult` attribute mismatch, update the sequential orchestrator to use `ToolRequest`/`ToolResult`, update service APIs, and create evidence files for each fix. [2]

# MVP Day 1

`CLAUDE_MVP_DAY1.md` gives a day-one vertical-slice setup guide: start Neo4j with Docker, test SQLite, verify Python dependencies, and build a basic document-processing pipeline. It includes examples with local paths and default/demo credentials, so it should not be used as current setup guidance. [3]

# Phase 2.1 Completion

`CLAUDE_PHASE21_COMPLETION.md` defines implementation requirements for T59 Scale-Free Network Analysis and T60 Graph Export. It expects statistical power-law fitting, hub detection, preferential attachment analysis, multiple graph export formats, property preservation, large-graph streaming, and Gemini validation before Phase 2.1 is marked complete. [4]

# Evidence-First Repair Instructions

`CLAUDE_deferred.md` is the strongest policy-evolution artifact. It emphasizes evidence-first development, no fake completion claims, raw execution logs in `Evidence.md`, real functionality tests, and no placeholder/simulation logic. [5]

It also records Gemini findings: async migration issues, partial ConfidenceScore integration, heavy unit-test mocking, and academic pipeline integration problems. The prescribed repairs include replacing simulated async sleeps with real async operations, replacing placeholder tools, reducing mocks, building end-to-end academic pipeline tests, and validating with Gemini. [5]

# Interpretation

This archive shows a transition from local implementation instructions toward the stricter evidence and verification norms now present in the ecosystem policy. The core themes are contract-first interfaces, no-mocks testing, raw evidence, external validation, and current-status honesty.

Do not treat these files as current instructions. Their value is historical: they explain why later policies emphasize evidence discipline, contract boundaries, and avoiding duplicate or stale agent guidance.

# Credential Scan

A targeted scan of this old-CLAUDE archive found no literal OpenAI or Google API keys. [1]

# Relationship To Wiki

- [Digimon Lineage Root Cleanup 2025 08 29](digimon-lineage-root-cleanup-2025-08-29.md): related cleanup/policy lineage.
- [Digimon Lineage Tool Orchestration ADRs](digimon-lineage-tool-orchestration-adrs.md): related contract-first and tool-interface ADRs.
- [Contract-First Migration](../concepts/contract-first-migration.md): related migration concept.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): later durable version of evidence-first repair principles.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): related current-status honesty discipline.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_claude_md_versions/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_claude_md_versions/CLAUDE_OLD2.md`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_claude_md_versions/CLAUDE_MVP_DAY1.md`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_claude_md_versions/CLAUDE_PHASE21_COMPLETION.md`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_claude_md_versions/CLAUDE_deferred.md`
