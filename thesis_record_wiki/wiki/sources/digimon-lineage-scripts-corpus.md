---
type: Source
title: Digimon Lineage Scripts Corpus
description: Inventory and interpretation of the large-lineage scripts directory as an operational verification, analysis, demo, repair, and validation layer.
tags: [source, digimon-lineage, scripts, validation, verification, demos, repair, operations]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/scripts/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/scripts/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/scripts/CLAUDE.md
confidence: medium
---

# Summary

`scripts/` is a 127-file, 1,204,684-byte operational-code corpus with aggregate hash `6dfaff3fb7ff7050b0859c56a926c9d30c00d2bf02ab2b39df2048590dc6c2a8`. [1]

The README says this directory consolidates scripts that were previously scattered at repository root and organizes them by purpose: validation, analysis, demos, and testing. [2]

This is not a duplicate of [Digimon Lineage Legacy Tools Duplicate](/wiki/sources/digimon-lineage-legacy-tools-duplicate.md). The duplicated `tools/` tree has 49 files; this `scripts/` tree has 127 files and a broader validation/repair surface. [1]

# Directory Inventory

| Subtree | Files | Bytes | Aggregate hash | Role |
|---|---:|---:|---|---|
| root scripts | 83 | 776,330 | included in root aggregate | top-level demos, repairs, validation, tests, config/database utilities |
| `analysis/` | 16 | 132,094 | `2bb28e8beddc36308e217e904c4c2b2c2753ce46a02af09c328f528f46129b96` | architecture review, ADR extraction, Neo4j/SQLite checks, static visualization, Carter/Kunst analysis |
| `demo/` | 6 | 116,639 | `c5bffebaac4427bae9de41bd169506d01842f02e06c1d325d962e76fd9d56602` | database, agent, academic paper, and Kunst demos |
| `monitoring/` | 2 | 27,768 | `80a749d318f83e5cbc74401b089c4661699445fed5bc9605153452edbe5ac331` | monitoring dashboard/startup |
| `testing/` | 8 | 44,032 | `53ae5dce0134e1f4d20a48b71c4ae87afb0b085a781127b96d9efdead93e0490` | reliability fixes, MCP, Neo4j graph tools, relationship extraction, pipeline tests |
| `validation/` | 12 | 107,821 | `2dbe5ea14641993d92534c3b5e9ab8f94bf7a733630a6a2e84a9585485a1825a` | final/direct validation, MCP validation, reliability certification, ontology/tool inventory validation |

# Operational Themes

The root scripts are dominated by validation and repair names: 13 `test*`, 13 `validate*`, 5 `fix*`, 5 `simple*`, and 4 `verify*` files. [1]

Representative themes:

- **Current-state validation**: `validate_complete_architecture.py`, `validate_environment.py`, `validate_config.py`, `validate_nl_to_dag_complete.py`, `validate_reasoning_system.py`, `verify_claims.py`, `verify_dag_claims.py`, `verify_tool_success_rate.py`
- **Repair/migration pressure**: `fix_db_hardcoding.py`, `fix_llm_hardcoding.py`, `fix_tool_categories.py`, `fix_tool_ids.py`, `migrate_config_references.py`, `migrate_tool_interfaces.py`, `remove_fallbacks.py`
- **Graph/analytics demos**: `real_carter_pipeline.py`, `proper_vertical_slice_demo.py`, `complete_multimodal_demo.py`, `benchmark_graph_analytics.py`, `nl_to_dag_carter_demo.py`
- **LLM/structured-output checks**: `gemini_structured_demo.py`, `test_gemini_structured_output.py`, `test_llm_interface_integration.py`, `test_real_llm_reasoning_integration.py`
- **MCP/runtime surfaces**: validation and testing scripts for MCP tools, external architecture, and relationship extraction

# Verification Culture

Several scripts are explicitly about preventing inflated claims:

- `verify_claims.py` checks tool-contract integration claims such as confidence-score methods, absence of mock dependencies, tool ID standardization, and tool registration. [1]
- `remove_fallbacks.py` scans production code for fallback/mock patterns and reports violations. [1]
- `validate_environment.py` checks database, API keys, system config, security, monitoring, production config, file paths, and network config, then produces a production-readiness score. [1]
- `validate_complete_architecture.py` attempts to validate Phase 8.5 claims around a complete GraphRAG pipeline, external MCP architecture, performance/monitoring, evidence-based claims, and integration completeness. [1]
- `validation/run_reliability_certification.py` frames reliability certification as a long-running validation suite, optionally including a 24-hour test. [1]

These scripts connect directly to [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md) and [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): they show the project adding executable checks around claims, but the scripts themselves are not proof that those checks passed.

# Caveats

- Many scripts manipulate `sys.path` or assume repository-root execution; portability needs current runtime verification. [1][2]
- Some scripts are validators that generate reports; the reports/results need separate ingestion before citing pass/fail outcomes.
- Several scripts include production-readiness or certification language. Treat that as validation intent unless paired with preserved execution output.
- The scripts tree overlaps thematically with `tools/` and CI workflows, but it is larger and not byte-identical to the duplicated legacy-tools tree.

# Links

- [Digimon Lineage Legacy Tools Duplicate](/wiki/sources/digimon-lineage-legacy-tools-duplicate.md)
- [Digimon Lineage Ops Scaffolding](/wiki/sources/digimon-lineage-ops-scaffolding.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/scripts/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/scripts/README.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/scripts/CLAUDE.md`
