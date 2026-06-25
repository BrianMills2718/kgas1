---
type: SourceSummary
title: Digimon Lineage Temp Debug Files
description: Low-priority temp debug archive containing database inspection scripts, relationship/chunk debugging, structured-output migration tests, Phase 2.1 claim validators, reliability certification runners, and one facade-POC session transcript fragment.
tags: [source, digimon-lineage, archive, temp-debug, validation, structured-output, reliability, neo4j, sqlite, relationship-extraction, facade-poc]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/
confidence: high
---

# Summary

`archive/temp_debug_files/` is a 32-file, 169,254-byte archive of KGAS temporary debug and validation files. Its aggregate content-manifest hash is `cc0dcbe3f36b2267c5a6b31e16a4e274c6e72dc3f969f34365bab2a489beb098`. [1]

This is the lowest-value top-level archive queue from the current audit, but it still contains useful provenance: database inspection scripts, relationship-extraction debugging, structured-output migration tests, Phase 2.1 claim validators, reliability certification runners, and one session transcript fragment about a facade proof of concept. [1]

# Inventory Themes

| Theme | Representative Files | Role |
| --- | --- | --- |
| Database inspection | `check_neo4j_data.py`, `check_neo4j_relationships.py`, `check_sqlite_data.py`, `demo_both_databases.py`, `demo_real_neo4j_data.py` | Local Neo4j/SQLite inspection and demo scripts. [1] |
| Relationship/debug probes | `debug_chunk_refs.py`, `debug_relationships.py`, `test_pipeline_fix.py` | Debugging chunk refs, relationship extraction, and pipeline fixes. [2] |
| Structured output migration | `test_phase1_structured_output_migration.py`, `test_phase2_structured_reasoning.py` | Token-limit, feature-flag, structured LLM service, and reasoning-engine migration tests. [3] |
| Claim validation | `validate_claim*.py`, `run_claim1_validation.py`, `run_final_validation.py`, `validate_phase21_claims.py`, `validate_phase21_detailed.py` | Scripted checks for implementation claims and Phase 2.1 real-implementation claims. [4] |
| Reliability certification | `run_reliability_certification.py`, `validate_reliability.py`, `demonstrate_reliability_fixes.py` | Reliability certification/demo scripts for real database testing and monitoring. [5] |
| Graph/tool tests | `test_fixed_graph_tools.py`, `test_neo4j_graph_tools.py`, `test_connection_pool_demo.py`, `test_mcp_integration.py` | Local graph/tool smoke tests and connection-pool checks. [1] |
| Session transcript fragment | `temp1.txt` | Captured agent/session text about KGAS justification and facade POC debugging. [6] |

# Structured Output Migration

`test_phase1_structured_output_migration.py` tests the early structured-output migration plan: increased token limits, feature flags, structured LLM service availability, fail-fast/logging settings, and reasoning-engine integration. It expects LLM reasoning structured output to be enabled while entity extraction remains disabled at that phase. [3]

This connects directly to the old-docs structured-output migration plan and later ecosystem structured-output rules.

# Claim Validation And Reliability

`validate_phase21_claims.py` uses AST/file checks to evaluate whether named classes and imports exist for Phase 2.1 claims such as `RealEmbeddingService`, `RealLLMService`, `AdvancedScoring`, `RealPercentileRanker`, `TheoryKnowledgeBase`, and updated cross-modal/linker/synthesizer components. It distinguishes fully, partially, and not resolved claims. [4]

`run_reliability_certification.py` is a long-running reliability certification wrapper targeting "10/10 bulletproof reliability" with real Neo4j and SQLite instances, failure recovery scenarios, load testing, and optional 24-hour stability testing. This is a certification runner, not evidence that certification passed. [5]

# Facade POC Transcript Fragment

`temp1.txt` starts with an argument that KGAS differs from normal GraphRAG because it targets automated theory-driven computational social science rather than only retrieval and question-answering. It then contains a captured agent-session fragment showing a facade POC being edited and rerun, including relationship-extraction regex fixes for multiline text. [6]

This is messy but historically useful because it preserves live debugging context around the facade proof-of-concept, relationship extraction, and the "KGAS is not normal GraphRAG" framing.

# Caveats

This directory heavily overlaps with already represented archives: scripts archive, demos/examples, analysis validation, and generated outputs. Do not over-weight it as independent evidence.

Many scripts are local-environment probes and would need current-path, dependency, database, and credential review before rerun. Several are validators for claims, but the validator scripts alone are not the same thing as preserved validation results.

The targeted credential scan found no literal OpenAI or Google API keys in this archive. [1]

# Interpretation

This page closes the top-level archive coverage queue from the audit. The durable takeaways are that KGAS debugging focused on relationship extraction, structured-output migration, claim validation, database state inspection, and reliability certification. `temp1.txt` also preserves the project-framing claim that KGAS was meant as theory-driven computational social science infrastructure rather than conventional GraphRAG.

# Relationship To Wiki

- [Digimon Lineage Scripts Archive 2025 08](digimon-lineage-scripts-archive-2025-08.md): larger overlapping debug/demo/fix/test script corpus.
- [Digimon Lineage Demos Examples 2025 08](digimon-lineage-demos-examples-2025-08.md): neighboring demo/example archive.
- [Digimon Lineage Analysis Validation 2025 08](digimon-lineage-analysis-validation-2025-08.md): validation configs and claim checks.
- [Digimon Lineage Old Docs 2025 08](digimon-lineage-old-docs-2025-08.md): structured-output migration planning and old docs.
- [Relationship Extraction Bottleneck](../concepts/relationship-extraction-bottleneck.md): recurring relationship-extraction failure/repair theme.
- [GraphRAG Upstream Lineage](../concepts/graphrag-upstream-lineage.md): context for the "not normal GraphRAG" distinction.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): needed when interpreting claim validators versus proof.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/debug_relationships.py`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/test_phase1_structured_output_migration.py`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/validate_phase21_claims.py`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/run_reliability_certification.py`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/temp1.txt`
