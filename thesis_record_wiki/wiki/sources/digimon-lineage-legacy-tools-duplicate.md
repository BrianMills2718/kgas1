---
type: Source
title: Digimon Lineage Legacy Tools Duplicate
description: Inventory of the byte-identical tools/ and config/legacy_tools/ operational tool-script trees in the large Digimons lineage.
tags: [source, digimon-lineage, tools, legacy-tools, duplicate, scripts, demos, verification]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tools/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/legacy_tools/
confidence: high
---

# Summary

The large lineage preserves two byte-identical copies of the same 49-file operational tooling tree:

- `tools/`: 49 files, 561,112 bytes, aggregate hash `a464f676a62f09fb3d3eaf261896248bea06f086816c49724989fb5da2c51054` [1]
- `config/legacy_tools/`: 49 files, 561,112 bytes, same aggregate hash `a464f676a62f09fb3d3eaf261896248bea06f086816c49724989fb5da2c51054` [2]

A relative-path/hash comparison found zero mismatches. These should be treated as duplicate preserved operational surfaces, not two independent tool corpora.

# Directory Shape

| Subtree | Files | Bytes | Aggregate hash | Contents |
|---|---:|---:|---|---|
| `demos/` | 2 | 18,785 | `9b108b46cbfbed114ab4aa83c717248d10290821ab859c485b2d2d855032b945` | tool registry demo and full system demo |
| `examples/` | 11 | 320,322 | `35b41c7367e8222cd3c09092be9f1f7a3de5e0dd4b643508800f1f150fe2f139` | climate/test text/PDF examples, minimal working example, install verification, stats runner |
| `scripts/` | 36 | 222,005 | `1c1e9e8bdc7b7db61dc9b0cea253af72eab3cdb74d20e76a83e8f7955e5f1c67` | audit, docs, validation, Neo4j, CLI/server, service startup, UI startup, migration, contract validation, and verification scripts |

# Representative Tooling

The duplicated tree preserves several operational themes:

- demo surfaces: `demo_tool_registry.py`, `full_system_demo.py`, `minimal_working_example.py`
- graph/service scripts: `ensure_neo4j.py`, `init_neo4j_schema.py`, `verify_database.py`, graph-query demo scripts, PageRank demo
- interface and contract tooling: `validate_contracts.py`, `migrate_to_pipeline_orchestrator.py`, `fix_all_sys_path.py`
- documentation governance: `check_doc_drift.py`, `verify_all_documentation_claims.sh`, `concatenate_evergreen.py`, `concatenate_roadmap.py`
- runtime launchers: `graphrag_cli.py`, `simple_fastmcp_server.py`, `start_services.sh`, `start_ui.sh`
- verification: `comprehensive_validation.py`, `verify_system.py`, `verify_implementation.py`, Gemini validation script, Phase 2 evidence quick verification

`capability_numbered_list.txt` enumerates a larger historical capability surface, including classes/functions around PDF loading, Neo4j fallback/base tools, PageRank, fusion tools, MCP tools, vertical-slice workflows, and UI phase adapters. [1][2]

# Interpretation

This duplicate tree is useful provenance because it preserves an intermediate operational layer between architecture docs and source code:

- it shows how agents/humans were expected to exercise the system
- it preserves demos and verification scripts that can explain historical claims
- it shows migration pressure toward `PipelineOrchestrator`
- it records documentation-claim checking as an operational concern

The duplicate location is also a cleanup clue. Future cleanup should avoid indexing both copies as separate evidence and should choose a canonical source before any archive/move operation. For now both raw locations are preserved unchanged.

# Links

- [Digimon Lineage Config And Contracts](/wiki/sources/digimon-lineage-config-contracts.md)
- [Digimon Lineage Ops Scaffolding](/wiki/sources/digimon-lineage-ops-scaffolding.md)
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md)
- [Contract First Migration](/wiki/concepts/contract-first-migration.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tools/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/legacy_tools/`
