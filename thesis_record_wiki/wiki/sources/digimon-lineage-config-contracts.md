---
type: Source
title: Digimon Lineage Config And Contracts
description: Inventory and interpretation of the large-lineage core config files and contract system, excluding config/schemas and config/legacy_tools for separate slices.
tags: [source, digimon-lineage, config, contracts, tool-contracts, theory-aware, monitoring, orchestration]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/contracts/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/contracts/README.md
confidence: medium
---

# Summary

This slice covers the large lineage's core configuration and contract directories:

- `config/`: 112 files, 1,108,189 bytes, aggregate hash `c37d76322e66e9266488061e6fcbd3897560d2c928f25fe104e1c6b8e7d43f06` [1]
- `contracts/`: 20 files, 62,727 bytes, aggregate hash `fdc237a27607056584251b95b5a86aac1b1f677299ba37854052bfebccd1fe17` [2]

This page intentionally focuses on core config/orchestration/monitoring and the top-level `contracts/` system. The larger `config/schemas/` theory-schema bundle and `config/legacy_tools/` subtree should be handled in separate slices. [1]

# Config Directory Shape

The config README describes this as the centralized configuration directory for KGAS, including main config, environment-specific configs, monitoring, schema definitions, contracts, and examples. [3]

Important inventory:

| Config area | Files | Bytes | Aggregate/hash | Role |
|---|---:|---:|---|---|
| root config files | 12 | 29,031 | individual hashes in manifest | `config.yaml`, `base/default/development/production/testing.yaml`, `services.yaml`, `config_loader.py`, `pyproject.toml`, docs/instructions |
| `build/` | 5 | 4,867 | `c6901e6b3ee9a35870a327bf7bc0204984c5519b714ef21a795db5a295baa65a` | build/test config: Makefile, pytest, tox, docker-compose test |
| `contracts/` | 1 | 1,004 | `449311e23e9d39ef45b5851e276f38a1a4b42c6ed0a1d1b560ead574da88bb2b` | Phase1-to-Phase2 adapter config |
| `environments/` | 4 | 28,729 | `22670dbcc3e4b0544977b19f9757a50c92ec1d86a40931c635b6ec8b7dd7b6f8` | Docker compose environment files and local instructions |
| `examples/` | 3 | 15,247 | `89dd3e110685d95de2deb4278727cf987161c35285fe8aeaf06a053127c8d9ff` | docs, Gemini, and verification review examples |
| `monitoring/` | 6 | 43,128 | `4fe589af46a1726d9d0235ed77197eb03286cc302e0bdb7402f348c922400057` | Prometheus/Grafana/integration-monitoring configs |
| `orchestration/` | 2 | 11,893 | `2f8aa990369058bde7d51aa575752007dd94e8a98632610fb74ed05cb4ef4b3a` | simple sequential and parallel workflow configs |
| `templates/` | 2 | 11,094 | `d0a7734075f38bee0c7dfe1cdc961d9c7aff450e003ae7fc4ab6a8d5d5b3486e` | CLAUDE template variants |

Security caveat: `config/credential.salt` is preserved as a 16-byte salt file. The wiki records its existence and hash in the manifest but does not reproduce its value. [1]

# Config Loader And Runtime Assumptions

`config_loader.py` is intended to provide environment-specific loading, environment variable overrides, validation, and type-safe settings. The README lists environment-variable support for Neo4j, OpenAI, Anthropic, and log level, and tells users to store sensitive values in environment variables rather than committed config files. [3]

The orchestration default config names a simple sequential flow:

- document loading/chunking
- entity and relationship extraction
- graph construction with entity/edge building and PageRank
- graph query [1]

This reinforces the recurring [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md) split: config files describe intended wiring, while runtime pages separately check which imports and services actually work in the current environment.

# Contract System

The `contracts/README.md` frames the contract system as active work for theory schema and contract integration. Its goals include version checking, theory schema validation, immutable theory-aware phase interfaces, contract validation, and CI integration. [4]

The contract system has:

- `phase_interfaces/`: immutable phase request/result interfaces, `TheoryConfig`, `TheorySchema`, `TheoryAwareGraphRAGPhase`, and a theory-aware phase registry [2]
- `schemas/tool_contract_schema.yaml`: schema for validating tool contract files [2]
- `tools/`: nine YAML tool contracts [2]
- `validation/theory_validator.py`: validates entities and relationships against theory schema metadata and concept libraries [2]

The README still describes a broader 121-tool ecosystem, while the preserved `contracts/tools/` slice contains nine concrete tool contracts. This should be read as a partial contract-system artifact, not complete coverage of every planned tool. [4]

# Preserved Tool Contracts

| Tool id | Name | Category | Depends on |
|---|---|---|---|
| `T01_PDF_LOADER` | PDF Document Loader | document processing | none |
| `T15A_TEXT_CHUNKER` | Text Chunker | document processing | `T01_PDF_LOADER` |
| `T23A_SPACY_NER` | spaCy Named Entity Recognition | entity extraction | `T15A_TEXT_CHUNKER` |
| `T27_RELATIONSHIP_EXTRACTOR` | Relationship Extractor | entity extraction | `T15A_TEXT_CHUNKER`, `T23A_SPACY_NER` |
| `T31_ENTITY_BUILDER` | Entity Builder | graph building | `T23A_SPACY_NER` |
| `T34_EDGE_BUILDER` | Edge Builder | graph building | `T27_RELATIONSHIP_EXTRACTOR` |
| `T49_MULTI_HOP_QUERY` | Multi-hop Graph Query | analysis | `T31_ENTITY_BUILDER`, `T34_EDGE_BUILDER` |
| `T68_PAGE_RANK` | PageRank Calculator | analysis | `T31_ENTITY_BUILDER`, `T34_EDGE_BUILDER` |
| `T85_TWITTER_EXPLORER` | Twitter API Explorer | social media analysis | none |

These contracts preserve a canonical graph pipeline from PDF loading through chunking, NER, relationship extraction, graph construction, graph analytics/querying, plus a social-media exploratory tool. [2]

# Interpretation

This slice is a bridge between architecture intent and executable enforcement:

- config files try to make services, monitoring, and orchestration explicit
- the contract directory tries to define machine-checkable tool/phase interfaces
- the phase interfaces add theory-aware processing and validation metadata to pipeline boundaries
- the tool contracts preserve a concrete dependency graph for a subset of the KGAS tool ecosystem

The main caveat is implementation status. Config and contract files show intended structure and partial enforcement artifacts. They do not prove the full main system was runnable, that every planned tool had a contract, or that theory validation was complete.

# Links

- [Contract First Migration](/wiki/concepts/contract-first-migration.md)
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md)
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/contracts/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/config/README.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/contracts/README.md`
