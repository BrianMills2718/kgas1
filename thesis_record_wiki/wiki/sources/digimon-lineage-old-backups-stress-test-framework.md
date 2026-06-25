---
type: SourceSummary
title: Digimon Lineage Old Backups Stress Test Framework
description: Support framework files for the old-backups stakeholder-theory stress test, including config, schemas, database integration, algorithms, theory specs, registry metadata, policy data, and tool capability YAMLs.
tags: [source, digimon-lineage, old-backups, stress-test, schema, theory, tools, database]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/config/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/schemas/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/theory/
confidence: high
---

# Summary

This slice covers the support files under `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/`: `config/`, `data/`, `database/`, `framework/`, `schemas/`, `scripts/`, `theory/`, and `tool_capabilities/`. The support-file aggregate hash is `aggregate-sha256:39f6d8fc33778f697a46e3f711b37475d8b95516047effc83d535dace7a6a39f`. [1]

These files show the stress test as a small framework package rather than only a report bundle. It includes Pydantic schemas, Neo4j setup code, schema registry code, tool capability metadata, theory JSON, custom algorithms, configuration, and mock/realistic input text. [1]

# Inventory

Support files by area:

- `config/default.yaml` - development config with empty API-key fields, Neo4j defaults, processing thresholds, backup/encryption flags, and theory validation settings. [2]
- `data/policy_documents/climate_policy_proposal.txt` - realistic policy input used by the stakeholder/resource-dependency stress test. [3]
- `database/neo4j_setup.py` - Neo4j manager for stakeholder analysis with schema setup, node/relationship creation, queries, and mock behavior when no connection exists. [4]
- `framework/schema_registry.py` - schema registration, semantic version metadata, checksums, compatibility matrix, and ecosystem validation. [5]
- `framework/tool_integration.py` - tool capability registration and type-compatibility/pipeline scaffolding. [6]
- `schemas/base_schemas.py`, `schemas/stakeholder_schemas.py`, `schemas/resource_dependency_schemas.py` - Pydantic data types for base objects, stakeholder theory, and resource-dependency calculations. [7]
- `scripts/salience_calculator.py` and `scripts/dependency_calculator.py` - custom theory algorithms with validation and test-case runners. [8]
- `theory/stakeholder_theory_v10.json`, `theory/resource_dependency_theory_v10.json`, `theory/mcl_mock.yaml`, and `theory/registry.json` - theory meta-schema artifacts, mock concept library, and registry/compatibility metadata. [9]
- `tool_capabilities/stakeholder_analyzer.yaml` and `tool_capabilities/dependency_analyzer.yaml` - declared tool inputs, outputs, theory compatibility, performance characteristics, and resource requirements. [10]

# Theory And Schema Evidence

The stakeholder theory JSON uses `theory_meta_schema_version` 10.0 and describes Stakeholder Theory around legitimacy, urgency, power, stakeholder salience, influence networks, prompts, data contracts, cross-modal mappings, validation tests, and an LLM validation setting. [9]

The resource-dependency JSON also uses meta-schema 10.0. It defines dependency strength in terms of resource criticality, scarcity, and substitute availability; includes dependency validation rules and custom-script test cases; and declares compatibility with stakeholder theory and institutional theory. [9]

The registry file is a useful caution. It registers `stakeholder_theory` and `resource_dependency_theory` version 1.0.0, but the compatibility matrix marks each direction as `incompatible` with "No integration points found." That sits uneasily beside the resource-dependency schema's own `compatible_with` declaration and should be treated as registry-state evidence, not a resolved theory-integration truth. [9]

The tool capability YAMLs preserve the intended typed pipeline: `stakeholder_analyzer` takes `TextChunk` and emits `StakeholderEntity`; `dependency_analyzer` takes `OrganizationEntity` and emits `DependencyScore`. This connects directly to the later KGAS/Digimons type-contract theme. [10]

# Runtime Caveats

The config has empty API-key fields and development Neo4j defaults, so it is not evidence that API-backed or database-backed execution succeeded in this environment. It is configuration scaffolding plus local defaults. [2]

`database/neo4j_setup.py` explicitly prints "Running in mock mode..." when core imports fail, and many Neo4j operations return mock data or skip work when no driver connection exists. This makes database code presence distinct from live database execution. [4]

The stress-test runner has a similar fallback: if imports fail, it declares "Framework components running in mock mode" and defines local stub versions of schema registry and tool integration classes. That makes the root result reports partly demonstration-harness evidence unless rerun traces show live dependencies. [11]

The registry path fields preserve absolute historical paths under `/home/brian/projects/Digimons/...`. These are useful provenance clues, but they should not be copied into new code as stable runtime paths. [9]

# Relationship To Wiki

- [Digimon Lineage Old Backups Stress Test Root](/wiki/sources/digimon-lineage-old-backups-stress-test-root.md): root reports and scripts that exercised these support files.
- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md): tool capability YAMLs and compatibility scaffolding are an earlier concrete form of that thread.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): mock-mode fallbacks make it important to distinguish code presence, demonstration execution, and live dependency execution.
- [Model Form Routing](/wiki/concepts/model-form-routing.md): the theory schemas explicitly name graph, table, and vector mappings as theory-application surfaces.
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md): registry/schema compatibility disagreement is another example of schema-state drift inside the same artifact bundle.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/config/default.yaml`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/data/policy_documents/climate_policy_proposal.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/database/neo4j_setup.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/framework/schema_registry.py`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/framework/tool_integration.py`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/schemas/stakeholder_schemas.py`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/scripts/salience_calculator.py`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/theory/registry.json`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/tool_capabilities/stakeholder_analyzer.yaml`  
[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/stress_test_2025.07211755/run_stress_test.py`

