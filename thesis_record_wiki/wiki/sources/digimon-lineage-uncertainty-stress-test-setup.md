---
type: SourceSummary
title: Digimon Lineage Uncertainty Stress Test Setup
description: Setup and demo scripts for the archived uncertainty stress test, including Neo4j Docker automation, one-click UI helpers, complete demo output generation, and safety/rerun caveats.
tags: [source, digimon-lineage, uncertainty, setup, neo4j, docker, demo]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/setup/
confidence: high
---

# Summary

This slice covers `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/setup/`. The directory contains three Python files, about 36K on disk, and aggregate hash `aggregate-sha256:3fe3c3e1836099bb623670b807242726dbf3d9bc729cd7c461e6086fd14b26e5`. [1]

The setup folder preserves convenience automation for a historical Neo4j-backed KGAS demo: Docker container management, one-click setup/status/stop wrappers, and a complete demo script intended to show Neo4j, structured/SQL-equivalent, vector, and final outputs. [1]

# Inventory

| File | Bytes | Role |
| --- | ---: | --- |
| `auto_neo4j_setup.py` | 11,428 | `AutomaticNeo4jManager` for Docker availability checks, port/container checks, container creation, readiness waits, connection info, safe cleanup, and optional stop/remove-data behavior. [2] |
| `one_click_kgas_setup.py` | 5,196 | One-click wrapper around `AutomaticNeo4jManager` for CLI/UI setup, status, and stop functions. [3] |
| `complete_kgas_demo.py` | 11,791 | Demo runner that sets up Neo4j, creates two synthetic psychological profiles, runs a KGAS validator pipeline, displays Neo4j/structured/vector/final outputs, and writes JSON/CSV demo files. [4] |

# Operational Meaning

The Neo4j manager creates or starts a Docker container, defaulting to `neo4j-graphrag`, ports `7474` and `7687`, password `testpassword`, and image `neo4j:5.15`. The one-click wrapper uses container name `kgas-neo4j` and the same demo password. [2][3]

The demo script is useful because it shows what a complete demonstration was supposed to expose: raw processing results, Neo4j graph data, vector embeddings, structured tabular output, final analysis summary, and saved JSON/CSV outputs. It also explicitly says vector storage was not implemented in the current pipeline. [4]

# Safety And Rerun Caveats

These scripts should not be run casually by an agent during preservation work. They can start Docker containers, bind local ports, pull a Neo4j image, stop containers, and in one code path remove Docker containers and volumes when `remove_data=True`. The default UI stop path preserves data, but the deletion code exists. [2][3]

The password `testpassword` is a hardcoded demo credential, not a secret to preserve or reuse. The scripts print connection information including username/password. No literal OpenAI or Google API keys were found in this setup folder during a targeted credential-pattern scan. [1][2][3][4]

`complete_kgas_demo.py` imports `kunst_memory_safe_validator` directly after appending the `setup/` directory and its parent to `sys.path`; the actual preserved validator file is in the sibling `validation/` directory. That means the import path appears fragile unless the runtime working directory or Python path supplied that sibling module externally. [4][5]

No preserved `kgas_demo_results_*.json` or `kgas_demo_data_*.csv` outputs were found in this setup directory during this ingest. The page records the demo script's intended outputs, not an observed completed run. [1][4]

# Relationship To Wiki

- [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md): contains `kunst_memory_safe_validator.py`, the validator the complete demo tries to import.
- [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md): service implementation layer that these setup/demo scripts are meant to make runnable.
- [Digimon Lineage Uncertainty Stress Test Docs](/wiki/sources/digimon-lineage-uncertainty-stress-test-docs.md): specification layer that discusses Neo4j and integration points.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): setup automation and demo scripts do not equal preserved successful demo outputs.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): historical Docker scripts require current environment verification before use.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/setup/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/setup/auto_neo4j_setup.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/setup/one_click_kgas_setup.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/setup/complete_kgas_demo.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/kunst_memory_safe_validator.py`
