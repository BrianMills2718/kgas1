---
type: SourceSummary
title: Digimon Lineage Archived Uncertainty Experiments Code Delta
description: Difference-focused source summary for non-duplicate code in the 2025-07 archived uncertainty experiments tree versus the already-ingested reorganized uncertainty tree.
tags: [source, digimon-lineage, uncertainty, code-delta, testing, ner, socialmaze]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/
confidence: high
---

# Summary

This slice records the code-level differences between the non-duplicate `2025_07_experiments/uncertainty_stress_test_system/` tree and the already-ingested duplicate/reorganized uncertainty tree. Most experiment code directories are subsets of the reorganized copy; this page documents only the deltas. [1][2]

Aggregate hashes for the experiment code directories are: analysis `aggregate-sha256:0171fae8767859bb9e2db3634c8d33d8ca6b5aa254049a4f493a81c5e60da399`, core services `aggregate-sha256:904c1feb7eb14736273db0a9d143ff14ea64718755ddd81221814e7f3fbe6b64`, Bayesian `aggregate-sha256:2a63310d8bfca210121f22dfd0e7460c87098f93eeb9d2d7605a4a90566fbf20`, testing `aggregate-sha256:21b9c03139fc954796c49746add132cb880481e3ab667c1941b9a1ae6d15cbc3`, and setup `aggregate-sha256:7e2437837ed2d891843960e5641553a8c36627424ae3b414782cb80077d70ed5`. [1]

# Directory Delta

| Directory | Difference Versus Reorganized Copy |
| --- | --- |
| `analysis/` | No unique experiment files; reorganized copy adds `massive_text_processor.py` and `parallel_reader_coordinator.py`. [1][2] |
| `core_services/` | No unique experiment files; reorganized copy adds four later service files. [1][2] |
| `bayesian/` | No unique experiment files; reorganized copy adds two later Bayesian scripts. [1][2] |
| `setup/` | No unique experiment files; reorganized copy adds `auto_neo4j_setup.py` and `one_click_kgas_setup.py`. [1][2] |
| `testing/` | Two unique experiment files: `test_ner_direct.py` and `test_socialmaze_uncertainty.py`; reorganized copy adds four other test harness files. [1][2] |

# Unique Test Files

`testing/test_ner_direct.py` is a direct spaCy NER diagnostic. It loads `en_core_web_sm`, runs NER over two synthetic profile texts, formats entities in a KGAS-like structure, and prints that direct NER is working and the issue is in the KGAS wrapper if extraction succeeds. This is diagnostic evidence about wrapper/resource-manager failure, not a general KGAS pipeline test. [3]

`testing/test_socialmaze_uncertainty.py` is a SocialMaze-inspired Bayesian uncertainty test harness. It defines five social-reasoning scenarios covering theory of mind, group dynamics, deception detection, authority bias, and conformity bias. It tries to import uncertainty core services, falls back to mock classes if imports fail, and also uses mock mode when `OPENAI_API_KEY` is absent. It writes result JSON and a markdown report under `validation/socialmaze_results/`. [4]

The preserved SocialMaze output covered in [Digimon Lineage Archived Uncertainty Experiments Docs And Validation](/wiki/sources/digimon-lineage-archived-uncertainty-experiments-docs-validation.md) is therefore best read with this code caveat: the harness was explicitly designed to run in mock mode when services or API credentials were unavailable. [4][5]

# Rerun Caveats

Do not rerun these scripts as preservation work. `test_ner_direct.py` requires a local spaCy model, and `test_socialmaze_uncertainty.py` depends on import paths and optionally live OpenAI credentials. If rerun is needed later, create a separate reproducibility task that records environment, dependencies, mode, and output paths. [3][4]

# Relationship To Wiki

- [Digimon Lineage Archived Uncertainty Tests Overview](/wiki/sources/digimon-lineage-archived-uncertainty-tests-overview.md): parent overview for the archived uncertainty bundle.
- [Digimon Lineage Archived Uncertainty Experiments Docs And Validation](/wiki/sources/digimon-lineage-archived-uncertainty-experiments-docs-validation.md): records the corresponding SocialMaze output and validation results.
- [Digimon Lineage Uncertainty Stress Test Testing](/wiki/sources/digimon-lineage-uncertainty-stress-test-testing.md): already covers the larger reorganized testing harness.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): unique diagnostic and mock-capable tests should not be promoted to broad runtime proof.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_reorganized/uncertainty_stress_test/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/testing/test_ner_direct.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/testing/test_socialmaze_uncertainty.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/socialmaze_results/socialmaze_uncertainty_test_20250724_000934.json`
