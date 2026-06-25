---
type: SourceSummary
title: Digimon Lineage Archived Uncertainty Tests Overview
description: Overview of the archived uncertainty-tests bundle, including duplicate reorganized copy, 2025-07 experiments tree, datasets, incomplete personality-prediction checkout, cleanup recommendations, and credential-scan caveats.
tags: [source, digimon-lineage, uncertainty, archive, datasets, personality-prediction]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/
confidence: high
---

# Summary

This slice covers `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/`. The bundle is about 57 MB and contains 178 regular files plus 53 symlinks. Excluding nested `.git` files, the aggregate hash is `aggregate-sha256:a9ec1d41262727f788fc8078cec0007c0ceda0957362114bc6dcce5bf36c6a44`. [1]

The directory has two major subtrees. `2025_07_reorganized/uncertainty_stress_test/` is a content-identical duplicate of the already-ingested `archive/uncertainty_stress_test/` tree: both have 95 real files, 9,218,730 bytes, and content-manifest hash `80ad1494b4c7b88ca5fe14473844f4ea6caf953d5033f80a936ec3d634c598af`. [1][2][3]

The non-duplicate value is `2025_07_experiments/uncertainty_stress_test_system/`, which contains datasets, configuration, extra docs, an archive cleanup rationale, a small result JSON, and an incomplete `personality-prediction` checkout. Excluding nested `.git` files, the experiments subtree has aggregate hash `aggregate-sha256:f3008b7209a8428dabb9444ba0e14b88762dd228829c8540c5b03daf2eaff4a6`. [4]

# Inventory

| Subtree | Role |
| --- | --- |
| `2025_07_reorganized/uncertainty_stress_test/` | Duplicate reorganized copy of the uncertainty-stress-test tree already represented in the wiki. [2][3] |
| `2025_07_experiments/uncertainty_stress_test_system/` | Earlier/larger experiment tree with datasets, config, docs, validation scripts/results, and incomplete personality-prediction checkout. [4] |
| `legacy_scripts/` | Empty or effectively negligible legacy bucket; aggregate hash from an empty file list is `abcfa6a9d4df344d1781bc2560b5e4cdcae08b39ed303063535e7e1e926a304a`. [1] |

# 2025-07 Experiments Tree

The experiments tree preserves four dataset/ground-truth files at root:

| File | Size | Observed Shape |
| --- | ---: | --- |
| `100_users_500tweets_dataset.json` | 16,027,151 bytes | JSON list of 100 users; each item has `survey_scores`, `tweet_count`, `tweets`, and `user_info`. [5] |
| `100_users_500tweets_ground_truths.json` | 15,060 bytes | JSON object with `user_count: 100`, `tweets_per_user: 500`, `creation_method: systematic_replication`, and 100 ground-truth entries. [6] |
| `high_volume_500tweet_dataset.json` | 13,420,747 bytes | JSON list of 84 users; each item has `survey_scores`, `synthetic_metadata`, `tweet_count`, `tweets`, and `user_info`. [7] |
| `high_volume_ground_truths.json` | 12,655 bytes | JSON object with `user_count: 84`, `creation_method: pattern_replication`, and 84 ground-truth entries. [8] |

It also preserves `uncertainty_test_results_20250727_024013.json`, which records two tests: tool-chain uncertainty propagation and cross-modal evidence integration. The propagation test produced final confidence `0.514564824846515` and marked final-confidence appropriateness false. The evidence-integration test produced final confidence `0.6531040268456375`, detected conflict, and passed its four listed success criteria. [9]

The config file `config/default.yaml` contains empty API key/password fields, Neo4j defaults, text/entity/graph processing thresholds, theory settings, and workflow settings. It is configuration evidence, not a secret leak. [10]

# Personality Prediction Caveat

The experiments tree contains `personality-prediction/`, but in this preserved copy it only contains a `.git` directory and no working files. Git reports remote `https://github.com/yashsmehta/personality-prediction.git`, and `git log` fails because the current `master` branch has no commits. [11]

This explains the broken symlinks recorded in [Digimon Lineage Uncertainty Stress Test Organization](/wiki/sources/digimon-lineage-uncertainty-stress-test-organization.md): the organization layer expected a populated `personality-prediction` working tree, but this archived path preserves only an incomplete or failed checkout. [11]

# Cleanup Documents

`archive_2025_07_25/DELETION_JUSTIFICATION.md` recommends deleting many remaining archive files after useful components were moved to active directories. It lists criteria such as superseded, historical, experimental, security risk, redundant, and outdated, and it claims zero-risk deletion because production value had been extracted. Treat that as a historical cleanup proposal, not as permission to delete preserved raw material now. [12]

`archive_2025_07_25/DIRECTORY_REVIEW_RECOMMENDATIONS.md` recommends keeping core production directories, docs, validation, testing, setup, Bayesian code, analysis, personality-prediction, datasets, and config. It recommends deleting external `BERT_personality_detection/` and `applied-bayesian-updating/`, and conditionally keeping audit-trail material. Several of those paths are not present in this preserved experiments tree. [13]

# Security And Privacy Notes

A targeted literal-key scan found placeholder API-key examples but no literal OpenAI or Google API keys in non-JSON files. The large dataset JSON files include tweet text and user-like records, so they should be treated as sensitive research data for sharing/export decisions even if they are synthetic or replicated. [5][7]

# Relationship To Wiki

- [Digimon Lineage Uncertainty Stress Test Organization](/wiki/sources/digimon-lineage-uncertainty-stress-test-organization.md): this page explains where some broken symlink targets were expected to come from.
- [Digimon Lineage Uncertainty Stress Test Root](/wiki/sources/digimon-lineage-uncertainty-stress-test-root.md): already covers the duplicate reorganized tree at the root/result level.
- [Digimon Lineage Uncertainty Stress Test Analysis](/wiki/sources/digimon-lineage-uncertainty-stress-test-analysis.md): already covers the duplicate analysis subtree.
- [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md): already covers much of the duplicate validation subtree.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): cleanup proposals and "zero risk" deletion claims need preservation-policy context.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_reorganized/uncertainty_stress_test/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/100_users_500tweets_dataset.json`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/100_users_500tweets_ground_truths.json`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/high_volume_500tweet_dataset.json`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/high_volume_ground_truths.json`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/uncertainty_test_results_20250727_024013.json`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/config/default.yaml`  
[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/personality-prediction/`  
[12] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/archive_2025_07_25/DELETION_JUSTIFICATION.md`  
[13] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/archive_2025_07_25/DIRECTORY_REVIEW_RECOMMENDATIONS.md`
