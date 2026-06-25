---
type: SourceSummary
title: Digimon Lineage Archived Uncertainty Datasets
description: Structural provenance and privacy review for the large dataset/config/result files in the 2025-07 archived uncertainty experiments tree, without reproducing raw tweet text.
tags: [source, digimon-lineage, uncertainty, datasets, privacy, provenance]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/100_users_500tweets_dataset.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/high_volume_500tweet_dataset.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/config/default.yaml
confidence: high
---

# Summary

This slice covers the large root-level dataset, ground-truth, config, and result files in `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/`. It intentionally records structure and privacy implications without quoting raw tweet text. [1]

The datasets are valuable because they preserve a concrete validation corpus for psychological-trait inference from Twitter-like behavior. They are also sensitive because they include `twitter_id`, `twitter_screen_name`, tweet IDs, timestamps, full tweet text, public handles/mentions, links, and psychological survey/ground-truth scores. [2][3][4][5]

# File Inventory

| File | Size | SHA-256 | Role |
| --- | ---: | --- | --- |
| `100_users_500tweets_dataset.json` | 16,027,151 | `171d1d8354f2401e97a6ba293f4019663c7e79c3cca588f1a9d6c80dd9693fc6` | Main 100-user dataset. [2] |
| `100_users_500tweets_ground_truths.json` | 15,060 | `cca8b674a5334c7ff4a54ddd745dbeae92d161f73556697891a5725a236af476` | Ground-truth scores for the 100-user dataset. [3] |
| `high_volume_500tweet_dataset.json` | 13,420,747 | `ec1b1db16ec20f9a84e8ee970deb5df102a127a3a7da2201b9915addac2f6f08` | 84-user high-volume dataset. [4] |
| `high_volume_ground_truths.json` | 12,655 | `a5b298aff3e9a15473027bef4ede65f761f874a2278ae89ac6a162eeff0fc633` | Ground-truth scores for the 84-user dataset. [5] |
| `uncertainty_test_results_20250727_024013.json` | 1,723 | `c5e940cb56b01a9c591ca251ca2b8507ef7b119ff731d3e904b85948cffe5efc` | Two uncertainty propagation/integration test results. [6] |
| `config/default.yaml` | 1,316 | `231c9e0b23463a92e084a667e6e8402557cd1fc28e790fe3207a22680020101b` | Empty-key development config with Neo4j and processing defaults. [7] |

# Dataset Structure

`100_users_500tweets_dataset.json` is a JSON list with 100 user records. Each record has `survey_scores`, `tweet_count`, `tweets`, and `user_info`. `user_info` includes psychological item fields, `twitter_id`, `twitter_screen_name`, `original_source_id`, and `user_index`. Each tweet object has `created_at`, `id`, `lang`, `like_count`, `retweet_count`, and `text`. [2]

`high_volume_500tweet_dataset.json` is a JSON list with 84 user records. Each record has `survey_scores`, `synthetic_metadata`, `tweet_count`, `tweets`, and `user_info`. `user_info` includes psychological item fields, `twitter_id`, `twitter_screen_name`, `original_id`, and `synthetic_version`. Each tweet object uses the same tweet fields as the 100-user dataset. [4]

The ground-truth files contain lists of psychological scores with `political_orientation`, `narcissism`, `conspiracy_mentality`, and `science_denialism`. The 100-user ground-truth file records `creation_method: systematic_replication` and `tweets_per_user: 500`; the high-volume ground-truth file records `creation_method: pattern_replication`. [3][5]

# Count And Consistency Notes

The actual number of tweet objects is 50,000 in the 100-user dataset and 42,000 in the high-volume dataset. The per-record `tweet_count` field sums to 30,905 and 25,705 respectively, so `tweet_count` should not be treated as a reliable count of included tweet objects without checking how it was defined. [2][4]

A structural scan found many public-handle and link-like patterns: 38,209 tweets with @mentions and 20,641 tweets with t.co/Twitter-like links in the 100-user dataset; 32,247 tweets with @mentions and 16,993 tweets with t.co/Twitter-like links in the high-volume dataset. The same scan found a small number of email-like strings in tweet text. [2][4]

# Privacy And Sharing Policy

Treat these datasets as sensitive research data. Even if the corpus was partially synthetic, replicated, or transformed, the preserved files contain identifiers, handles, tweet IDs, timestamps, full tweet text, and psychological scores in the same records. Do not quote raw tweet text in wiki pages, commit excerpts to public docs, or export these files without a separate ethics/privacy review. [2][3][4][5]

For future work, use manifest-level summaries and derived aggregate statistics by default. If model evaluation requires the raw files, write a specific access plan covering redaction, storage, sharing, and whether public handles or IDs may be retained. [2][4]

# Config And Result Notes

`config/default.yaml` has empty API key and password fields, development Neo4j defaults, text/entity/graph thresholds, theory settings, and workflow settings. It is useful as a historical configuration artifact but does not contain live secrets. [7]

`uncertainty_test_results_20250727_024013.json` contains two root tests: tool-chain uncertainty propagation and cross-modal evidence integration. The first records final confidence `0.514564824846515` and marks `final_confidence_appropriate` false; the second records final confidence `0.6531040268456375`, detects conflict, and passes its listed success criteria. [6]

# Relationship To Wiki

- [Digimon Lineage Archived Uncertainty Tests Overview](/wiki/sources/digimon-lineage-archived-uncertainty-tests-overview.md): parent overview for the archived bundle.
- [Digimon Lineage Archived Uncertainty Experiments Docs And Validation](/wiki/sources/digimon-lineage-archived-uncertainty-experiments-docs-validation.md): docs and validation page that uses these datasets as historical context.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): dataset presence does not by itself establish broad validation, especially with count inconsistencies and privacy constraints.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/100_users_500tweets_dataset.json`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/100_users_500tweets_ground_truths.json`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/high_volume_500tweet_dataset.json`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/high_volume_ground_truths.json`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/uncertainty_test_results_20250727_024013.json`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/config/default.yaml`
