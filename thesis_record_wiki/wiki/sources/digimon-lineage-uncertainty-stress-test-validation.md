---
type: SourceSummary
title: Digimon Lineage Uncertainty Stress Test Validation
description: Validation subdirectory for the archived uncertainty stress test, including preserved basic connectivity, formal Bayesian, LLM-native comparison, SocialMaze mock-mode outputs, validator scripts, and missing-output caveats.
tags: [source, digimon-lineage, uncertainty, validation, bayesian, llm, socialmaze, bias]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/
confidence: high
---

# Summary

This slice covers `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/`. The directory contains 17 files, about 364K on disk, and aggregate hash `aggregate-sha256:d920be2abb26f6049a920aee1f642a7de07af9b4a91c5dafdaef4c97cb6d4042`. [1]

The validation folder preserves a mixed evidence layer: real-output JSON/report files for basic setup, formal Bayesian examples, LLM-native comparison, and SocialMaze-inspired analysis; plus larger validator scripts for comprehensive testing, ground-truth cases, bias analysis, and a memory-safe Kunst dataset validator. [1]

# Inventory

Preserved outputs:

- `basic_test_results.json` - basic API/import/data/connectivity check. [2]
- `formal_bayesian_test_results.json` and `formal_bayesian_report.md` - medical-treatment formal Bayesian example. [3][4]
- `extraordinary_claim_results.json` and `extraordinary_claim_report.md` - cold-fusion extraordinary-claim formal Bayesian example. [5][6]
- `llm_native_test_results.json` - two early LLM-native test results. [7]
- `llm_native_comprehensive_results_20250723_151652.json` and `llm_native_comparison_report_20250723_151652.md` - seven-case LLM-native versus rule-based comparison. [8][9]
- `socialmaze_results/socialmaze_uncertainty_test_20250724_000934.json` and `SOCIALMAZE_UNCERTAINTY_REPORT_20250724_000934.md` - SocialMaze-inspired uncertainty analysis output. [10][11]

Preserved validation scripts:

- `comprehensive_uncertainty_test.py` - planned all-component tests for Bayesian aggregation, uncertainty engine, CERQual, integrated workflow, and stress/performance. [12]
- `ground_truth_validator.py` - generated high/low/moderate/edge confidence cases with expected ranges and report generation. [13]
- `bias_analyzer.py` and `quick_bias_test.py` - source-prestige, recency, domain, confirmation, sample-size, complexity, and gender-bias test scaffolding. [14][15]
- `llm_native_comprehensive_test.py` and `test_extraordinary_claim.py` - scripts corresponding to preserved LLM-native and extraordinary-claim outputs. [16][17]
- `kunst_memory_safe_validator.py` - memory-safe validator for a claimed 4.5GB Kunst dataset run using KGAS infrastructure, with Neo4j/Docker setup and no-mocking language in the docstring. [18]

# Positive Evidence

The basic result file records `api_key_status: present`, successful imports, basic instantiation, test data availability, 10 test files found, working data structures, LLM connectivity, and no errors at `2025-07-23T12:49:12`. [2]

The formal Bayesian medical example reports posterior belief `0.9642857142857143`, confidence bounds `[0.8520359926824961, 0.999]`, Bayes factor `9.0`, one API call, and "major belief revision" for the claim that Treatment X significantly reduces mortality compared with standard care. [3][4]

The formal Bayesian cold-fusion example reports posterior belief `0.06572399985459643`, confidence bounds `[0.001, 0.47620366078131604]`, Bayes factor `16.0`, one API call, and "minor belief change" despite strong evidence because the prior is very low. That is good evidence that the formal Bayesian path could represent extraordinary-claim conservatism. [5][6]

The LLM-native comparison report records seven total cases, 422.8 seconds of execution, LLM-native accuracy of 7/7 within expected ranges, and rule-based accuracy of 5/7. The preserved JSON records LLM-native confidences for `medical_strong` 0.82, `physics_theoretical` 0.52, `social_science_weak` 0.35, `interdisciplinary_complex` 0.55, `climate_meta_analysis` 0.88, `extraordinary_claim` 0.25, and `humanities_interpretation` 0.70. [8][9]

The SocialMaze output covers theory-of-mind, group-dynamics, deception-detection, authority-bias, and conformity-bias scenarios. Its summary reports four test groups, five scenarios, mean calibration error about 0.06, bias detection accuracy 1.0, and calibration quality `excellent`. [10][11]

# Caveats

The SocialMaze output is explicitly `mock_mode: true` with a near-zero duration, so it is a conceptual/demo validation artifact rather than live LLM or full-system evidence. [10][11]

The comprehensive, ground-truth, and bias-analysis scripts write outputs named `comprehensive_test_results_*`, `test_summary_report_*`, `ground_truth_validation_results.json`, `ground_truth_validation_report.md`, `bias_analysis_results.json`, `bias_analysis_report.md`, and `bias_mitigation_strategies.json`, but those files are not present in this validation directory. Their preserved code is valuable, but the wiki should not infer that those full validation runs exist here. [12][13][14]

Several validation scripts hardcode the historical path `/home/brian/projects/Digimons/uncertainty_stress_test/...`, require `OPENAI_API_KEY`, and append the old `core_services` directory to `sys.path`. That is normal for a preserved archive but weakens any claim that the scripts are currently portable or rerunnable without repair. [12][13][14][15][16][17]

The root-level `VALIDATION_STATUS_REPORT.md` says ground-truth accuracy was 5/6 and bias resistance was poor, but the detailed ground-truth and bias output files are not present in this `validation/` slice. Treat the root report as a separate status artifact until the missing outputs are found elsewhere. [19]

No literal OpenAI or Google API keys were found in this validation folder during a targeted credential-pattern scan. The preserved outputs do say an API key was present during historical execution. [1][2]

# Relationship To Wiki

- [Digimon Lineage Uncertainty Stress Test Root](/wiki/sources/digimon-lineage-uncertainty-stress-test-root.md): root status/readiness reports constrain the validation interpretation.
- [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md): these outputs and scripts exercise the service code summarized there.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md): this page is the validation counterpart to the move from rule-based confidence toward LLM-native and formal Bayesian uncertainty.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): validation outputs support specific historical test claims, but missing outputs and mock-mode must remain visible.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): historical validation artifacts are not equivalent to current checkout runtime proof.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/basic_test_results.json`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/formal_bayesian_test_results.json`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/formal_bayesian_report.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/extraordinary_claim_results.json`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/extraordinary_claim_report.md`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/llm_native_test_results.json`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/llm_native_comprehensive_results_20250723_151652.json`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/llm_native_comparison_report_20250723_151652.md`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/socialmaze_results/socialmaze_uncertainty_test_20250724_000934.json`  
[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/socialmaze_results/SOCIALMAZE_UNCERTAINTY_REPORT_20250724_000934.md`  
[12] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/comprehensive_uncertainty_test.py`  
[13] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/ground_truth_validator.py`  
[14] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/bias_analyzer.py`  
[15] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/quick_bias_test.py`  
[16] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/llm_native_comprehensive_test.py`  
[17] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/test_extraordinary_claim.py`  
[18] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/validation/kunst_memory_safe_validator.py`  
[19] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/VALIDATION_STATUS_REPORT.md`
