---
type: SourceSummary
title: Digimon Lineage Archived Uncertainty Experiments Docs And Validation
description: Non-duplicate docs and validation slice from the 2025-07 archived uncertainty experiments tree, emphasizing external-review claims, bias warnings, Kunst validation, LLM-native results, and evidence-level caveats.
tags: [source, digimon-lineage, uncertainty, validation, external-review, kunst]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/
confidence: high
---

# Summary

This slice covers the non-duplicate docs and validation outputs under `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/`. The docs folder contains 13 markdown files with aggregate hash `aggregate-sha256:038e7fa0ed08e033ed69f0daf8bdfcbb21186e86e5f245b2b3f177a9bc7f40d3`; the validation folder contains 17 files with aggregate hash `aggregate-sha256:67deb8346ddc59a98a8ec2d63bae1c9ecc0cfeafcb015ec068aab8abcdf8603d`. [1][2]

The slice matters because it preserves a tension in the historical record. Some files call the uncertainty framework complete, production-ready, and externally reviewable. Other files, especially `VALIDATION_STATUS_REPORT.md`, say the system was 75% ready, had serious sample-size and language-complexity biases, and should fix critical issues before external review. [3][4][5]

# Docs Inventory

| File | Role |
| --- | --- |
| `COMPREHENSIVE_EXTERNAL_REVIEW_DOCUMENTATION.md` | Large compiled external-review packet combining implementation, validation, mathematical foundations, and code excerpts. [3] |
| `IMPLEMENTATION_REPORT.md` | Claims complete implementation, real LLM integration, success criteria met, and production-readiness direction. [4] |
| `VALIDATION_STATUS_REPORT.md` | Cautious status report: 75% ready, 83.3% ground-truth accuracy, 1/3 bias tests passed, fix critical biases before external review. [5] |
| `EXTERNAL_EVALUATION_DOCUMENT.md` | Technical review brief with limitations, bias questions, and 75% production-readiness framing. [6] |
| `LLM_NATIVE_EXTERNAL_EVALUATION.md` | LLM-native approach evaluation claiming 7/7 LLM-native accuracy versus 5/7 rule-based. [7] |
| `EXTERNAL_VALIDATION_EVIDENCE_PACKAGE.md` | Davis-enhanced external-validation package claiming 29 tests and improved bias calibration. [8] |
| `kunst_plan.md`, `kunst_validation_plan.md`, `kunst_validation_results.md` | Plan and results for validating psychological inference against Kunst et al.-style Twitter/psychology data. [9][10][11] |
| `REFERENCES.md`, `academic_validation_justification.md`, `llm_bayesian_insights.md` | Academic framing around causal datasheets, synthetic validation, and LLM/Bayesian methodology. [12][13][14] |
| `STRESS_TEST_RESULTS.md` | IC-inspired five-test stress result summary. [15] |

# Validation Outputs

`basic_test_results.json` records a July 27, 2025 basic check with API key present, imports successful, basic instantiation true, test data available, 11 test files found, data structures working, and LLM connectivity true. This differs from the reorganized duplicate's July 23 result only by timestamp and 11 versus 10 test files found. [16]

`formal_bayesian_test_results.json` preserves the same medical-treatment example seen elsewhere: posterior belief `0.9642857142857143`, Bayes factor `9.0`, and one API call. `extraordinary_claim_results.json` preserves the cold-fusion example: posterior belief `0.06572399985459643`, Bayes factor `16.0`, and one API call. [17][18]

`llm_native_comprehensive_results_20250723_151652.json` records seven LLM-native and seven rule-based cases. Its comparative analysis reports LLM-native accuracy `1.0` and rule-based accuracy `0.7142857142857143`, matching the claims in `LLM_NATIVE_EXTERNAL_EVALUATION.md`. [7][19]

The SocialMaze result remains mock-mode evidence: `mock_mode` is true, duration is `0.00009` seconds, mean calibration error is about `0.06`, bias detection accuracy is `1.0`, and calibration quality is `excellent`. It is useful as a designed test output, not live SocialMaze validation. [20]

# Kunst Validation

The Kunst validation docs claim that KGAS inferred psychological traits from Twitter behavior with 95% calibration coverage, political-orientation MAE `2.46` on a 1-11 scale, denialism MAE `1.02` on a 1-7 scale, and no major biases in initial testing. The same report says there was no tweet text, only 97 users with sufficient data, and only 4 of 7 psychological measures could be validated. [11]

The validation scripts add two Kunst-specific files absent from the reorganized validation slice: `final_kunst_validation.py` and `kunst_direct_validation.py`. The reorganized validation slice instead has `bias_analyzer.py` and `ground_truth_validator.py`, which are absent here. [2]

# Evidence Tension

The most important preservation point is not which status claim is "true"; it is that the archive contains incompatible status framings:

- Production/complete framing: `IMPLEMENTATION_REPORT.md`, `LLM_NATIVE_EXTERNAL_EVALUATION.md`, and parts of the external evidence package describe the framework as complete, production-ready, or ready for review. [4][7][8]
- Cautious framing: `VALIDATION_STATUS_REPORT.md` says sample-size over-weighting and language-complexity bias are project-killer issues before external evaluation, recommends fixing critical biases first, and says current bias resistance is poor. [5]
- Limited-evidence framing: validation outputs are small, partly mock-mode, or domain-specific; they are strong artifacts for historical development, not enough by themselves to prove general deployment readiness. [16][19][20]

# Safety And Rerun Caveats

Several validation scripts require `OPENAI_API_KEY` from the environment and use hardcoded historical output paths under `/home/brian/projects/Digimons/uncertainty_stress_test/...`. They should not be rerun during preservation without an explicit rerun plan and path correction. [2]

A targeted literal-key scan of the docs and validation slice found placeholder examples but no literal OpenAI or Google API keys. [1][2]

# Relationship To Wiki

- [Digimon Lineage Archived Uncertainty Tests Overview](/wiki/sources/digimon-lineage-archived-uncertainty-tests-overview.md): parent overview for the 57 MB archived bundle.
- [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md): overlapping validation/result material in the duplicate reorganized tree.
- [Digimon Lineage Uncertainty Stress Test Docs](/wiki/sources/digimon-lineage-uncertainty-stress-test-docs.md): later/reorganized documentation slice with methodology and implementation specification docs.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): this page is a primary example of status-claim conflict inside preserved evidence.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/COMPREHENSIVE_EXTERNAL_REVIEW_DOCUMENTATION.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/IMPLEMENTATION_REPORT.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/VALIDATION_STATUS_REPORT.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/EXTERNAL_EVALUATION_DOCUMENT.md`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/LLM_NATIVE_EXTERNAL_EVALUATION.md`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/EXTERNAL_VALIDATION_EVIDENCE_PACKAGE.md`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/kunst_plan.md`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/kunst_validation_plan.md`  
[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/kunst_validation_results.md`  
[12] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/REFERENCES.md`  
[13] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/academic_validation_justification.md`  
[14] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/llm_bayesian_insights.md`  
[15] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/docs/STRESS_TEST_RESULTS.md`  
[16] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/basic_test_results.json`  
[17] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/formal_bayesian_test_results.json`  
[18] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/extraordinary_claim_results.json`  
[19] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/llm_native_comprehensive_results_20250723_151652.json`  
[20] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_uncertainty_tests/2025_07_experiments/uncertainty_stress_test_system/validation/socialmaze_results/socialmaze_uncertainty_test_20250724_000934.json`
