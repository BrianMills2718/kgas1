---
type: SourceSummary
title: Digimon Lineage Uncertainty Stress Test Testing
description: Testing harness for the archived uncertainty stress test, covering IC-inspired deterministic stress tests, runners, methodology walkthrough, uncertainty-propagation test, and the root stress-test summary output.
tags: [source, digimon-lineage, uncertainty, testing, ic, ach, calibration, bayesian]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/
confidence: high
---

# Summary

This slice covers `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/`. The directory contains nine Python files, about 180K on disk, and aggregate hash `aggregate-sha256:c8cb5737ad98dc5cbe692f0260d0471cd1eb0ccb8670eb9ca3069afbd45113c6`. [1]

The folder is mostly harness code, not stored outputs. It defines IC-inspired uncertainty stress tests for information value, stopping rules, ACH theory competition, calibration, and mental-model auditing; the corresponding `stress_test_summary.json` output is preserved at the uncertainty-stress-test root and reports 5/5 success. [1][2]

# Inventory

| File | Bytes | Role |
| --- | ---: | --- |
| `run_all_tests.py` | 4,256 | Master runner for five IC-inspired stress tests; writes `stress_test_summary.json`. [3] |
| `test_information_value_assessment.py` | 15,605 | Heuer four-type information-value categorization over academic scenarios. [4] |
| `test_stopping_rules.py` | 18,992 | Multiple stopping rules for information collection, including diminishing returns, confidence threshold, cost/benefit, time, convergence, and discrimination. [5] |
| `test_ach_theory_competition.py` | 29,314 | Analysis of Competing Hypotheses engine with evidence consistency, diagnosticity, probability updates, and sensitivity analysis. [6] |
| `test_calibration_system.py` | 23,649 | Calibration tracking, Brier-score style metrics, resolution/reliability, category calibration, feedback, and large-scale prediction generation. [7] |
| `test_mental_model_auditing.py` | 38,444 | Mental-model bias auditing across confirmation, availability, anchoring, representativeness, hindsight, overconfidence, base-rate neglect, framing, sunk cost, and groupthink patterns. [8] |
| `run_basic_test.py` | 7,906 | Basic import, instantiation, data-structure, test-data, and optional LLM-connectivity runner. [9] |
| `run_uncertainty_test.py` | 17,827 | Uncertainty propagation test over a Digimon tool-chain idea with text quality, entity uncertainty, and cross-modal evidence integration. [10] |
| `methodology_walkthrough.py` | 6,603 | Step-by-step LLM-guided Bayesian inference walkthrough using an illustrative narcissism-score example. [11] |

# Preserved Result Link

The root `stress_test_summary.json` records the master runner result from `2025-07-23T05:50:35`. It reports five total tests, five successful, zero failed, and the assessment that all IC-inspired features were working correctly. [2]

The summary includes stored excerpts for each test:

- information value: two scenarios, 5 and 7 information pieces, diagnostic/anomalous/consistent/irrelevant distribution, 1,000-piece processing claim, and three edge cases handled [2][4]
- stopping rules: two scenarios, three edge cases, stop points at iterations 10 and 19, and configurable combination strategy findings [2][5]
- ACH competition: three scenarios, up to 50 hypotheses and 100 evidence items, contradiction handling, probability updates, and sensitivity analysis [2][6]
- calibration system: three tests, three edge cases, academic Brier score about 0.197, large-scale Brier score about 0.223, and no temporal drift detected [2][7]
- mental-model auditing: 103 models tested, 10 detectable biases, and bias-detection-rate summaries for confirmation bias, overconfidence, anchoring, and groupthink [2][8]

# Methodology Importance

The testing folder is historically useful because it shows the "intelligence-community inspired" turn as executable demonstrations, not only prose. The implemented families line up with the root README and result claims: Heuer information-value categories, stopping rules, ACH, calibration, and mental-model auditing. [2][3]

The methodology walkthrough captures the intended hybrid division of labor: the LLM extracts psychologically meaningful evidence and likelihood ratios, while deterministic Bayesian math performs precision-weighted updating and uncertainty intervals. This complements the formal Bayesian and LLM-native service pages. [11]

# Caveats

These are largely synthetic/demonstration stress tests, not locked benchmarks against external expert labels. The root result file shows successful execution of the harness, but it does not prove that the algorithms generalize outside the constructed scenarios. [2]

`run_basic_test.py` and `run_uncertainty_test.py` contain hardcoded historical paths under `/home/brian/projects/Digimons`. `run_basic_test.py` appends `./core_services` even though the preserved testing directory is a sibling of `core_services`, and it looks for lit-review test texts under an old absolute path. `run_uncertainty_test.py` appends the old Digimons project root to `sys.path`. These are preservation facts and likely portability blockers for a current rerun. [9][10]

The master runner captures only the last 1,000 characters of each test's stdout in `stress_test_summary.json`. The full run logs are not preserved in this folder, so detailed intermediate behavior must be reconstructed from the scripts or rerun in a repaired environment. [2][3]

No literal OpenAI or Google API keys were found in this testing folder during a targeted credential-pattern scan. [1]

# Relationship To Wiki

- [Digimon Lineage Uncertainty Stress Test Root](/wiki/sources/digimon-lineage-uncertainty-stress-test-root.md): root page summarizes the 5/5 IC-inspired test result.
- [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md): service code behind the uncertainty framework.
- [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md): separate validation-output directory for LLM-native, formal Bayesian, and SocialMaze examples.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md): this page shows the IC-feature implementation/testing side of the uncertainty evolution.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): a 5/5 synthetic harness result should be stated as harness evidence, not broad system proof.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/stress_test_summary.json`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/run_all_tests.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/test_information_value_assessment.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/test_stopping_rules.py`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/test_ach_theory_competition.py`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/test_calibration_system.py`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/test_mental_model_auditing.py`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/run_basic_test.py`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/run_uncertainty_test.py`  
[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/testing/methodology_walkthrough.py`
