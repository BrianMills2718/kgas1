---
type: Concept
title: Proposal Validation Evolution
description: Difference review of KGAS dissertation validation plans, preserving how validation moved from concrete extraction/SME metrics toward construct validity, uncertainty, and no-target baseline reporting.
tags: [concept, proposal, validation, dissertation, provenance, difference-review, claim-discipline]
created: 2026-06-26
updated: 2026-06-26
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/1/47_page_older_proposal.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/proposal_rewrite.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/annex_c_validation_framework.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/annex_d_human_subjects_protection.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/validation/academic-validation-matrix.md
  - /wiki/sources/proposal-old-to-final-comparison-2026-06-26.md
  - /wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md
confidence: high
---

> Sources consulted: `47_page_older_proposal.txt` validation sections · `proposal_rewrite.txt` RQ/validation sections · `annex_c_validation_framework.txt` · `annex_d_human_subjects_protection.txt` · `academic-validation-matrix.md` · [Proposal Old To Final Comparison 2026 06 26](/wiki/sources/proposal-old-to-final-comparison-2026-06-26.md) · [Final Proposal Annex Preservation Review 2026 06 26](/wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md). Status: bounded validation-plan difference review, not evidence that any proposed validation study was completed.

# Summary

The validation plan changed in a useful but lossy way. Older proposal text preserved concrete operational validation: hand-coded posts, two independent coders, Cohen's kappa, precision/recall/F1, HotPotQA-like multi-hop tests, qualitative SME face validation, replication against published findings, and model-validation examples. [1]

The final proposal package became more defensible for dissertation submission: construct estimates, correlation with validated psychological scales, crowd coding, Davis validity dimensions, fourteen uncertainty dimensions, no predetermined performance targets, baseline establishment, and explicit HSPC/SME boundaries. [2][3][4]

The right reconstruction is not "old validation was worse." The old plan is stronger for engineering concreteness. The final plan is stronger for academic claim discipline and human-subjects risk control.

# Validation Evolution Table

| Validation layer | Older proposal plan | Final proposal / annex plan | Preservation judgment |
| --- | --- | --- | --- |
| Extraction accuracy | Hand-coded 300-500 documents, two coders, concrete entity/stance/claim extraction, Cohen's kappa target above 0.80, precision/recall/F1. [1] | Basic extraction accuracy through crowd-sourced coding; Annex C names entity recognition F1 and extraction-completeness recall as dimensions. [2][3] | Preserve both. The old plan gives concrete protocol shape; the final plan avoids overpromising dissertation success. |
| Higher-order constructs | Abstract concepts such as narratives and causal claims are explicitly harder, with quantitative benchmarking focused on concrete tasks and qualitative checks deferred to Essay 3. [1] | RQ2 centers construct estimates and asks whether computational estimates correlate with validated psychological scales. [2] | Final plan is stronger as a dissertation construct-validity frame, but old caution about abstract coding difficulty remains important. |
| SME validation | SME summaries and tool feedback appear as face-validity/usability checks, with HSPC screening caveats. [1] | Systematic SME evaluation becomes an HSPC modification trigger; final package emphasizes crowd coding and construct validation instead. [4] | This is risk reduction, not deletion of SME value. SME evidence should be treated as optional/internal unless formal review exists. |
| Cross-modal value | Convergent analysis across theories and tools supports plausibility and theory robustness. [1] | RQ3 asks what graph/table/vector modalities reveal, with operational indicators such as influential-user overlap, variance-explained improvement, and analyst interpretation time. [2] | Final plan makes cross-modal comparison more operational, but older convergent-analysis rationale explains why it mattered. |
| Theory validation | Replication of published findings and applying theory schemas to original datasets appear as validation methods. [1] | Annex C maps this into Davis description, causal explanation, postdiction, exploratory, and aspirational prediction validity. [3] | Final annex gives a better taxonomy; old text preserves concrete replication instincts. |
| Performance targets | Older text uses explicit reliability/performance targets in places. [1] | Final package says report actual capabilities, establish baselines, include failures, and avoid predetermined targets. [3][5] | Final target discipline is better for committee-facing claims; old targets can survive as exploratory process metrics. |
| Ethics/HSPC | Older text says the current design is not human subjects research but still requires HSPC screening and modification if SME evaluation becomes systematic. [1] | Annex D gives a fuller boundary: public/de-identified data, no interaction, no PII collection, identity-protection safeguards, and HSPC triggers. [4] | Final annex is the stronger governance artifact; older text shows the concern was already present. |

# What Was Improved

The final package improved claim posture in three ways.

First, it moved from tool-success language toward construct-validity language. The dissertation claim became whether KGAS can generate construct estimates that align with available measures, not whether KGAS broadly "improves" analysis. [2]

Second, it adopted a no-target baseline philosophy. Annex C requires metric definitions, procedures, results, evidence files, and limitations, then explicitly says to report actual capabilities rather than tune to predetermined targets. [3]

Third, it tightened the human-subjects boundary. Annex D preserves the non-human-subjects rationale while naming modification triggers for systematic SME evaluation, restricted data, external participants, identifiable information, or funding/IRB requirements. [4]

# What The Older Plan Preserves

The older proposal remains valuable because it contains protocol-level detail that later compression can hide:

- sample size scale for manual coding;
- independent-coder design;
- specific coding targets such as named entities, stance, claims, and sentiment;
- use of precision, recall, F1, Cohen's kappa, and qualitative face validity;
- multi-hop reasoning and replication instincts;
- a direct mitigation strategy for LLM extraction risk. [1]

Those details should not become dissertation success promises automatically. They are a reusable validation design menu.

# Academic Validation Matrix Role

The academic validation matrix sits between the older and final plans. It lists planned tests such as HotPotQA multi-hop retrieval, COVID construct correlations, hand-coded theory comparison, inter-LLM reliability, theory replication, Mechanical Turk coding, cross-modal capabilities, and full-pipeline paper/dataset replication. [5]

Its key contribution is priority separation:

- must-do proof-of-concept validation;
- should-do strengthening tests;
- could-do tests if time permits.

Its caveat is that it is planning evidence, not completed validation evidence. It should be cited as the validation backlog and design rationale, not as proof that KGAS passed those tests.

# Practical Rule

When reconstructing or restarting thesis validation work, separate five categories:

1. **Completed runtime evidence**: tests, logs, or artifacts actually run in the current repo.
2. **Historical validation outputs**: preserved reports from older variants, with date and runtime caveats.
3. **Proposal validation commitments**: tests promised or described in proposal text.
4. **Validation design menu**: optional older ideas worth reusing.
5. **Governance boundaries**: HSPC, data privacy, and claim-discipline constraints.

Do not collapse these into one validation status.

# Links

- [Proposal Framing Evolution](/wiki/concepts/proposal-framing-evolution.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Proposal Old To Final Comparison 2026 06 26](/wiki/sources/proposal-old-to-final-comparison-2026-06-26.md)
- [Final Proposal Annex Preservation Review 2026 06 26](/wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md)
- [Digimon Lineage Proposal Rewrite 2025 08 12](/wiki/sources/digimon-lineage-proposal-rewrite-2025-08-12.md)
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/1/47_page_older_proposal.txt`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/proposal_rewrite.txt`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/annex_c_validation_framework.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/annex_d_human_subjects_protection.txt`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/validation/academic-validation-matrix.md`
