---
type: Concept
title: KGAS Dissertation Claim Map
description: Thesis-facing map of what the KGAS dissertation was trying to argue, preserving the distinction between final proposal claims, older ambition, validation design, runtime evidence, and unresolved caveats.
tags: [concept, dissertation, kgas, proposal, thesis, synthesis, claim-discipline]
created: 2026-06-26
updated: 2026-06-26
sources:
  - /wiki/concepts/thesis-recovery-current-state-2026-06-26.md
  - /wiki/concepts/proposal-framing-evolution.md
  - /wiki/sources/proposal-old-to-final-comparison-2026-06-26.md
  - /wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md
  - /wiki/concepts/proposal-validation-evolution.md
  - /wiki/concepts/hspc-data-governance-boundary.md
  - /wiki/concepts/kgas-evolution-checkpoint-2026-06-25.md
  - ../../investigations/2026-06-26-runtime-completion-review.md
confidence: high
---

> Sources consulted: [Thesis Recovery Current State 2026 06 26](/wiki/concepts/thesis-recovery-current-state-2026-06-26.md) · [Proposal Framing Evolution](/wiki/concepts/proposal-framing-evolution.md) · [Proposal Old To Final Comparison 2026 06 26](/wiki/sources/proposal-old-to-final-comparison-2026-06-26.md) · [Final Proposal Annex Preservation Review 2026 06 26](/wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md) · [Proposal Validation Evolution](/wiki/concepts/proposal-validation-evolution.md) · [HSPC Data Governance Boundary](/wiki/concepts/hspc-data-governance-boundary.md) · [KGAS Evolution Checkpoint 2026 06 25](/wiki/concepts/kgas-evolution-checkpoint-2026-06-25.md) · `investigations/2026-06-26-runtime-completion-review.md`. Status: thesis-facing synthesis map from existing wiki/review artifacts, not a new proposal draft and not evidence of committee approval.

# Summary

The dissertation claim that survived final proposal compression was not "KGAS is a finished AI policy-analysis product." It was closer to:

> Can a theory-aware computational framework characterize improvements in how social-science theories are selected, operationalized, applied to discourse data, and validated through construct estimates and cross-modal analysis?

That framing is intentionally narrower than the older KGAS ambition. It centers computational theory application, construct-estimate validation, cross-modal comparison, and baseline documentation. [1][2][3]

# Final Proposal Claim Surface

The final proposal claim surface has five parts:

1. **Theory application:** KGAS should help characterize systematic selection and application of social-science theories to discourse analysis. [3]
2. **Construct estimates:** Automated outputs should be evaluated by whether extracted construct estimates align with validated measures, not by broad claims of perfect measurement. [3][4]
3. **Cross-modal comparison:** The same discourse should be analyzed through graph, table, and vector modalities to learn what each representation adds. [3][4]
4. **Validation without optimization:** Results should establish baselines and report actual performance, not chase predetermined targets. [4][5]
5. **Governed data use:** The design assumes public/de-identified data and non-systematic internal consultation unless HSPC review changes the scope. [6]

# Older Ambition Preserved

Older proposal variants preserve material that the final proposal compressed or softened:

- low-code SME workbench and usability motivation;
- policy-analysis and strategic-foresight framing;
- broader fringe-discourse/UFO/COVID case horizon;
- concrete extraction metrics such as coder agreement, precision, recall, and F1;
- more expansive GraphRAG/DIGIMON/StructGPT comparison;
- dynamic tool generation, schema discovery, and cross-modal orchestration ambition. [2][3][7]

This older material is not obsolete junk. It is provenance for why the final proposal had to narrow, soften, and govern its claims.

# Validation Logic

The validation logic evolved from "prove the tool performs well" toward "document whether theory-grounded computational outputs align with available evidence."

The older plan is still useful because it gives concrete protocol ideas: hand-coded samples, two independent coders, Cohen's kappa, F1, HotPotQA-style multi-hop testing, SME face validity, and replication against published findings. [5]

The final plan is better for dissertation claim discipline because it centers construct validity, psychological-scale correlations, Davis validity categories, uncertainty dimensions, evidence files, limitations, and no predetermined performance targets. [4][5]

The reusable synthesis is:

- use older protocol details as a validation menu;
- use final proposal language as the claim boundary;
- never cite planned validation as completed validation.

# Runtime Evidence Boundary

The recovered current runtime is stronger than a paper-only archive, but it proves only bounded local slices.

As of the runtime completion review, current local evidence supports `.txt`, tiny `.pdf`, tiny `.md`, and tiny `.docx` `/api/analyze` paths through live Neo4j-backed smoke tests, plus a batch path wrapping the same single-document flow. Legacy `.doc` remains explicit 501. [8]

This evidence helps recover the program, but it is not the same as dissertation validation. It proves selected implementation paths can run locally; it does not prove the full construct-validation study, committee acceptance, or all historical architecture claims.

# Governance Boundary

The HSPC/data-governance story is also a claim boundary:

- the proposal reasoned that the design used public/de-identified data and no direct human interaction;
- systematic SME evaluation, restricted/private data, external participants, identifiable information, or funding/IRB requirements would trigger HSPC review;
- preserved HSPC references and Annex D are proposal/governance artifacts;
- no actual HSPC determination letter has been located in the current record. [6]

That means future restart work should not casually revive SME validation as if it were already approved research.

# What A Future Dissertation Restart Would Need

A credible restart would need to decide which layer is being revived:

| Layer | What exists now | What would still be needed |
| --- | --- | --- |
| Proposal argument | Strongly preserved in final proposal, annexes, and comparison pages. | Human decision on whether this is still the desired dissertation claim. |
| Historical ambition | Strongly preserved in older drafts and architecture/wiki pages. | Selection of which ambitions remain in scope. |
| Current runtime | Locally proven for narrow upload/batch paths. | Reproducible setup review and new validation study design. |
| Validation design | Strong menu of planned tests and claim-discipline rules. | Actual datasets, protocols, results, and evidence files. |
| Governance | Proposal HSPC reasoning and reference materials preserved. | Actual determination/approval status or new review if scope changes. |
| Public sharing | Private docs-only export exists. | Human privacy/narrative review before public release. |

# Practical Reading Rule

When reading any KGAS thesis artifact, ask which claim level it belongs to:

1. **Vision:** what KGAS could become.
2. **Proposal claim:** what the dissertation was willing to argue.
3. **Validation design:** how the claim would be tested.
4. **Historical evidence:** what older artifacts report.
5. **Current runtime proof:** what has been rerun locally.
6. **Governance boundary:** what cannot proceed without review.

Most confusion in the record comes from mixing these levels.

# Links

- [Thesis Recovery Current State 2026 06 26](/wiki/concepts/thesis-recovery-current-state-2026-06-26.md)
- [Proposal Framing Evolution](/wiki/concepts/proposal-framing-evolution.md)
- [Proposal Validation Evolution](/wiki/concepts/proposal-validation-evolution.md)
- [HSPC Data Governance Boundary](/wiki/concepts/hspc-data-governance-boundary.md)
- [KGAS Evolution Checkpoint 2026 06 25](/wiki/concepts/kgas-evolution-checkpoint-2026-06-25.md)

# Citations

[1] `/wiki/concepts/thesis-recovery-current-state-2026-06-26.md`  
[2] `/wiki/concepts/proposal-framing-evolution.md`  
[3] `/wiki/sources/proposal-old-to-final-comparison-2026-06-26.md`  
[4] `/wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md`  
[5] `/wiki/concepts/proposal-validation-evolution.md`  
[6] `/wiki/concepts/hspc-data-governance-boundary.md`  
[7] `/wiki/concepts/kgas-evolution-checkpoint-2026-06-25.md`  
[8] `../../investigations/2026-06-26-runtime-completion-review.md`
