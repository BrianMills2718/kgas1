---
type: SourceSummary
title: Proposal Old To Final Comparison 2026 06 26
description: Bounded comparison of older, full, and final KGAS dissertation proposal variants, focused on research questions, system positioning, validation framing, and preserved rationale.
tags: [source, proposal, dissertation, comparison, provenance, difference-review]
created: 2026-06-26
updated: 2026-06-26
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/1/47_page_older_proposal.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/2/Mills_Proposal_extracted.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/fragments/proposal_full_2025.08061424.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/proposal_rewrite.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/fragments/prateek_critique.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/final_submission_summary.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/final_revisions_summary.md
confidence: high
---

> Sources consulted: `47_page_older_proposal.txt` · `Mills_Proposal_extracted.txt` · `proposal_full_2025.08061424.txt` · `proposal_rewrite.txt` · `proposal_outline.md` · `prateek_critique.txt` · `final_submission_summary.md` · `final_revisions_summary.md`; plus file-size inventories for duplicate proposal trees under `archive/theoretical_exploration/...` and `archive/ARCHIVE_BEFORE_CLEANUP_20250805/...`. Status: bounded structure/delta review, not a full line-by-line textual diff.

# Summary

This comparison reviews three proposal variants:

| Variant | Size / structure | Role |
| --- | ---: | --- |
| `47_page_older_proposal.txt` | 679 lines | Older long proposal centered on low-code SME use, fringe discourse, and KGAS as an investigative workbench. [1] |
| `proposal_full_2025.08061424.txt` | 1046 lines | Expanded August 2025 proposal plus annexes, with strong theory-first architecture and detailed tool-suite sections. [2] |
| `proposal_rewrite.txt` | 550 lines | Final compressed proposal package, paired with final-submission and final-revisions summaries. [3][5][6] |

The proposal did not evolve by simply replacing an incorrect old idea with a correct new one. It evolved by changing the center of gravity:

- from low-code SME workbench to theory-aware computational-social-science proof of concept;
- from "improve analysis" to "characterize improvements" and document alignment;
- from broad fringe/UFO/COVID case possibilities to COVID/Kunst as the central validation case;
- from capability breadth to construct-estimate validation, baselines, and no predetermined targets;
- from detailed architecture in main text to compressed proposal plus annexes.

# Research Question Delta

The older proposal's core hypothesis says a low-code LLM/knowledge-graph tool can improve fringe-discourse analysis for policy insight. Its two primary questions ask whether SMEs can generate tailored knowledge graphs and whether SME users identify discourse elements more efficiently and accurately than manual or less-tailored tools. It also includes an F1 target above 0.80 for key entity and relationship extraction tasks. [1]

The final rewrite changes the research questions substantially. RQ1 asks whether computational frameworks can characterize improvements in systematic theory selection and application. RQ2 asks whether automated systems can extract construct estimates that correlate with validated measures across theoretical domains. RQ3 asks what graph, table, and vector modalities reveal when applied to the same discourse. [3]

This is not just wording. It changes what counts as success:

- older: SME usability, low-code graph generation, extraction accuracy, better/faster analysis;
- final: theory application, construct-estimate correlation, cross-modal comparison, documented alignment.

The final-revisions summary explicitly confirms one key edit: "improve" was changed to "characterize improvements" to avoid implying guaranteed performance. [6]

# System-Positioning Delta

The older proposal already contained a serious GraphRAG comparison. It distinguishes standard RAG's chunk-retrieval limits, GraphRAG's explicit entity relationships, DIGIMON-style operator vocabularies, StructGPT-style structured reasoning, and KGAS's theory-driven graph construction. [1]

The full August draft expands this into a more detailed theory-first architecture. It explicitly compares standard RAG/GraphRAG query-retrieve-generate flow with KGAS's theory-selection, theory-extraction, and multi-modal analysis flow. It also adds a two-layer architecture separating one-time theory extraction from repeated question-driven application. [2]

The final rewrite keeps the same conceptual comparison but compresses it. It removes some tool-suite detail and changes the tone from comprehensive capability breadth toward proposal readability. It still presents KGAS as theory-first infrastructure, but it moves more technical detail into annexes and makes validation framing more prominent. [3][5]

# Dataset And Case-Study Delta

The older proposal includes COVID discourse but also retains UFO discourse as a second case study. It mentions the COVID dataset eight times and UFO five times. [1]

The Prateek critique asks for more detail on dataset structure, filtering, evaluation strategy, Concordia/Park-style ABM, ABM calibration, and offline-behavior risk. [4]

The final rewrite centers the Kunst COVID dataset much more clearly: 2,506 users, psychological profiles, and validation against conspiracy mentality and related characteristics. It removes UFO references from the main proposal and turns offline-behavior limitations into an explicit scope boundary. [3][5]

Term-count scan supports the shift:

| Term | Older proposal | Full August draft | Final rewrite |
| --- | ---: | ---: | ---: |
| COVID | 8 | 15 | 19 |
| Kunst | 2 | 5 | 6 |
| UFO | 5 | 0 | 0 |
| Concordia | 1 | 2 | 2 |
| ABM | 5 | 7 | 3 |

# Validation Delta

The older proposal includes an extraction-performance target: F1 scores above 0.80 for key entity and relationship extraction tasks. It also frames SME tool evaluation as a central empirical question. [1]

The full August draft and final rewrite move toward baseline establishment. The full draft says validation should establish performance baselines for a novel methodology because many capabilities lack established benchmarks. The final rewrite says construct correlations, entity extraction, cross-modal coherence, system capability assessment, and process metrics should be reported without predetermined targets. [2][3]

The final-submission summary confirms the shift: validation uses crowd coding for extraction accuracy, correlation with psychological scales, no predetermined performance targets, and uncertainty propagation with IC standards. [5]

The final version is therefore stronger for academic defensibility, but the older version preserves a useful engineering instinct: extraction accuracy should have concrete operational metrics. The correct synthesis is not to discard that older metric impulse; it is to label such targets as optional process baselines rather than dissertation success promises.

# Terminology Delta

The critique and final summaries show deliberate terminology stabilization:

- "computational proxy" and "construct score" became "construct estimate";
- "hypothesis" became "research question";
- "improve" became "characterize improvements";
- "lossless" became "information-preserving conversions between formats within theoretical constructs of interest";
- SME validation was removed or deemphasized in favor of crowd coding and construct validation. [4][5][6]

Term-count scan also shows this:

| Term | Older proposal | Full August draft | Final rewrite |
| --- | ---: | ---: | ---: |
| construct estimate | 0 | 1 | 6 |
| computational proxy | 2 | 0 | 0 |
| SME | 27 | 27 | 11 |
| Subject Matter Expert | 5 | 1 | 0 |
| predetermined | 0 | 2 | 5 |
| baseline | 3 | 10 | 7 |

# What The Older Proposal Preserves

The older proposal is not obsolete junk. It preserves:

- the SME-facing/low-code usability motivation;
- a stronger policy-analysis framing around strategic foresight and decision support;
- an explicit investigative-workbench framing;
- detailed comparison to RAG, GraphRAG, DIGIMON, StructGPT, and traditional computational social science;
- early extraction-metric instincts;
- the broader case-study horizon before the final COVID/Kunst narrowing. [1]

For provenance, the older proposal is especially useful for understanding what was lost or moved during compression: usability ambitions, UFO/fringe-discourse breadth, and a more direct policy-tool posture.

# What The Final Rewrite Improves

The final rewrite is better for the dissertation submission task because it:

- centers theory-aware computational social science rather than tool-building alone;
- uses research questions instead of a broad core hypothesis;
- foregrounds construct-estimate validation;
- removes UFO material from the main claim surface;
- turns GraphRAG comparison into one part of a theory-first positioning;
- reduces page count and moves detail to annexes;
- avoids predetermined performance targets and stronger measurement claims. [3][5][6]

# Open Caveats

- This review did not perform a full semantic diff across all proposal text and annexes.
- `Mills_Proposal_extracted.txt` is not empty despite reporting zero newline-delimited lines in `wc -l`. A binary-aware inspection found an 85,062-byte UTF-8 text file stored as one long line, with no LF/CR/NUL bytes; it includes the proposal title, table-of-contents material, KGAS/GraphRAG terms, timeline material, and references. The matching `ARCHIVE_BEFORE_CLEANUP_20250805/.../Mills_Proposal_extracted.txt` copy has the same SHA-256 hash (`f0db826411bfe03806e525b0182ccd69447bb276e94b2c38cf6cae74d84d9115`). This review records the file-format/provenance finding but still has not done a full semantic comparison of that extracted text. [7][8]
- The duplicate `ARCHIVE_BEFORE_CLEANUP_20250805/.../proposal_old/` tree appears to mirror the theoretical-exploration `proposal_old/` file set by size and path names, but this review only hashed the `Mills_Proposal_extracted.txt` duplicate pair.
- Annexes were inventoried and partially represented via summaries; a future slice should compare annex content, especially validation, HSPC, theory meta-schema, and technical architecture.

# Relationship To Wiki

- [Proposal Framing Evolution](/wiki/concepts/proposal-framing-evolution.md): cross-source concept-level review that this page concretizes.
- [Digimon Lineage Theoretical Exploration Proposal Evolution](/wiki/sources/digimon-lineage-theoretical-exploration-proposal-evolution.md): parent source summary.
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md): scope and tone guardrail.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): validation-level guardrail.
- [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md): GraphRAG/DIGIMON comparison context.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/1/47_page_older_proposal.txt`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/fragments/proposal_full_2025.08061424.txt`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/proposal_rewrite.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/fragments/prateek_critique.txt`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/final_submission_summary.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/final_revisions_summary.md`
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/2/Mills_Proposal_extracted.txt`
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/proposal_rewrite_20250812_about_to_drop_all_complex_uncerainty/proposal_old/2/Mills_Proposal_extracted.txt`
