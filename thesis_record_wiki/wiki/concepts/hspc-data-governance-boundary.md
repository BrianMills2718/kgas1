---
type: Concept
title: HSPC Data Governance Boundary
description: Preservation review of KGAS dissertation human-subjects, privacy, SME, and data-governance boundaries across proposal text, Annex D, and preserved RAND HSPC reference materials.
tags: [concept, hspc, ethics, privacy, proposal, governance, dissertation, provenance]
created: 2026-06-26
updated: 2026-06-26
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/hspc_summary.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/hspc.pdf
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/hspc-standard-operating-procedures.pdf
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/RAND_MG654-1.pdf
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/Use Social Media.html
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/1/47_page_older_proposal.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/annex_d_human_subjects_protection.txt
confidence: high
---

> Sources consulted: `hspc_summary.txt` · HSPC policy PDF inventory and hashes · saved `Use Social Media.html` filename/metadata only · older proposal ethical-considerations section · `annex_d_human_subjects_protection.txt`. Status: governance-boundary preservation review, not legal advice and not proof that an actual HSPC determination letter is present in the thesis record.

# Summary

The KGAS proposal materials preserve a consistent human-subjects boundary: the planned dissertation analyzes public or de-identified materials, does not interact with individuals, does not collect identifiable private information, and treats any systematic SME evaluation or restricted/private data integration as a trigger for HSPC review. [1][6][7]

The final Annex D is the strongest governance artifact because it expands the older ethical-considerations paragraph into a structured boundary: data sources, identity-protection measures, informal SME limits, vulnerable-population exclusions, consent contingencies, modification triggers, LLM usage constraints, and documentation requirements. [7]

The preserved HSPC folder is valuable as reference provenance, but it does not by itself prove approval or determination. The wiki should therefore distinguish:

- proposal reasoning about why the work was expected to be "not human subjects research";
- preserved policy/reference materials used to shape that reasoning;
- missing or not-yet-located determination evidence.

# Preserved HSPC Reference Set

The preserved HSPC material under `proposal_materials/academic/hspc/hspc/` contains:

| File | Size | Preservation role |
| --- | ---: | --- |
| `hspc_summary.txt` | 10,574 bytes | Local proposal statement, fuller HSPC rationale, and ChatGPT review of reviewed policy categories. [1] |
| `hspc.pdf` | 172,984 bytes | Preserved policy reference. This review records the file, not a full PDF content analysis. [2] |
| `hspc-standard-operating-procedures.pdf` | 1,037,095 bytes | Preserved SOP reference. This review records the file, not a full PDF content analysis. [3] |
| `RAND_MG654-1.pdf` | 932,933 bytes | Preserved RAND reference PDF in the HSPC bundle. This review records the file, not a full PDF content analysis. [4] |
| `Use Social Media.html` | 1,863,214 bytes | Saved HSPC/social-media guidance page. This review uses only filename/metadata because the HTML is a large intranet capture. [5] |

Hashes were recorded during this slice for future duplicate/provenance checks, but the source files were not modified.

# Boundary In The Proposal

The older proposal already included the main ethical boundary. It states that the project would analyze publicly available documents and online content, would not collect identifiable private information, and would confine any SME feedback to internal RAND staff without identifiable or sensitive data. It also says HSPC screening should occur before execution and that systematic SME data collection for generalizable knowledge would require modification and review. [6]

This matters because SME-centered validation appears elsewhere in older proposal text. The ethical section constrains that idea: SME input is not automatically available as research evidence unless the review posture changes.

# Boundary In Annex D

Annex D makes the boundary more operational. It records:

- current classification as not human subjects research under the stated design;
- primary reliance on public/de-identified data, public social-media content, academic literature, and public documents;
- no de-anonymization, triangulation, PII linking, credential harvesting, or persistent user-identifier storage;
- internal/informal SME consultation only, with no systematic data collection or identifiable feedback recording;
- HSPC modification triggers for systematic SME evaluation, restricted datasets, external participants, identifiable information, or funding/IRB assurance;
- future consent/CITI/RHINO requirements if the scope changes. [7]

That makes Annex D a governance guardrail, not just an appendix.

# Caveats

- No actual HSPC determination letter was identified in this bounded slice. A targeted filename and text search found HSPC reference materials, Annex D duplicates, and proposal text naming "HSPC determination letter" as a planned deliverable, but not a preserved determination-letter artifact. Two archived service-data paths under the April 2026 filesystem snapshot returned permission-denied errors, so the search is strong but not absolutely exhaustive. The record currently supports "proposal reasoning and reference materials exist," not "determination letter found."
- The preserved HTML is a large saved intranet page. This review records its existence and hash but does not summarize internal page content.
- The HSPC summary contains an AI-generated review section. Use that as process provenance, not as authoritative policy.
- The proposal's public/de-identified-data claim should be rechecked before any public release or new analysis using raw datasets.

# Practical Rule

For future thesis recovery or restart work:

1. Treat public/de-identified data analysis as the baseline only if the specific dataset still matches that description.
2. Treat systematic SME evaluation as a governance trigger, not a casual validation shortcut.
3. Treat HSPC summary and Annex D as proposal artifacts until an actual determination letter is found.
4. Treat public export as a separate privacy/security review even when the dissertation design says "not human subjects research."

# Links

- [Proposal Validation Evolution](/wiki/concepts/proposal-validation-evolution.md)
- [Final Proposal Annex Preservation Review 2026 06 26](/wiki/sources/final-proposal-annex-preservation-review-2026-06-26.md)
- [Proposal Old To Final Comparison 2026 06 26](/wiki/sources/proposal-old-to-final-comparison-2026-06-26.md)
- [Public Export Security Boundary](/wiki/concepts/public-export-security-boundary.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/hspc_summary.txt`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/hspc.pdf`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/hspc-standard-operating-procedures.pdf`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/RAND_MG654-1.pdf`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/proposal_materials/academic/hspc/hspc/Use Social Media.html`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/1/47_page_older_proposal.txt`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/theoretical_exploration/archived_proposal_materials/historical_versions/proposal_old/3/annex_d_human_subjects_protection.txt`
