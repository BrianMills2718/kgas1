---
type: Concept
title: Analysis Expansion Architecture
description: KGAS architecture expanded from graph extraction toward cross-modal analysis, schema paradigms, ABM simulation, statistics, and local automation interfaces.
tags: [concept, cross-modal, analysis, statistics, abm, schema, api]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-006-cross-modal-analysis.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-015-Cross-Modal-Orchestration.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-020-Agent-Based-Modeling-Integration.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-021-Statistical-Analysis-Integration.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-023-Comprehensive-Schema-Modeling-Ecosystem.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-026-Cross-Modal-REST-API.md
confidence: medium
---

# Summary

Analysis expansion architecture is the KGAS thesis ambition to become more than a graph-building system. The architecture tries to let a researcher move between graph, table, vector, schema, simulation, and statistical modes while preserving identity, provenance, and theory grounding.

# Main Expansion Axes

- **Cross-modal analysis**: synchronized graph/table/vector views over shared entities. [1][2]
- **Schema paradigms**: multiple ways to represent the same political or theoretical fact, including RDF/OWL, UML, ORM, TypeDB-style ER, and n-ary graph schemas. [5]
- **ABM simulation**: translate theory schemas into agent configurations to test counterfactuals and emergent behavior. [3]
- **Statistical analysis**: add SEM, factor analysis, regression, causal inference, multivariate methods, and statistical reporting. [4]
- **Local automation API**: expose high-level analysis, conversion, recommendation, and batch workflows over localhost FastAPI. [6]

# Thesis Interpretation

This thread shows KGAS moving from "extract a graph from documents" toward "operationalize social theory across multiple analytical representations." That is a stronger thesis vision, but also a much larger verification burden.

The wiki should therefore separate:

- conceptual architecture and research ambition
- code implementation claims inside ADRs
- independent evidence that a capability worked in a real workflow
- current status in the cleaned repo

# Links

- [Digimon Lineage Analysis Expansion ADRs](/wiki/sources/digimon-lineage-analysis-expansion-adrs.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Graph Build Manifest](/wiki/concepts/graph-build-manifest.md)
- [Complexity Accuracy Pattern](/wiki/concepts/complexity-accuracy-pattern.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-006-cross-modal-analysis.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-015-Cross-Modal-Orchestration.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-020-Agent-Based-Modeling-Integration.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-021-Statistical-Analysis-Integration.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-023-Comprehensive-Schema-Modeling-Ecosystem.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-026-Cross-Modal-REST-API.md`
