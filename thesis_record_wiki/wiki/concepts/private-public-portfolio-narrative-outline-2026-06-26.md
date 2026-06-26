---
type: Concept
title: Private Public Portfolio Narrative Outline 2026 06 26
description: Private draft outline for a future public/portfolio narrative about KGAS, theory-aware GraphRAG, schema extraction, uncertainty, and evidence discipline.
tags: [concept, draft, portfolio, public-narrative, kgas, thesis]
created: 2026-06-26
updated: 2026-06-26
status: private-draft-not-for-publication
sources:
  - /wiki/concepts/thesis-artifact-decision-brief-2026-06-26.md
  - /wiki/concepts/kgas-dissertation-claim-map.md
  - /wiki/concepts/theory-schema-application-lineage.md
  - /wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md
  - /wiki/concepts/graphrag-upstream-lineage.md
  - /wiki/concepts/evidence-claim-discipline.md
  - /wiki/concepts/public-export-security-boundary.md
confidence: high
---

> Sources consulted: [Thesis Artifact Decision Brief 2026 06 26](/wiki/concepts/thesis-artifact-decision-brief-2026-06-26.md) · [KGAS Dissertation Claim Map](/wiki/concepts/kgas-dissertation-claim-map.md) · [Theory Schema Application Lineage](/wiki/concepts/theory-schema-application-lineage.md) · [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md) · [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md) · [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md) · [Public Export Security Boundary](/wiki/concepts/public-export-security-boundary.md). Status: private draft outline for Brian review, not publication copy.

# Status

This is a private draft outline. It is not approved for publication, not a public export, and not a claim that the raw thesis repository is public-ready.

# Recommended Frame

Use this public-facing frame:

> KGAS explored how GraphRAG-style systems could become theory-aware research infrastructure: extracting social-science theories into executable schemas, applying them to discourse, comparing graph/table/vector representations, and preserving evidence and uncertainty discipline.

This frame is accurate because it presents KGAS as a research program and method lineage, not as a finished product or completed dissertation validation. [1][2][3][4]

# Possible Titles

1. **Making GraphRAG Theory-Aware**
2. **From Social-Science Theory to Executable Analysis**
3. **KGAS: A Research Lineage in Theory-Aware GraphRAG**
4. **Evidence Discipline for AI-Assisted Social Science**

Recommended title: **Making GraphRAG Theory-Aware**. It is concrete, avoids overclaiming, and ties the work to a recognizable technical category while leaving room for the social-science contribution.

# Audience

Primary audience:

- AI/retrieval engineers interested in GraphRAG beyond generic document QA;
- computational social scientists interested in theory operationalization;
- research-tool builders interested in evidence/provenance discipline.

Secondary audience:

- portfolio reviewers;
- collaborators who need to understand why the archive matters;
- future academic readers if Brian reopens dissertation-style writing.

# Outline

## 1. The Problem

Most retrieval systems can find relevant text, but social-science research needs more than retrieval. It needs theories, constructs, operational definitions, evidence quality, and explicit uncertainty.

Claim boundary: do not say ordinary GraphRAG is useless. Say KGAS explored what has to be added when the goal is theory-grounded analysis rather than answer generation.

Sources: [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md), [KGAS Dissertation Claim Map](/wiki/concepts/kgas-dissertation-claim-map.md)

## 2. The Core Idea

KGAS tried to turn this chain into infrastructure:

```text
paper -> theory schema -> model-form routing -> discourse application -> comparable outputs -> validation evidence
```

The key move was to treat theories as operational artifacts, not just background reading.

Sources: [Theory Schema Application Lineage](/wiki/concepts/theory-schema-application-lineage.md)

## 3. What Made It Different

Three differentiators:

1. **Theory schemas:** extract constructs, vocabulary, formal notation, algorithms, and application rules from papers.
2. **Model-form routing:** choose graph/table/vector/sequence/statistical/hybrid forms based on the theory.
3. **Evidence discipline:** separate design intent, historical evidence, current code, runtime proof, and validation claims.

Sources: [Theory Schema Application Lineage](/wiki/concepts/theory-schema-application-lineage.md), [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

## 4. What The Work Proved

Safe public claim:

- the research lineage produced concrete schema/application artifacts;
- the current recovered runtime has bounded local proof for selected document-analysis paths;
- the wiki preserves the design evolution and evidence boundaries.

Do not claim:

- the dissertation was completed or accepted;
- KGAS is fully validated;
- all historical architecture claims are current runtime behavior;
- raw archives are public-ready.

Sources: [KGAS Dissertation Claim Map](/wiki/concepts/kgas-dissertation-claim-map.md), [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

## 5. Uncertainty And Why It Matters

The uncertainty story is a feature, not a defect. The project moved through several designs: normalized confidence, CERQual, entity-resolution uncertainty, IC-informed ADR-029/Comprehensive7, stress-test branches, and finally a stronger habit of labeling claim levels.

Public phrasing:

> The most important lesson was not one perfect uncertainty formula; it was the need to keep uncertainty reasoning auditable and tied to the level of evidence actually available.

Sources: [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md)

## 6. What I Would Build Next

Potential future-facing section:

1. a small theory-schema workbench;
2. schema-to-application examples on public texts;
3. a validation harness that reports baselines instead of chasing a target;
4. a public, sanitized demo that does not expose raw thesis archives.

Claim boundary: present this as future direction, not completed system status.

Sources: [Thesis Artifact Decision Brief 2026 06 26](/wiki/concepts/thesis-artifact-decision-brief-2026-06-26.md)

## 7. Why The Archive Exists

The private archive is intentionally complete. It preserves false starts, superseded designs, validation tensions, and old ambitions because those explain the work.

Public phrasing:

> I kept the full research record private and messy, then built a wiki layer to make it navigable. Public artifacts should be derived from that record, not expose it wholesale.

Sources: [Thesis Artifact Decision Brief 2026 06 26](/wiki/concepts/thesis-artifact-decision-brief-2026-06-26.md), [Public Export Security Boundary](/wiki/concepts/public-export-security-boundary.md)

# Suggested Short Version

> KGAS was my attempt to make GraphRAG theory-aware. Instead of treating documents as chunks to retrieve from, the system explored how social-science theories could be extracted into schemas, routed into suitable model forms, applied to discourse, and evaluated with explicit evidence and uncertainty boundaries. The most valuable result is not a finished product claim; it is a preserved research lineage showing what theory-aware analysis infrastructure requires: vocabulary preservation, model-form routing, executable schemas, cross-modal comparison, and disciplined separation between design intent, historical evidence, current runtime proof, and validation.

# Review Checklist Before Any Public Use

- Remove or avoid raw archive paths unless they are already in a reviewed public export.
- Do not mention raw tweet text, identifiers, credentials, local passwords, `.env` files, or detailed security-risk summaries.
- Do not claim HSPC approval.
- Do not claim dissertation completion.
- Do not say KGAS is production-ready.
- Do not say ADR-029/Comprehensive7 is current runtime behavior.
- Decide whether to mention leaving the thesis program, or frame only the research lineage.
- Decide whether the public artifact is a personal portfolio page, technical blog post, academic preprint, or project README.

# Next Private Draft

The next safe artifact is a one-page private draft using this outline. It should live in the repo as draft text and remain unpublished until Brian reviews tone, privacy, and audience.

# Links

- [Thesis Artifact Decision Brief 2026 06 26](/wiki/concepts/thesis-artifact-decision-brief-2026-06-26.md)
- [KGAS Dissertation Claim Map](/wiki/concepts/kgas-dissertation-claim-map.md)
- [Theory Schema Application Lineage](/wiki/concepts/theory-schema-application-lineage.md)
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Public Export Security Boundary](/wiki/concepts/public-export-security-boundary.md)

# Citations

[1] `/wiki/concepts/thesis-artifact-decision-brief-2026-06-26.md`  
[2] `/wiki/concepts/kgas-dissertation-claim-map.md`  
[3] `/wiki/concepts/theory-schema-application-lineage.md`  
[4] `/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md`  
[5] `/wiki/concepts/graphrag-upstream-lineage.md`  
[6] `/wiki/concepts/evidence-claim-discipline.md`  
[7] `/wiki/concepts/public-export-security-boundary.md`
