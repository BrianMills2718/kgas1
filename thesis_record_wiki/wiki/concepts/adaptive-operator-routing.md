---
type: Concept
title: Adaptive Operator Routing
description: DIGIMON thesis thread that agents should compose graph retrieval operators per question instead of using a fixed pipeline.
tags: [concept, digimon, operator-routing, evaluation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_autoloop/README.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/plans/03_prove_adaptive_routing.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/reports/2026-03-19_musique_50q_postmortem.md
confidence: medium
---

# Summary

Adaptive operator routing is the later DIGIMON research thesis: agents should select and compose retrieval operators per question, rather than forcing all questions through one fixed GraphRAG pipeline. [1][2]

The autoloop variant makes this thesis concrete with 28 typed operators and agent-composed retrieval DAGs. It also makes the evaluation burden concrete: adaptive routing must beat both a non-graph baseline and the best fixed graph comparator under a locked evaluation protocol. [1][2]

# Decision Gate

Plan #3 defines two gates:

- Graph value: the best fixed graph pipeline must beat the non-graph baseline by at least 2 EM or 3 LLM_EM. [2]
- Adaptive value: adaptive routing must beat the best fixed graph pipeline by at least 3 EM or 5 LLM_EM. [2]

The plan also adds guardrails for completion rate, cost, and latency. This matters because adaptive routing can appear attractive while consuming substantially more tool calls, time, and model budget. [2]

# Current Evidence State

The 2026-03-18/19 MuSiQue 50-question development comparison does not support the thesis:

- baseline: 34.0 EM, 60.0 LLM_EM, $2.03
- fixed graph: 32.0 EM, 54.0 LLM_EM, $1.85
- hybrid/adaptive: 32.0 EM, 44.0 LLM_EM, $5.50

The postmortem notes a provider-fallback confound in the hybrid run, but also says hybrid showed no first-half upside before the confounder. Its recommendation is one constrained salvage iteration, not a locked eval or open-ended tuning. [3]

# Relationship To Thesis Record

This concept is central because it converts the broad KGAS/Digimons engineering arc into an explicit falsifiable research question. The record should preserve both the ambition and the negative evidence. Future summaries should not say "adaptive routing worked" unless they cite a later locked evaluation that passes the plan gates.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_autoloop/README.md`  
[2] `../archive_full_record/lineage_variants/digimon_autoloop/docs/plans/03_prove_adaptive_routing.md`  
[3] `../archive_full_record/lineage_variants/digimon_autoloop/docs/reports/2026-03-19_musique_50q_postmortem.md`
