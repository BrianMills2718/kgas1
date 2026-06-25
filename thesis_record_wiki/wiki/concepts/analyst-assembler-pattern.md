---
type: Concept
title: Analyst Assembler Pattern
description: Workflow pattern that assigns semantic interpretation to the LLM and deterministic schema assembly/validation to code.
tags: [concept, workflow, llm, schema, deterministic-assembly]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/overview.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/post_processing.md
confidence: high
---

# Summary

The Analyst Assembler pattern splits theory-schema generation into two responsibilities:

- the LLM acts as analyst, reading and interpreting theory content into a concise structured blueprint
- deterministic code acts as assembler, injecting shared definitions, enforcing schema shape, and producing the final valid artifact

The legacy lit-review system states this explicitly as the "Analyst & Assembler" workflow. [1]

# Why It Matters

This pattern is a recurring design lesson in the thesis work. It avoids asking the LLM to simultaneously perform difficult interpretation and exact boilerplate/schema construction. It also makes common ontology injection and schema validity programmatic rather than conversational.

# Links

- [Lit Review Legacy System Framing](/wiki/sources/lit-review-legacy-system-framing.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Complexity Conservation In Theory Application](/wiki/concepts/complexity-conservation-in-theory-application.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/overview.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/post_processing.md`
