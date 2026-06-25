---
type: Concept
title: Research Lineage
description: Relationship between early Digimons experiments, KGAS, cleaned repo states, and recovered archive variants.
tags: [lineage, digimons, kgas]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
  - ../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md
confidence: medium
---

# Summary

The thesis work should be understood as a lineage, not as one repo. The clean tracked checkout is one curated endpoint. The full record includes earlier Digimons archives, later Digimon lineage repos, sparse/contract-oriented variants, documentation extraction variants, autoloop experiments, and a filesystem snapshot of archive-only material.

# Initial Lineage Buckets

- [Current Clean Repo](/wiki/variants/current-clean-repo.md)
- [Filesystem Snapshot 2026-04-04](/wiki/variants/filesystem-snapshot-2026-04-04.md)
- [Digimons Old](/wiki/variants/digimons-old.md)
- [Digimon Lineage Digimons](/wiki/variants/digimon-lineage-digimons.md)
- [Digimon v2](/wiki/variants/digimon-v2.md)
- [Digimon Core Sparse](/wiki/variants/digimon-core-sparse.md)
- [Digimons Minimal](/wiki/variants/digimons-minimal.md)
- [Digimons Clean For Real](/wiki/variants/digimons-clean-for-real.md)
- [Digimons Docs](/wiki/variants/digimons-docs.md)
- [Digimon Autoloop](/wiki/variants/digimon-autoloop.md)

# Emerging Through-Lines

Two through-lines are now visible from the first small-variant ingest:

- [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md): repeated attempts to separate target architecture from verified implementation status.
- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md): a later tool-compatibility strategy using semantic data types and exact schemas.
- [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md): Brian's note that much local Digimon work extends or forks JayLZhou GraphRAG / DIGIMON.
- [Contract-First Migration](/wiki/concepts/contract-first-migration.md): the sparse variant's formal migration from split interfaces to KGASTool contracts.
- [Relationship Extraction Bottleneck](/wiki/concepts/relationship-extraction-bottleneck.md): stress-test evidence that the graph layer could fail by extracting entities but no relationships.
- [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md): the autoloop variant's sharpened, falsifiable thesis that per-question operator selection should beat fixed pipelines.
- [MCP Autoloop Interface](/wiki/concepts/mcp-autoloop-interface.md): early MCP/UKRF integration planning preserved alongside later MCP/direct retrieval interfaces.
- [Graph Build Manifest](/wiki/concepts/graph-build-manifest.md): a later contract for exposing tools only when graph topology, attributes, and artifacts support them.
- [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md): the large lineage bundle's sequence of documentation corrections, overcorrections, and roadmap consolidation.
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md): the architectural split that made implementation status hard to summarize in one number.
- [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md): large-bundle copy of the type-based composition direction, tied to active vertical-slice work.
- [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md): architecture thread where uncertainty sophistication ran into provenance and implementation practicality gaps.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): evidence archive correction that separates component tests from system integration proof.
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md): lit-review experiment thread closest to the dissertation question of extracting theories from academic papers and applying them to data.
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md): architecture ADR thread defining KGAS as local academic proof-of-concept rather than enterprise production product.
- [Storage Architecture Evolution](/wiki/concepts/storage-architecture-evolution.md): scale-indexed storage thread moving from Qdrant removal to Neo4j/SQLite and later Neo4j/PostgreSQL.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md): confidence/uncertainty design thread from normalized fields to auditable local construct-mapping judgments.
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md): tool architecture thread reconciling raw implementations, internal contracts, and MCP exposure.
- [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md): research-platform expansion thread from graph extraction to cross-modal, schema, ABM, statistics, and local automation.
- [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md): procedural verification thread using isolated implementation and external evaluation agents.
- [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md): human-facing demo/inspection thread preserved through archived and recovered UI files.

# Open Interpretive Work

The current inventory identifies preserved units, but it does not explain the conceptual differences between all variants. Future ingest should sample README, CLAUDE, roadmap, evidence, and docs files from each variant and update these pages with a richer lineage narrative.

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`  
[2] `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`
