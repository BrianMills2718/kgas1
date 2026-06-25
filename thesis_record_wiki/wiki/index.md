# Overview

* [Overview](overview.md) - top-level map of the thesis work record and preservation strategy.

# Sources

* [Recovery Archive Manifest](sources/recovery-archive-manifest-2026-04-04.md) - why `archive_full_record/` exists and what it preserves.
* [Recovery Inventory](sources/recovery-inventory.md) - preservation metadata, sizes, file counts, git heads, and verification errors.
* [Current Repo Context](sources/current-repo-context.md) - current README, CLAUDE notes, branch, and git history baseline.
* [Digimons Documentation Repository](sources/digimons-docs-documentation-repository.md) - documentation-only lineage variant and its source-of-truth split.
* [Digimons Minimal Clean Reference](sources/digimons-minimal-clean-reference.md) - compact 2025-09-04 clean KGAS reference line.
* [Tool Compatibility Decision](sources/tool-compatibility-decision.md) - type-based composition decision and POC context.
* [JayLZhou GraphRAG Upstream](sources/jaylzhou-graphrag-upstream.md) - external DIGIMON / GraphRAG reference repository.
* [Digimon Core Sparse Contract Layer](sources/digimon-core-sparse-contract-layer.md) - sparse variant contract-first migration, evidence, and bottleneck findings.
* [Digimon Autoloop Operator Routing](sources/digimon-autoloop-operator-routing.md) - later operators-first DIGIMON state, benchmark evidence, and supported-surface boundary.

# Entities

* [KGAS](entities/kgas.md) - Knowledge Graph Analysis System, the thesis implementation line.

# Concepts

* [Full Record Preservation](concepts/full-record-preservation.md) - policy for preserving the complete messy historical record before cleanup.
* [Research Lineage](concepts/research-lineage.md) - how Digimons variants, KGAS, archives, and cleaned repo relate.
* [Verification Gaps](concepts/verification-gaps.md) - known places where the record is incomplete, permission-limited, or needs later review.
* [Documentation Status Truthfulness](concepts/documentation-status-truthfulness.md) - recurring attempt to separate target architecture from verified implementation status.
* [Type-Based Tool Composition](concepts/type-based-tool-composition.md) - later compatibility strategy for turning incompatible tools into composable typed operators.
* [GraphRAG Upstream Lineage](concepts/graphrag-upstream-lineage.md) - how local Digimon/KGAS work relates to JayLZhou GraphRAG and DIGIMON method decomposition.
* [Contract-First Migration](concepts/contract-first-migration.md) - migration from split tool interfaces toward KGASTool contracts.
* [Relationship Extraction Bottleneck](concepts/relationship-extraction-bottleneck.md) - evidence that graph construction failed when relationship extraction was missing or silent.
* [Adaptive Operator Routing](concepts/adaptive-operator-routing.md) - falsifiable thesis that per-question operator routing should beat fixed graph and non-graph baselines.
* [MCP Autoloop Interface](concepts/mcp-autoloop-interface.md) - historical MCP/UKRF plans and later MCP tool-surface status.
* [Graph Build Manifest](concepts/graph-build-manifest.md) - build-manifest and tool-gating contract for truthful graph capability exposure.

# Timeline

* [Evolution Timeline](timeline/evolution-timeline.md) - initial timeline from git history and recovery metadata.

# Lineage Variants

* [Current Clean Repo](variants/current-clean-repo.md) - tracked repo state after cleanup and recovery.
* [Filesystem Snapshot 2026-04-04](variants/filesystem-snapshot-2026-04-04.md) - moved snapshot preserving pre-cleanup archive-only material.
* [Digimons Minimal](variants/digimons-minimal.md) - local repo matching `kgas1` `origin/master` commit `2c59a1f`.
* [Digimons Clean For Real](variants/digimons-clean-for-real.md) - cleanup-focused branch history snapshot.
* [Digimons Old](variants/digimons-old.md) - large pre-KGAS / earlier Digimons archive.
* [Digimon Core Sparse](variants/digimon-core-sparse.md) - sparse / contract-oriented variant with many evidence documents.
* [Digimons Docs](variants/digimons-docs.md) - documentation extraction variant.
* [Digimon Lineage Digimons](variants/digimon-lineage-digimons.md) - largest preserved lineage bundle.
* [Digimon v2](variants/digimon-v2.md) - later Digimon lineage repo.
* [Digimon Autoloop](variants/digimon-autoloop.md) - autoloop-related lineage variant.
