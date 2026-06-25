---
type: Source
title: Digimon Lineage Performance Tests
description: Inventory of the preserved tests/performance directory, including benchmark surfaces, async/parallel/load themes, and the caveat that test code is not benchmark output.
tags: [source, digimon-lineage, tests, performance, benchmarks, async, parallel, load, production-scale]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_agent_performance_benchmarks.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_performance_benchmark.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_sequential_vs_parallel.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_working_pipeline.py
confidence: medium
---

# Summary

`tests/performance/` is a 16-file, 320,008-byte performance-test corpus with local aggregate hash `3636729420c8146520f883f7a02f6b31e12153861ce92904398e35df7a41544c`. [1]

Unlike functional and reliability tests, this directory has no local README or instruction file in the preserved corpus. Its role is inferred from file names, docstrings, and code structure, so confidence is lower than slices backed by explicit local documentation.

The corpus defines benchmark surfaces for agent performance, analytics tools, async clients, collaborative load testing, orchestrator performance, parallel discovery, resource conflicts, production scale, sequential-vs-parallel execution, and working pipeline checks. [1]

# Inventory

| Area | Files | Bytes | Aggregate hash | Role |
|---|---:|---:|---|---|
| root test files | 16 | 320,008 | `3636729420c8146520f883f7a02f6b31e12153861ce92904398e35df7a41544c` | benchmark and performance test definitions |

# Benchmark Surfaces

Representative files:

- `test_agent_performance_benchmarks.py`: comprehensive agent-architecture benchmarking with realistic document sets, workload variation, system monitoring, single-document, batch, variable-size, collaborative, memory, and resource-efficiency benchmarks. [2]
- `test_performance_benchmark.py`: benchmark suite for text chunking, entity extraction, PageRank, and memory usage, with code to emit benchmark evidence files. [3]
- `test_sequential_vs_parallel.py`: direct sequential-vs-parallel timing comparison for multi-document processing, explicitly framed as real evidence rather than estimates. [4]
- `test_working_pipeline.py`: end-to-end working pipeline check for schema initialization, entity extraction, relationship extraction, persistence, and UI queries. [5]
- `test_production_scale.py`: production-scale database testing with realistic load/conditions by file docstring. [1]
- `test_collaborative_load_testing.py`: load testing for collaborative processing, stress, failure recovery, and scalability validation. [1]

# Performance Themes

The performance corpus focuses on:

- async client and async execution performance
- sequential vs parallel speedup
- orchestrator and pipeline performance
- analytics performance under real or realistic data
- resource conflict detection and parallel discovery
- production-scale database behavior
- agent/collaboration workloads
- memory, latency, throughput, and success-rate tracking [1][2]

This complements [Digimon Lineage Reliability Tests](/wiki/sources/digimon-lineage-reliability-tests.md): reliability asks whether the system behaves correctly and recovers under stress; performance asks how fast and resource-efficient it is under benchmarked workloads.

# Evidence Caveat

These are test and benchmark definitions, not preserved benchmark results. Some files contain code to generate evidence artifacts such as performance benchmark JSON/Markdown, but this source slice does not establish that those artifacts were generated or what the measurements were. [3]

Future work should ingest benchmark output files, CI artifacts, or fresh benchmark runs before citing concrete speedups, latencies, memory usage, or production readiness.

# Links

- [Digimon Lineage Tests Corpus](/wiki/sources/digimon-lineage-tests-corpus.md)
- [Digimon Lineage Functional Tests](/wiki/sources/digimon-lineage-functional-tests.md)
- [Digimon Lineage Reliability Tests](/wiki/sources/digimon-lineage-reliability-tests.md)
- [Digimon Lineage Scripts Corpus](/wiki/sources/digimon-lineage-scripts-corpus.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_agent_performance_benchmarks.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_performance_benchmark.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_sequential_vs_parallel.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/performance/test_working_pipeline.py`
