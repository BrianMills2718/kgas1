---
type: SourceSummary
title: Digimon Lineage Agent Stress Testing
description: Agent stress-testing archive covering dual-agent coordination, adaptive planning demos, MCP/KGAS/Claude integration attempts, stress-test scripts, traces, and proof-of-concept tool execution evidence.
tags: [source, digimon-lineage, archive, agents, stress-testing, mcp, kgas, adaptive-planning]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/
confidence: high
---

# Summary

`archive/agent_stress_testing/` is a 52-file, 1.1 MB archive of agent stress-testing scripts, demos, configs, sample data, and trace outputs. Its aggregate content-manifest hash is `5d462d369ac0dd778a1eea80c791bf6f365a8d7c361cd9faf9f20864f2aced94`. [1]

The archive targets a dual-agent research architecture: a research/planning agent and an execution/tool agent coordinating KGAS workflows, memory integration, MCP tools, Claude CLI/Code integration, and adaptive workflow correction. [2]

# Inventory

| Area | Examples | Role |
| --- | --- | --- |
| Root demos/scripts | `real_mcp_adaptive_agents.py`, `real_kgas_integration.py`, `working_mcp_client.py`, `comprehensive_stress_runner.py`, `adversarial_input_test.py` | Main demo and stress-test implementations. [1] |
| Summaries | `README.md`, `WORKING_SYSTEM_SUMMARY.md`, `ADAPTIVE_AGENT_DEMO_SUMMARY.md` | Framework goals and asserted working-system/adaptive-demo claims. [2] |
| Integration subdirs | `claude_code_integration/`, `dual_agent_tests/`, `memory_integration_tests/`, `workflow_execution_tests/`, `research_scenario_tests/` | Focused test harnesses for coordination, memory, workflow, and scenario testing. [1] |
| Config/data | `config/`, `stress_test_configs/`, `demo_data/`, `test_data/` | Agent prompts, sample documents, and predefined research scenarios. [1] |
| Trace/result JSON | `adaptive_demo_results.json`, `advanced_adaptive_demo_results.json`, `proof_of_concept_*.json`, `real_mcp_trace_*.json`, `trace_output_*.json` | Preserved execution traces and demo outputs. [1] |

# Intended Test Framework

The README defines five test categories: dual-agent coordination, memory integration, workflow execution, Claude Code integration, and research scenario tests. It also names success metrics for agent efficiency, memory utilization, workflow success rate, error recovery, research quality, context consistency, and scalability. [2]

This is partly aspirational framework design: the README lists tests that are not all present as files, such as several context-switching, memory, and workflow tests. The archive nevertheless contains many concrete scripts and traces.

# Working-System Claims

`WORKING_SYSTEM_SUMMARY.md` claims the system became fully operational with real MCP tool integration, six real KGAS MCP tools, document processing, spaCy NER, chunking, relationship extraction, quality assessment, and about 0.9 seconds per document processing time. It also claims 6/6 MCP tools connected and 100% success rate in testing. [3]

These claims are important historical evidence, but they should be treated as demo/report claims unless corroborated by the specific trace outputs and current-code verification.

# Concrete Proof Artifact

`proof_of_concept_proof_demo_20d08680_20250724_001434.json` is the strongest small proof artifact in this slice. It records a 2.216-second execution using T15A Text Chunker and T23A spaCy NER, reports 12 spaCy entities found, one chunk created, functional KGAS integration, operational service manager, working tool request pattern, async execution, and robust error handling. [4]

The raw entities include company/person/location examples such as Apple Inc., Tim Cook, Microsoft Corporation, Satya Nadella, Cupertino, California, Sundar Pichai, Amazon, Andy Jassy, and Seattle. [4]

# Adaptive-Agent Demos

`ADAPTIVE_AGENT_DEMO_SUMMARY.md` describes a dual-agent adaptive planning demonstration with strategic multi-path planning, seven adaptation strategies, quality trend analysis, resource awareness, and learning-pattern summaries. It explicitly frames the agents as demonstrating "genuine intelligence, not just automation." [5]

The JSON demo outputs show planned workflows and simulated or demo execution results for academic document analysis, including document processing, NER, relationship extraction, network analysis, fallback strategies, quality scores, and adaptation traces. [6]

Because the adaptive summaries mix simulated tool behavior, real integration readiness, and strong claims, they are best read as architecture/demo evidence rather than proof of production agent intelligence.

# Credential Scan

A targeted scan of this agent-stress-testing archive found no literal OpenAI or Google API keys. [1]

# Interpretation

This archive preserves the shift from static KGAS tool execution toward agentic workflow orchestration: planning, execution, adaptation, tracing, and learning loops. The best-supported evidence is the small proof demo that ran real chunking and spaCy NER. The broader adaptive-agent claims are valuable as design and demo lineage, but should not be cited as production capability without replayable tests or current runtime verification.

# Relationship To Wiki

- [Digimon Lineage Analysis Validation 2025 08](digimon-lineage-analysis-validation-2025-08.md): related validation configs and claim-specific checks.
- [Digimon Lineage Generated Outputs 2025 08](digimon-lineage-generated-outputs-2025-08.md): related trace/provenance/generated-output artifacts.
- [MCP Autoloop Interface](../concepts/mcp-autoloop-interface.md): related MCP/autoloop lineage.
- [Multi Agent Evidence Harness](../concepts/multi-agent-evidence-harness.md): related multi-agent implementation/evaluation pattern.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): relevant guardrail for separating demo claims from runtime proof.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): relevant because archived agent demos may not reflect current runtime state.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/README.md`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/WORKING_SYSTEM_SUMMARY.md`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/proof_of_concept_proof_demo_20d08680_20250724_001434.json`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/ADAPTIVE_AGENT_DEMO_SUMMARY.md`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/adaptive_demo_results.json` and `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/agent_stress_testing/advanced_adaptive_demo_results.json`
