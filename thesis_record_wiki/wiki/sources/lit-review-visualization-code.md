---
type: Source
title: Lit Review Visualization Code
description: Inventory and interpretation of visualization scripts for Carter cognitive maps and Semantic Hypergraph instances.
tags: [source, lit-review, code, visualization, cognitive-map, semantic-hypergraph, networkx]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/create_ascii_network.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/visualize_iran_hypergraph_formal.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/visualize_sh_readable.py
confidence: high
---

# Summary

The preserved `src/visualization/` folder contains 9 Python scripts with aggregate hash `aa2ec84bd9436113fe5ad954d9f04f7ba762871298a055f45f43f830f52cc7f9`. [1]

This folder is the human-inspection layer for the lit-review experiment. It turns generated Carter cognitive maps and Semantic Hypergraph instances into ASCII reports, NetworkX/Matplotlib diagrams, and readable claim graphs.

# Structural Facts

| Script | Hash prefix | Role | Flags |
|---|---|---|---|
| `convert_and_visualize_debate.py` | `910b4a99d044` | converts debate instance material for visualization | hardcoded local path |
| `create_ascii_network.py` | `f5ee5b0bfa03` | creates a text report for Carter's cognitive map | hardcoded local path |
| `create_networkx_visualization.py` | `35c98f74540e` | creates a NetworkX image for Carter's cognitive map | NetworkX, Matplotlib, hardcoded local path |
| `visualize_carter_cognitive_map.py` | `de30a4fa0399` | visualizes Carter cognitive-map graph and summary | NetworkX, Matplotlib, hardcoded local path |
| `visualize_carter_text.py` | `a5091b1a0554` | creates Carter text visualization | Matplotlib, hardcoded local path |
| `visualize_iran_hypergraph.py` | `f575d78cf1cd` | visualizes Iran debate Semantic Hypergraph material | NetworkX, Matplotlib, hardcoded local path |
| `visualize_iran_hypergraph_formal.py` | `f91b89bda77b` | parses and plots formal Semantic Hypergraph notation | NetworkX, Matplotlib, hardcoded local path |
| `visualize_sh_instance.py` | `6139955f5d83` | visualizes YAML Semantic Hypergraph instances | NetworkX, Matplotlib |
| `visualize_sh_readable.py` | `177ff3966072` | creates readable labels and claim graphs for SH instances | NetworkX, Matplotlib |

Seven scripts import NetworkX and/or Matplotlib. Six contain hardcoded `/home/brian/...` paths, so this folder is best treated as historical inspection code rather than current portable tooling. [1]

# Carter Cognitive Map Visualizations

`create_ascii_network.py` loads `carter_young1996_faithful_analysis.yml`, groups relationships by Young-style categories, prints centrality-like connection counts, and includes Young structural measures such as size and connectedness. [2]

`create_networkx_visualization.py` builds a directed graph from Carter concepts and relationships. It colors key concepts separately, styles edges by relationship category, wraps labels, and renders a graph image. [3]

Together, these scripts show that the experiment did not stop at generating YAML. It attempted to inspect whether the extracted cognitive map was readable and whether relationship categories could be visually checked.

# Semantic Hypergraph Visualizations

`visualize_iran_hypergraph_formal.py` defines `HyperedgeParser`, parses Semantic Hypergraph notation such as typed atoms and nested relation/specifier structures, infers hyperedge types from connector types, and plots both complete and simplified graph views. [4]

`visualize_sh_readable.py` loads YAML SH instances, converts technical atom and hyperedge IDs into human-readable labels, identifies main claims and sub-claims, and creates a readable claim graph. [5]

This connects directly to [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md): the visualization code had to understand notation, type labels, nested hyperedges, and role structure to make outputs inspectable.

# Caveats

- The scripts assume historical local file locations.
- They are not documented as a reusable visualization package.
- Several scripts print or save outputs directly rather than exposing importable functions or stable CLI contracts.
- The visualizations are evidence of inspection practice, not proof that the extracted theory applications were fully faithful.

# Links

- [Lit Review Src Code Inventory](/wiki/sources/lit-review-src-code-inventory.md)
- [Lit Review Schema Application Code](/wiki/sources/lit-review-schema-application-code.md)
- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/create_ascii_network.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/create_networkx_visualization.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/visualize_iran_hypergraph_formal.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/visualization/visualize_sh_readable.py`
