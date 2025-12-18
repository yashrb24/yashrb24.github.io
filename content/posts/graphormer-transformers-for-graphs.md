+++
title = "Graphormer: Transformers for Graph-Structured Data"
description = "Understanding how Graphormer adapts the transformer architecture for graph representation learning"
date = 2025-12-16

[taxonomies]
tags = ["graph-neural-networks", "transformers", "machine-learning"]

[extra]
toc = false
comment = false
external_url = "https://gram-blogposts.github.io/blog/2024/graphormer/"
+++

Transformers have revolutionized NLP and computer vision, but applying them to graph-structured data presents unique challenges. Unlike sequences or grids, graphs have irregular topology without a natural ordering of nodes.

Graphormer addresses this by introducing novel structural encodings that capture graph properties within the transformer framework:

- **Centrality Encoding**: Captures node importance using degree information
- **Spatial Encoding**: Encodes pairwise distances between nodes in the attention mechanism
- **Edge Encoding**: Incorporates edge features along shortest paths between node pairs

These encodings allow the standard transformer architecture to reason about graph structure while leveraging the powerful attention mechanism for learning node representations.

*Published as part of the GRAM workshop's blogpost track @ ICML.*
