# MQuAKE Entity Swap

This repository provides tools for generating counterfactual variations by swapping entities with similar alternatives from Wikidata.

## Overview

The repository contains two main modules:

- **mquake_swap**: Generates counterfactual variations based on the prebuilt MQuAKE dataset. Since MQuAKE provides structured fact chains with Wikidata entity IDs, this module creates exact counterfactuals for the same questions by replacing entities with similar alternatives.

- **trace_swap**: Generates counterfactual variations for arbitrary reasoning traces. Since traces may not have explicit entity IDs, this module approximates entity matching by mapping entity mentions to Wikidata entries based on similarity, then swaps them with alternatives.

## Repository Structure

```
.
├── mquake_swap/           # MQuAKE counterfactual generator
│   ├── generate_counterfactuals.py
│   └── README.md
├── trace_swap/            # Trace entity swapper
│   ├── trace_entity_swapper.py
│   └── README.md
├── data/                  # Dataset files and loading scripts
├── prompts/               # Templates for generation
├── cache/                 # Wikidata query cache
├── utils.py               # Shared utilities
└── config.json            # Configuration file
```

## Getting Started

For detailed usage instructions, please refer to the README in each module:

- [mquake_swap/README.md](mquake_swap/README.md) - Counterfactual dataset generation
- [trace_swap/README.md](trace_swap/README.md) - Reasoning trace entity swapping