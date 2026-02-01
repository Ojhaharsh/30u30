# Data Directory - Day 7: Coffee Automaton

This directory contains data for complexity theory and cellular automata experiments.

## Contents

### Generated Data
- `automaton_states/` - Saved states of cellular automata simulations
- `complexity_graphs/` - Network graphs for complexity analysis
- `visualizations/` - Generated visualization images

### Usage

The notebook will automatically create subdirectories as needed:

```python
import os

# Data paths
DATA_DIR = "data"
AUTOMATON_DIR = os.path.join(DATA_DIR, "automaton_states")
GRAPHS_DIR = os.path.join(DATA_DIR, "complexity_graphs")
VIZ_DIR = os.path.join(DATA_DIR, "visualizations")

# Create directories if needed
os.makedirs(AUTOMATON_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
```

## Notes

- No external datasets required
- All data is generated during experiments
- Visualizations can be saved for later reference
