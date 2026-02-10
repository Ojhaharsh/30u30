# Day 21 Exercises: Neural Message Passing for Quantum Chemistry

Five exercises building up the components of an MPNN, from individual message
functions to a complete molecular property prediction pipeline.

## Exercises

| # | File | Topic | Difficulty | Time |
|---|------|-------|-----------|------|
| 1 | `exercise_01_message_functions.py` | Implement three message function variants | Easy (2/5) | 30 min |
| 2 | `exercise_02_graph_construction.py` | Build molecular graphs from raw data | Easy (2/5) | 30 min |
| 3 | `exercise_03_readout_functions.py` | Sum pooling and Set2Set readout | Medium (3/5) | 45 min |
| 4 | `exercise_04_edge_networks.py` | Edge network that maps features to matrices | Medium (3/5) | 45 min |
| 5 | `exercise_05_property_prediction.py` | Full MPNN training on multi-target regression | Hard (4/5) | 60 min |

## How to Use

Each exercise file has:
- A description of the task and relevant paper sections
- TODO blocks where you need to fill in the implementation
- A `test_*()` function that validates your implementation
- Hints in comments

Run an exercise:
```bash
python exercise_01_message_functions.py
```

If your implementation is correct, the test function will print "PASSED".
If not, it will tell you what went wrong.

## Solutions

Complete solutions are in `solutions/`. Try to get stuck before looking.

## Tips

- Read the corresponding paper sections before starting each exercise
- The implementation in `../implementation.py` contains working reference code,
  but try to write your own version first
- Exercise 5 builds on concepts from exercises 1-4, so do them in order
- If you are unsure about tensor shapes, add print statements to check
  shapes at each step
