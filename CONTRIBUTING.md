# Contributing to GC-Bench

Thank you for your interest in improving GC-Bench!  
This document describes how to add new models, fix bugs, or extend the benchmark.

---

## Project structure recap

```
gc_bench/
  config.py        ← hyperparameters (CFG dataclass)
  data.py          ← HDF5 loading and preprocessing
  metrics.py       ← all evaluation metric functions
  training.py      ← train_torch, predict, DeepEnsemble
  visualization.py ← matplotlib helpers, light/dark themes
  models/
    deep.py        ← deterministic architectures
    generative.py  ← probabilistic / generative models
scripts/
  bench_*.py       ← one runnable benchmark per task
```

---

## Adding a new model

### 1. Implement the architecture

Add your model class to `gc_bench/models/deep.py` (deterministic) or
`gc_bench/models/generative.py` (probabilistic).

Follow the interface:

```python
class MyModel(nn.Module):
    def __init__(self, in_d: int, out_d: int, **kwargs):
        super().__init__()
        # ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_d)
        # returns: (B, out_d)
        return ...
```

Export it in `gc_bench/models/__init__.py`.

### 2. Register in the benchmark script

Open the relevant `scripts/bench_*.py` and add your model to the loop:

```python
for mname, model in [
    ...
    ("MyModel", MyModel(in_d, out_d, ...)),
]:
```

### 3. Verify metrics are computed correctly

Run the benchmark:

```bash
python scripts/bench_forward_scalar.py --data /path/to/data.h5
```

Check that your model appears in the summary table.

---

## Coding conventions

- **Type hints** on all public functions
- **Docstrings** on all classes and non-trivial functions (Google style)
- **No training-time prints** inside model `forward()` methods
- Use `free_memory(*models)` after evaluation to release GPU memory
- Keep each script independently runnable (no required cell ordering)

---

## Reporting results

If you find a model that significantly improves on the benchmark, please open a PR
with:

1. The model implementation
2. Updated `results/README.md` table
3. A short description of the approach and why it helps

---

## Questions?

Open a GitHub issue or start a discussion.
