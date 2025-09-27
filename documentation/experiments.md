# Writing Experiments

Experiments are Python files that define:
- One or more @trial functions with parameter grids.
- Optional @verify functions that evaluate trial results.

Example structure:

```python
from decorators import trial, verify, VerificationResult

@trial(1, {"x": [1, 2, 3]})
def t(x):
    return x * 2

@verify(["t"])  # or verify(trial_names=["t"]) is equivalent

def check(results):
    runs = results.get("t", [])
    if not runs:
        return VerificationResult.INCONCLUSIVE
    ok = all(r.get("success") and ("result" in r) for r in runs)
    return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
```

Decorator details:
- @trial(order, values): order is run order; values is a dict of parameter lists. The runner executes the cartesian product of all values.
- @verify(trial_names=[...]): Receives a dict of trial_name -> list of run dicts. Return a VerificationResult or equivalent value.

Guidelines:
- Keep side effects inside each trial isolated; the runner executes each combo in a separate temp file.
- Return small JSON-serializable results.
- Use print() for diagnostics; they will stream during live runs.

Machine Learning examples:
- See examples/linear_regression_numpy.py for a NumPy-only linear regression with gradient descent and a verification on MSE.
- You can also use frameworks (e.g., PyTorch, scikit-learn) if available in your environment.
