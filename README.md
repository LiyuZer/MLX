# MLX — Hypothesis‑Driven Experiment Runner 🧪🐉

<p align="center">
  <b>Think like a scientist:</b> write a hypothesis, attach experiments, run structured trials, verify, and conclude.
  <br/>
  Clean temp runs • Rich CLI • Lightweight decorators • Pretty logs
</p>

---

## ✨ Features
- Hypothesis tree with head switching, details, and conclusions
- Experiments defined in regular Python files using lightweight decorators
- Cartesian product trial execution with pretty progress bars
- Verification hooks returning enums/strings/bools
- Isolated temp runs with best‑effort cleanup and orphan removal
- Simple, readable CLI; colors where supported

## 📦 Requirements
- Python 3.9+ (uses ast.unparse)
- pip install colorama tqdm

```
pip install -U colorama tqdm
```

## 🚀 Quickstart
1) Initialize
```
mlx init
```

2) Create a hypothesis (title = first line; body may be multiline)
```
mlx hypothesis create -m "Improve Fibonacci runtime\nWe expect the iterative method to be faster than recursive for n>20."
# or from a file
mlx hypothesis create -F hypothesis.txt
```
If you have a virtual env:
```
mlx setup --env path/to/venv
```

3) Add an experiment file and register it

Example experiment (examples/fib_experiment.py):
```python
from decorators import trial, verify, VerificationResult

@trial(order=1, values={"n": [10, 20, 25], "method": ["rec", "iter"]})
def fib_trial(n, method):
    def fib_rec(k):
        return k if k < 2 else fib_rec(k-1) + fib_rec(k-2)

    def fib_iter(k):
        a, b = 0, 1
        for _ in range(k):
            a, b = b, a + b
        return a

    if method == "rec":
        return fib_rec(n)
    return fib_iter(n)

@verify(["fib_trial"])  # or @verify(trial_names=["fib_trial"])
def verify_fib(results):
    # Structural check: all runs should have a result
    ok = all(r.get("success") and ("result" in r) for r in results["fib_trial"])
    return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
```
Register the experiment:
```
mlx exp add examples/fib_experiment.py "Fibonacci Experiment"
```

4) Run the experiment
```
# Use the experiment ID prefix (shown after add)
mlx run 1234abcd
```

5) Check status and view the tree
```
mlx status 1234abcd
mlx log
```

6) Conclude your hypothesis
```
# Pick the hypothesis by ID prefix from the log
mlx hypothesis conclude abcd1234 --status supports -m "Iterative method is reliable and faster for n>20."
```

## 🧠 Core Concepts
- Hypotheses
  - Create rich, descriptive hypotheses.
  - Switch the active one (head), add experiments under it, and conclude when done.
  - The log prints a tree with statuses and conclusion labels.

- Experiments and Trials
  - Experiments are Python files with functions decorated by @trial and @verify.
  - @trial defines an order and a values dict mapping parameter names -> lists (MLX runs the cartesian product).
  - @verify accepts trial names and should return a VerificationResult (preferred), a known string, a bool, or None.

- Decorators (lightweight markers)
  - from decorators import trial, verify, VerificationResult
  - trial(order, values) or trial(order=..., values=...)
  - verify(["trial1", "trial2"]) or verify(trial_names=[...])
  - They are metadata markers only; MLX parses your file’s AST and reconstructs your functions unchanged.

- Verification Results
  - Return one of:
    - VerificationResult.SUPPORTS / REFUTES / INCONCLUSIVE / INVALID / CONTINUE
    - or strings: "supports", "refutes", "inconclusive", "invalid", "continue"
    - or boolean: True -> SUPPORTS, False -> REFUTES
    - or None -> INCONCLUSIVE

## 🧰 CLI Reference (high level)
- mlx init
- mlx hypothesis create [-m text | -F file]
- mlx hypothesis list
- mlx hypothesis show <hypo_id_prefix>
- mlx hypothesis switch <hypo_id_prefix>
- mlx hypothesis set-desc <hypo_id_prefix> [-m text | -F file]
- mlx hypothesis conclude <hypo_id_prefix> --status {supports,refutes,inconclusive} [-m notes | -F file]
- mlx hypothesis delete <hypo_id_prefix> [--force]
- mlx exp add <file_path> <name>
- mlx exp list | show <exp_id_prefix>
- mlx exp delete <exp_id_prefix>
- mlx run <exp_id_prefix>
- mlx status <exp_id_prefix>
- mlx log
- mlx setup --env venv_path
## 🧹 Temp Files and Cleanup
- Each run uses a dedicated directory: .mlx/tmp/run_<id>.
- On normal exit or Ctrl+C, MLX cleans the run directory.
- If the process is killed, orphan dirs are removed automatically on the next run.

## 💡 Tips
- Keep experiments small and focused; put helpers and imports at top level.
- Prefer Python 3.9+ so ast.unparse is available (or upgrade your venv interpreter).

---

Happy experimenting! 🐉🔥