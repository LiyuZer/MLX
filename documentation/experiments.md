# Writing Experiments

Experiments are Python files that define one or more trial functions and optional verification steps. You can also define a setup function to establish a shared run-level context (e.g., seeds, paths) used by all trials and verifications.

Contents
- Decorators overview
- @setup: run context and seeding
- @trial: parameter grids and execution
- @verify: consuming trial results
- Reserved names and constraints
- Context updates from trials
- Examples

## Decorators overview
- @setup(order: int | None = None)
  - Runs once before any trials to produce a JSON-serializable run context.
  - Multiple setup functions run in order (by the optional order value; otherwise by file order), and their returned dicts are merged (later keys override earlier ones).
  - The merged context is available as a global SETUP dict inside all generated trial/verification scripts.
  - The runner automatically seeds random generators based on fields present in the context (see Seeding below).

- @trial(order: int, values: dict[str, list])
  - Declares a function to run with a cartesian product of parameters.
  - order controls relative execution order among trials.
  - values holds parameter lists; the runner executes one combination per child subprocess.

- @verify(trial_names=[...])
  - Runs after trials finish, receives all trial results, and returns a verdict.

## @setup: run context and seeding
- Purpose: produce a JSON-serializable dictionary shared across the run.
- Where it appears: at module level, once or multiple times.
- Availability in trials/verifications: as a module-level global named SETUP.
- Automatic seeding: if the setup context includes any of the following keys, the runner attempts to seed before executing your trial code in each child process.
  - seed: int (used for Python's random and as default for numpy/torch if their specific keys are not set)
  - numpy_seed: int (numpy.random.seed)
  - torch_seed: int (torch.manual_seed; also CUDA via torch.cuda.manual_seed_all if available; cudnn set deterministic=True, benchmark=False)
  - pythonhashseed: int (sets PYTHONHASHSEED)

Notes
- Seeding is best-effort and only applies to libraries available in the environment.
- SETUP must be JSON-serializable; store paths/configs, not live objects (models, datasets). Reconstruct objects inside trials if needed.

Example
```python
from decorators import setup, trial, verify, VerificationResult
import numpy as np

@setup()
def init():
    # Consistent results across the whole run
    return {"seed": 42, "numpy_seed": 42, "torch_seed": 42}

@trial(1, {"n": [3]})
def t(n: int):
    # SETUP is available as a global
    x = np.random.RandomState(SETUP.get("numpy_seed", 0)).randn(n)
    return {"x_mean": float(x.mean())}

@verify(["t"]) 
def check(results):
    runs = results.get("t", [])
    if not runs:
        return VerificationResult.INCONCLUSIVE
    ok = all(r.get("success") and ("result" in r) for r in runs)
    return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
```

## @trial: parameter grids and execution
- values defines lists for each argument. The cartesian product over lists is executed.
- Only literals and containers of literals are allowed in values (parsed with ast.literal_eval). Avoid objects like Adam(lr=...). If parameters are coupled, use a single cfg parameter with a list of dicts.
- Reserved value keys:
  - "__parallel__": optional int to request parallel workers per trial (see Running doc). If missing, defaults to 1 (serial).
  - "setup": reserved; cannot be used as a parameter name.

Parallel execution
- If __parallel__ > 1 (or MLX_PARALLEL is set), combinations are executed in parallel worker threads, each spawning a subprocess.
- Per-combination streaming is disabled in parallel mode to keep logs readable; progress updates when combinations complete.

Streaming
- Single-combination trials stream stdout/stderr by default.
- For multi-combo trials, set MLX_STREAM_ALL=1 or use CLI flag --stream-all to stream each combination (sequential mode).

## @verify: consuming trial results
- Signature: verify(results: dict[str, list[dict]])
- results maps trial function name -> list of run dicts. Each run dict:
  - Success: {"success": true, "result": <JSON-serializable>}
  - Failure: {"success": false, "return_code": int, "error": str, "type": str}
- Return values supported:
  - VerificationResult enum (preferred), or
  - one of: "supports", "refutes", "inconclusive", "invalid", "continue" (case-insensitive), or
  - bool (True->supports, False->refutes), or
  - None (inconclusive)
- The global SETUP dict is available inside verification scripts as well.

## Reserved names and constraints
- values must be JSON-literal friendly (ints, floats, bool, str, None, lists, dicts of literals).
- Reserved keys: "__parallel__" (parallel workers), "setup" (disallowed in values).

## Context updates from trials
- A trial may return a dict with an optional field: {"context_update": {...}}.
- The runner merges the context_update into the run context if it is JSON-serializable.
- Sequential mode: merge happens immediately; subsequent combinations/trials see the update.
- Parallel mode: updates are applied after the batch completes (siblings do not see each other’s updates while running). Last-writer-wins.

Example: updating context
```python
from decorators import setup, trial

@setup()
def init():
    return {"path": "/tmp/data"}

@trial(1, {"n": [1, 2, 3]})
def gen(n: int):
    # Use SETUP for paths/configs; return updated info if needed
    p = SETUP["path"]
    # ... write files to p ...
    # Example update: record last_n for downstream trials
    return {"context_update": {"last_n": n}}
```

## Example: seeds for numpy and torch, plus a simple verification
```python
from decorators import setup, trial, verify, VerificationResult
import sys

@setup()
def init():
    # Seeding across libraries
    return {"seed": 123, "numpy_seed": 123, "torch_seed": 123}

@trial(1, {"n": [5]})
def train(n: int):
    import numpy as np
    try:
        import torch
        # torch seeded automatically by the runner in each child process
        t = torch.randn(n).tolist()
    except Exception:
        t = []
    # Using numpy seeded via SETUP
    rng = np.random.default_rng(SETUP.get("numpy_seed", 0))
    x = rng.standard_normal(n)
    return {"torch_sample": t, "np_mean": float(x.mean())}

@verify(["train"]) 
def check(results):
    runs = results.get("train", [])
    if not runs:
        return VerificationResult.INCONCLUSIVE
    # Minimal check: results exist and have expected keys
    ok = all(r.get("success") and ("np_mean" in r.get("result", {})) for r in runs)
    return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
```

Guidelines
- Keep return values JSON-serializable.
- Use SETUP for shared configuration and seeding.
- For coupled parameters, prefer a single cfg list pattern.
- For large objects, write them to disk and pass paths in SETUP.