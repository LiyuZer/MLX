from decorators import setup, trial, verify, VerificationResult


@setup()
def init():
    """Run-level setup: provide consistent seeds across libraries.
    The runner will automatically apply these seeds in each child process
    before executing trial code (random/numpy/torch best-effort).
    """
    return {"seed": 123, "numpy_seed": 123, "torch_seed": 123}


@trial(1, {"n": [5]})
def train(n: int):
    """Simple trial that uses NumPy and (optionally) Torch under a fixed seed.

    Notes:
    - SETUP is injected by the runner into the generated temp script.
    - Torch usage is optional; if not available, we proceed without it.
    """
    import numpy as np

    # NumPy sample using the seed from SETUP (provided by @setup)
    rng = np.random.default_rng(SETUP.get("numpy_seed", 0))
    x = rng.standard_normal(n)

    # Torch sample (if available) seeded automatically by the runner
    try:
        import torch
        t = torch.randn(n).tolist()
    except Exception:
        t = []

    # Return JSON-serializable result (verification consumes this)
    return {"np_mean": float(x.mean()), "torch_sample": t}


@verify(["train"])  # or @verify(trial_names=["train"]) is equivalent

def check(results):
    """Minimal verification: ensure the trial produced the expected keys.

    results: dict[trial_name -> list[run_dict]]
    run_dict (success): {"success": True, "result": {...}}
    run_dict (failure): {"success": False, "error": str, "type": str, ...}
    """
    runs = results.get("train", [])
    if not runs:
        return VerificationResult.INCONCLUSIVE

    ok = all(r.get("success") and ("np_mean" in (r.get("result") or {})) for r in runs)
    return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
