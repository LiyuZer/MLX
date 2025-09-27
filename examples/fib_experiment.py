"""
Example experiment for MLX demonstrating @trial and @verify.

- A Fibonacci trial runs with cartesian combinations of n and method.
- The trial returns a rich result dict including parameters and timing so
  verifications can reason about the outputs.
- Verifications check that all runs succeeded and that recursive/iterative
  methods agree on values, with iterative generally faster for larger n.
"""

from decorators import trial, verify, VerificationResult
import time


def fib_rec(k: int) -> int:
    return k if k < 2 else fib_rec(k - 1) + fib_rec(k - 2)


def fib_iter(k: int) -> int:
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


@trial(order=1, values={"n": [10, 20, 25], "method": ["rec", "iter"]})
def fib_trial(n: int, method: str):
    start = time.time()
    if method == "rec":
        val = fib_rec(n)
    else:
        val = fib_iter(n)
    elapsed_ms = (time.time() - start) * 1000.0

    # Return a rich dict (the MLX runner will JSON-encode it). Including
    # parameters in the result allows verifications to analyze behavior.
    return {
        "n": n,
        "method": method,
        "value": val,
        "time_ms": elapsed_ms,
    }


@verify(["fib_trial"])  # or @verify(trial_names=["fib_trial"])
def verify_fib(results):
    """
    Verify that:
    - All runs succeeded and have a result payload
    - For each n, recursive and iterative results match in value
    - For n >= 20, iterative should be faster than recursive (heuristic)

    The results structure is:
      results[trial_name] = [
        {"success": True, "result": {...}},
        {"success": False, "error": ..., ...},
        ...
      ]
    """
    runs = results.get("fib_trial", [])

    # Basic structural success check
    if not runs:
        return VerificationResult.INCONCLUSIVE

    if not all(r.get("success") and ("result" in r) for r in runs):
        return VerificationResult.INVALID

    # Organize by n -> {method: payload}
    by_n = {}
    for r in runs:
        payload = r.get("result", {})
        n = payload.get("n")
        method = payload.get("method")
        if n is None or method not in {"rec", "iter"}:
            return VerificationResult.INVALID
        by_n.setdefault(n, {})[method] = payload

    # Value equality and performance heuristic
    for n, methods in by_n.items():
        rec = methods.get("rec")
        it = methods.get("iter")
        # If we didn't get both methods for this n, we can't fully verify
        if not rec or not it:
            return VerificationResult.INCONCLUSIVE

        if rec.get("value") != it.get("value"):
            return VerificationResult.REFUTES

        # Heuristic: iterative should be faster for larger n
        if n >= 20:
            if not (it.get("time_ms", 1e9) < rec.get("time_ms", 0)):
                # Not strictly required; if it isn't faster, call it inconclusive
                return VerificationResult.INCONCLUSIVE

    return VerificationResult.SUPPORTS
