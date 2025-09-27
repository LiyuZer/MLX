from decorators import trial, verify, VerificationResult
import numpy as np

@trial(1, {"n": [1000], "lr": [0.01, 0.1], "epochs": [100]})
def train(n: int, lr: float, epochs: int):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 1))
    y = 3.5 * X[:, 0] + 2.0 + rng.normal(scale=0.1, size=n)
    X_b = np.c_[np.ones((n, 1)), X]  # bias term
    theta = np.zeros(2)
    for _ in range(epochs):
        preds = X_b @ theta
        grad = (2.0 / n) * X_b.T @ (preds - y)
        theta -= lr * grad
    mse = float(np.mean((X_b @ theta - y) ** 2))
    return {"theta": theta.tolist(), "mse": mse}

@verify(["train"])  # or verify(trial_names=["train"]) is equivalent

def check(results):
    runs = results.get("train", [])
    if not runs:
        return VerificationResult.INCONCLUSIVE
    # Consider successful if MSE is small for all runs
    ok = True
    for r in runs:
        if not r.get("success"):
            ok = False
            break
        res = r.get("result", {})
        mse = res.get("mse", 1e9)
        if mse >= 0.05:
            ok = False
            break
    return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
