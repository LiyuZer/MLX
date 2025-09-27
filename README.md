MLX — Hypothesis-Driven Experiment Runner

Overview
MLX lets you work like a scientist: create a hypothesis with a rich message, attach experiments, run structured trials, verify the results, and conclude what you learned. The CLI manages a tree of hypotheses and experiments, and the core executes trials in isolated temp files for clean runs.

Requirements
- Python 3.9+ (ast.unparse is used)
- pip install colorama tqdm

Quickstart
1) Initialize
   mlx init

2) Create a hypothesis (title is the first line; body can be multiline)
   mlx hypothesis create -m "Improve Fibonacci runtime\nWe expect the iterative method to be faster than recursive for n>20."
   # or from a file
   mlx hypothesis create -F hypothesis.txt

   If there is a virtual env setup with:
   mlx setup --env venv_path

3) Add an experiment file and register it
   # Example file you can create (see examples/fib_experiment.py):
   # from decorators import trial, verify, VerificationResult
   #
   # @trial(order=1, values={"n": [10, 20, 25], "method": ["rec", "iter"]})
   # def fib_trial(n, method):
   #     def fib_rec(k):
   #         return k if k < 2 else fib_rec(k-1) + fib_rec(k-2)
   #     def fib_iter(k):
   #         a, b = 0, 1
   #         for _ in range(k):
   #             a, b = b, a + b
   #         return a
   #     if method == "rec":
   #         return fib_rec(n)
   #     return fib_iter(n)
   #
   # @verify(["fib_trial"])  # or @verify(trial_names=["fib_trial"])
   # def verify_fib(results):
   #     # Simple structural check: all runs should have a result
   #     ok = all(r.get("success") and ("result" in r) for r in results["fib_trial"])
   #     return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES
   
   mlx exp add examples/fib_experiment.py "Fibonacci Experiment"

4) Run the experiment
   # Use the experiment ID prefix (shown after add). Example:
   mlx run 1234abcd

5) Check status and view the tree
   mlx status 1234abcd
   mlx log

6) Conclude your hypothesis
   # Pick the hypothesis by ID prefix from the log
   mlx hypothesis conclude abcd1234 --status supports -m "Iterative method is reliable and faster for n>20."

Core Concepts
- Hypotheses
  - Create rich, descriptive hypotheses.
  - Switch the active one (head), add experiments under it, and conclude when done.
  - The log prints a tree with statuses and conclusion labels.

- Experiments and Trials
  - Experiments are Python files with functions decorated by @trial and @verify.
  - @trial defines an order and a values dict mapping parameter names -> lists.
    MLX computes the cartesian product and executes each combination.
  - @verify accepts the names of trials to verify. It receives a results dict and
    should return a VerificationResult (preferred), a known string, a bool, or None.

Decorators (lightweight wrappers)
- from decorators import trial, verify, VerificationResult
- trial(order, values) or trial(order=..., values=...)
- verify(["trial1", "trial2"]) or verify(trial_names=[...])
- They are metadata markers only; MLXCore parses your file’s AST and reconstructs
  the functions for execution. Your function logic runs unchanged.

Verification Results
- Return one of:
  - VerificationResult.SUPPORTS / REFUTES / INCONCLUSIVE / INVALID / CONTINUE
  - or strings: "supports", "refutes", "inconclusive", "invalid", "continue"
  - or boolean: True -> SUPPORTS, False -> REFUTES
  - or None -> INCONCLUSIVE

CLI Reference (high level)
- mlx init
- mlx hypothesis create [-m text | -F file]
- mlx hypothesis list
- mlx hypothesis show <hypo_id_prefix>
- mlx hypothesis switch <hypo_id_prefix>
- mlx hypothesis set-desc <hypo_id_prefix> [-m text | -F file]
- mlx hypothesis conclude <hypo_id_prefix> --status {supports,refutes,inconclusive} [-m notes | -F file]
- mlx exp add <file_path> <name>
- mlx exp list | show <exp_id_prefix>
- mlx run <exp_id_prefix>
- mlx status <exp_id_prefix>
- mlx log
- mlx setup --env venv_path

Temp Files and Cleanup
- Each run uses a dedicated directory: .mlx/tmp/run_<id>.
- On normal exit or Ctrl+C (SIGINT/SIGTERM), MLX cleans the run directory.
- If the process is killed (e.g., SIGKILL), leftover dirs are removed automatically
  on the next run (orphan cleanup).

Notes
- Keep experiments small and focused; put helpers and imports at top-level in the file.
- Python 3.9+ is required for ast.unparse. If you must use an older Python, consider
  upgrading the interpreter in your virtual environment.

Happy experimenting! 🐉🔥
