# Architecture

MLX organizes work around hypotheses and experiments.

Core modules and responsibilities:
- models.py: Data structures for Hypothesis, Experiment, TrialRun.
- decorators.py: @trial and @verify decorators and VerificationResult enum.
- mlx_core.py: Loads/saves the hypothesis tree in .mlx; operations on hypotheses/experiments; delegates run orchestration to runner.
- runner.py: Parses experiment files, executes trial/verification code in isolated temp scripts, streams output, aggregates results, and stores them back.
- cli.py: Argparse-based command-line interface; dispatches to MLXCore methods.
- view.py: Tree printing and colored output.
- storage.py: Local filesystem helpers used by core.
- mlx_init.py: Entry-point module that invokes cli.main().

Data and persistence:
- .mlx/: Project-local storage for the hypothesis tree and runs.
- Experiments reference a file path; runs are attached to experiments.

High-level control flow when running an experiment:
1) CLI parses the command and loads MLXCore (.mlx).
2) MLXCore resolves the experiment by ID and delegates to runner.run_experiment_impl.
3) runner.py parses the file (AST) to locate @trial and @verify functions and supporting code.
4) For each parameter combination of each trial:
   - A temp Python file is generated with necessary imports and the single function body.
   - The temp file is executed as a child process with unbuffered IO.
   - Output is streamed (stdout and stderr) and the final JSON result is parsed.
5) Verifications run similarly over the aggregated trial results.
6) Results are written back to the hypothesis tree and statuses are updated.

Import relationships:
- cli -> mlx_core
- mlx_core -> runner (delegation)
- runner -> decorators, models
- view is imported lazily for printing
- Avoid circular imports by keeping runner independent of core internals (duck-typed access).
