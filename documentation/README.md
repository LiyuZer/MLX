MLX: Experiments for Curious Minds

Hello, explorer! This project helps you run little science adventures with code. You make a guess (a hypothesis), try things (trials), and then check your results (verification). Let’s learn how it works step by step.

What is inside the project?
- .mlx folder: A secret notebook. It remembers your hypotheses, experiments, and runs. Don’t delete it unless you want to start fresh.
- mlx_init.py: The main entry file you run from the command line. It uses the CLI (command line interface) to do things.
- cli.py: The brain that reads your commands and decides what to do.
- mlx_core.py: The heart that stores hypotheses and experiments, and talks to the runner to actually run your code.
- runner.py: The engine that runs your experiment functions safely, shows progress, and collects results.
- models.py: The shapes of your data (Hypothesis, Experiment, TrialRun).
- decorators.py: Special stickers to mark your functions as trial() or verify().
- view.py: Pretty printing for trees and colors.
- storage.py: Helpers for reading/writing files (used by core).
- tests/: Some tests that check the CLI and running behavior.
- mlx_init.spec: A recipe for packaging into a single executable using PyInstaller.

Your adventure map (concepts)
- Hypothesis: Your big idea. Example: “Doubling numbers always works.”
- Experiment: A file with code to try your idea. It has trials and verifications.
- Trial: A function that runs with different parameters (like trying n=10 or n=20).
- Verification: A function that reads the trial results and says SUPPORTS, REFUTES, or INCONCLUSIVE.

How to play (quick start)
1) Start a new lab
   mlx init

2) Create a hypothesis
   mlx hypothesis create -m "My First Idea\nI think doubling numbers always works."

3) Make an experiment file (exp.py)
   Example content:
   from decorators import trial, verify, VerificationResult

   @trial(1, {"x": [1, 2, 3]})
   def t(x):
       return x * 2

   @verify(["t"])  # or @verify(trial_names=["t"]) either is fine
   def check(results):
       runs = results.get("t", [])
       ok = all(r.get("success") and ("result" in r) for r in runs)
       return VerificationResult.SUPPORTS if ok else VerificationResult.REFUTES

4) Add the experiment to your hypothesis
   mlx exp add exp.py "Double Numbers"
   This prints an ID like 1234abcd — keep it!

5) Run the experiment
   mlx run 1234abcd
   Tip: To see live output for all parameter combinations, run:
   MLX_STREAM_ALL=1 mlx run 1234abcd

6) Check status and log
   - mlx status 1234abcd
   - mlx log

Commands you can use (CLI)
- mlx init
  Start a new .mlx folder in the current directory.

- mlx hypothesis [options] [subcommand]
  - create -m "Title\nMore details"  Make a new hypothesis.
  - list                         Show a tree of hypotheses and experiments.
  - show <id_prefix>             Show details.
  - switch <id_prefix>           Make one hypothesis active (HEAD).
  - set-desc <id_prefix> -m "..." Update description.
  - conclude <id_prefix> --status supports|refutes|inconclusive -m "notes"
  - delete <id_prefix> [--force] Delete a hypothesis.
  Tip: Back-compat shortcut: mlx hypothesis -m "..." (without subcommand) creates one.

- mlx exp <subcommand>
  - list                         List all experiments under the tree.
  - show <id_prefix>             Show details for one experiment.
  - add <file_path> <name>       Add a new experiment file.
  - delete <id_prefix>           Remove an experiment.
  Shortcuts:
  - mlx exp <id_prefix>          Shorthand for show.
  - mlx exp <file> <name>        Legacy add.

- mlx run <experiment_id>
  Run an experiment by its ID.
  Environment tips:
  - MLX_STREAM_ALL=1  Stream output for every parameter combo.
  - Uses your current Python by default; if you configured an environment folder, it uses that.

- mlx setup [--env path]
  Prepare a Python environment if your project uses one.

- mlx status <experiment_id>
  Show the latest status.

- mlx log
  Print a colorful tree of your hypotheses and experiments.

How running works (simple version)
- The runner looks at your experiment file and finds functions marked with @trial and @verify.
- For each trial, it tries all combinations of parameters you gave.
- It runs each combination in a small, safe, temporary script and captures the result.
- Then it calls verification functions to judge the results.
- Finally, it saves everything back into .mlx so you can see the history.

Seeing progress and prints
- For one parameter combo: the runner streams your print() lines live.
- For many combos: set MLX_STREAM_ALL=1 to stream each combo live.
- We also stream stderr now, so tqdm or warning messages appear.

Troubleshooting
- I ran mlx run <number> and it used a different file name!
  You passed an ID of an existing experiment. IDs map to saved experiments in .mlx. Use mlx exp show <id> to see the file. To run a new local file, first add it with mlx exp add path name and then run the new ID.

- I see 0% on the bar and no prints.
  For many combos the progress bar bumps only after each combo finishes. Use MLX_STREAM_ALL=1 if you want to see prints for each combo while they run.

- My packaged binary (dist/mlx_init) says ModuleNotFoundError: runner
  Rebuild with the included spec (mlx_init.spec) which now bundles local modules.
  Steps: pip install pyinstaller; rm -rf build dist; pyinstaller mlx_init.spec; then ./dist/mlx_init -h

For grown-ups (module tour)
- cli.py
  - Builds the argparse interface and calls functions in MLXCore.
  - Subcommands: init, hypothesis (create/list/show/switch/set-desc/conclude/delete), exp (list/show/add/delete), run, setup, status, log.

- mlx_core.py
  - Loads and saves the hypothesis tree in .mlx.
  - Adds experiments, finds items by ID prefix, switches head.
  - Delegates running to runner.run_experiment_impl.

- runner.py
  - Extracts @trial and @verify functions using AST parsing.
  - Creates small temp scripts to run each trial safely, captures JSON outputs.
  - Streams stdout and stderr lines (child is unbuffered), supports MLX_STREAM_ALL=1.
  - Summarizes and stores results as TrialRun.

- models.py
  - Data classes for Hypothesis, Experiment, TrialRun and related structures.

- decorators.py
  - trial(order, values) marks a function to be run with a parameter grid.
  - verify(trial_names=[...]) marks a function that judges the results.
  - VerificationResult enum with SUPPORTS, REFUTES, INCONCLUSIVE, INVALID, CONTINUE.

- view.py
  - Pretty prints the hypothesis tree with colors and symbols.

- storage.py
  - Helper I/O routines used by the core.

- tests/
  - tests/test_cli.py and others show usage of the CLI and end-to-end flow.

- mlx_init.spec (packaging)
  - Includes hiddenimports for local modules so the single-file executable works.

Safety tips
- Keep .mlx in your project if you want history.
- Use git to track your experiment code changes.
- If something breaks, try a new clean folder, run mlx init, and add your experiment again.

Have fun experimenting!
