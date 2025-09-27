import uuid
import time
import json
import os
import sys
from colorama import Fore, Style
import itertools as it
from tqdm import tqdm
import shutil
import atexit
import signal

# Public enums/utilities used by callers/tests
from decorators import VerificationResult

# Refactored models and (de)serialization
from models import (
    TrialRun,
    Experiment,
    Hypothesis,
)

# Execution/parsing helpers
from runner import (
    make_function as _make_function,
    extract_file_components as _extract_file_components,
    create_trial_temp_file as _create_trial_temp_file,
    create_verify_temp_file as _create_verify_temp_file,
    execute_temp_file as _execute_temp_file,
)

# Persistence / cleanup helpers
from storage import (
    store_hypothesis_tree as _store_hypothesis_tree,
    load_hypothesis_tree as _load_hypothesis_tree,
    cleanup_orphan_temp_dirs as _cleanup_orphan_temp_dirs,
    setup_environment as _setup_environment,
)


class MLXCore:
    def __init__(self, mlx_dir: str = ".mlx"):
        self.mlx_dir = mlx_dir
        # Go to the mlx.tree file and load the head hypothesis tree
        self.head_hypothesis: Hypothesis | None = None
        self.root_hypothesis: Hypothesis | None = None
        self.environment_path: str | None = None
        self.load_hypothesis_tree()
        # Ensure a dedicated temp root for runs exists
        self.temp_root_dir = os.path.join(self.mlx_dir, "tmp")
        os.makedirs(self.temp_root_dir, exist_ok=True)

    # ---------- Environment / Persistence ----------
    def setup_environment(self, env_path: str | None = None):
        # Delegate to storage for persistence logic
        new_env = _setup_environment(self.mlx_dir, env_path)
        self.environment_path = new_env
        # Keep persisted hypothesis tree in sync with current tree/head
        self.store_hypothesis_tree()

    def store_hypothesis_tree(self):
        _store_hypothesis_tree(
            self.mlx_dir, self.root_hypothesis, self.head_hypothesis, self.environment_path
        )

    def load_hypothesis_tree(self):
        root, head, env = _load_hypothesis_tree(self.mlx_dir)
        self.root_hypothesis = root
        self.head_hypothesis = head
        self.environment_path = env

    def cleanup_orphan_temp_dirs(self, max_age_hours: int = 24):
        """Remove leftover run directories older than max_age_hours under .mlx/tmp.
        Best-effort cleanup to prevent disk clutter from unexpected terminations.
        """
        _cleanup_orphan_temp_dirs(self.temp_root_dir, max_age_hours)

    # ---------- Tree Navigation / Helpers ----------
    def find_hypothesis_by_id(self, hypothesis: Hypothesis, hypo_id: uuid.UUID) -> Hypothesis | None:
        if hypothesis.id == hypo_id:
            return hypothesis
        for child in hypothesis.children:
            result = self.find_hypothesis_by_id(child, hypo_id)
            if result:
                return result
        return None

    def find_hypothesis_by_prefix(self, id_prefix: str) -> Hypothesis | None:
        """Find a hypothesis anywhere in the tree by UUID prefix."""
        if self.root_hypothesis is None:
            return None

        target: Hypothesis | None = None

        def walk(h: Hypothesis):
            nonlocal target
            if str(h.id).startswith(id_prefix):
                target = h
                return
            for c in h.children:
                if target:
                    return
                walk(c)

        walk(self.root_hypothesis)
        return target

    def set_head(self, id_prefix: str) -> bool:
        """Set the head hypothesis to the one matching the given id prefix."""
        h = self.find_hypothesis_by_prefix(id_prefix)
        if not h:
            print(f"{Fore.RED}Hypothesis with ID starting '{id_prefix}' not found.{Style.RESET_ALL}")
            return False
        self.head_hypothesis = h
        self.store_hypothesis_tree()
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Switched head to: {h.name} <{str(h.id)[:8]}>")
        return True

    def conclude_hypothesis(self, id_prefix: str, status: str | None = None, notes: str | None = None) -> bool:
        """Set conclusion on a hypothesis: status in {supports,refutes,inconclusive} and optional notes."""
        h = self.find_hypothesis_by_prefix(id_prefix)
        if not h:
            print(f"{Fore.RED}Hypothesis with ID starting '{id_prefix}' not found.{Style.RESET_ALL}")
            return False
        norm: str | None = None
        if isinstance(status, str):
            s = status.lower().strip()
            if s in ["supports", "refutes", "inconclusive"]:
                norm = s
            else:
                norm = s  # store as-is if custom
        h.conclusion_status = norm
        h.conclusion_notes = notes or h.conclusion_notes
        h.concluded_at = time.time()
        self.store_hypothesis_tree()
        lbl = norm.upper() if norm else "(none)"
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Concluded hypothesis {h.name} <{str(h.id)[:8]}>: {lbl}")
        return True

    def find_experiment_by_id(self, experiment_id: str) -> Experiment | None:
        # id is a string, might be partial
        def search_experiment(hypo: Hypothesis) -> Experiment | None:
            for exp in hypo.experiments:
                if str(exp.id).startswith(experiment_id):
                    return exp
            for child in hypo.children:
                result = search_experiment(child)
                if result:
                    return result
            return None

        if self.root_hypothesis is None:
            return None
        return search_experiment(self.root_hypothesis)

    def create_new_hypothesis(self, name: str) -> Hypothesis:
        new_hypo = Hypothesis(name)
        if self.root_hypothesis is None:
            self.root_hypothesis = new_hypo
        else:
            self.head_hypothesis.children.append(new_hypo)  # type: ignore[union-attr]
        self.head_hypothesis = new_hypo
        self.store_hypothesis_tree()
        return new_hypo

    def add_experiment(self, name: str, file_path: str | None = None) -> Experiment:
        if self.head_hypothesis is None:
            raise Exception("No active hypothesis. Please create a hypothesis first.")
        new_exp = Experiment(name, self.head_hypothesis, file_path)
        self.head_hypothesis.experiments.append(new_exp)
        self.store_hypothesis_tree()
        return new_exp

    # ---------- Logging / Display ----------
    def print_log(self):
        """
        Display the hypothesis tree with experiments, delegating rendering to the view module
        to keep concerns separated while preserving previous output formatting.
        """
        try:
            from view import print_hypothesis_tree
        except Exception:
            # Fallback: avoid crashing if view module is unavailable
            print("View module not available.")
            return
        print_hypothesis_tree(self.root_hypothesis, self.head_hypothesis)
    # ---------- Delegation wrappers (back-compat for tests) ----------
    def make_function(self, func_name, func_body, arg_names, arg_values, import_lines):
        return _make_function(func_name, func_body, arg_names, arg_values, import_lines)

    def extract_file_components(self, file_content):
        return _extract_file_components(file_content)

    def create_trial_temp_file(
        self, func_name, func_body, arg_names, arg_values, import_lines, other_code, base_dir, temp_dir=None
    ):
        return _create_trial_temp_file(
            func_name, func_body, arg_names, arg_values, import_lines, other_code, base_dir, temp_dir
        )

    def create_verify_temp_file(
        self, verify_name, verify_node, trial_results, import_lines, other_code, base_dir, temp_dir=None
    ):
        return _create_verify_temp_file(
            verify_name, verify_node, trial_results, import_lines, other_code, base_dir, temp_dir
        )

    def execute_temp_file(self, temp_file_path, python_exe, base_dir, stream_output=True):
        return _execute_temp_file(temp_file_path, python_exe, base_dir, stream_output)

    # ---------- Run Orchestration ----------
    def run_experiment(self, experiment_id: str):
        # STEP 0: VALIDATION
        experiment = self.find_experiment_by_id(experiment_id)
        if not experiment:
            print(f"{Fore.RED}✗ Experiment with ID '{experiment_id}' not found.{Style.RESET_ALL}")
            return

        if not experiment.file_path or not os.path.isfile(experiment.file_path):
            print(f"{Fore.RED}✗ Experiment file '{experiment.file_path}' not found.{Style.RESET_ALL}")
            return

        # Prepare temp environment (cleanup orphans, create run dir, register cleanup handlers)
        self.cleanup_orphan_temp_dirs(max_age_hours=24)

        # Generate run id early and create a dedicated temp directory
        run_id = str(uuid.uuid4())[:8]
        run_temp_dir = os.path.join(self.temp_root_dir, f"run_{run_id}")
        os.makedirs(run_temp_dir, exist_ok=True)

        # Setup signal/exit cleanup
        cleaned = {"done": False}

        def _cleanup():
            if not cleaned["done"]:
                cleaned["done"] = True
                try:
                    if os.path.isdir(run_temp_dir):
                        shutil.rmtree(run_temp_dir, ignore_errors=True)
                except Exception:
                    pass

        atexit.register(_cleanup)
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_signal(signum, frame):
            # Best-effort cleanup then propagate/exit
            _cleanup()
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            else:
                os._exit(1)

        try:
            try:
                signal.signal(signal.SIGINT, _handle_signal)
                signal.signal(signal.SIGTERM, _handle_signal)
            except Exception:
                # On some platforms signals might not be available; ignore
                pass

            # Print header
            print(f"\n{Fore.CYAN}{'═'*70}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Style.BRIGHT}Executing Experiment: {experiment.name}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} Hypothesis: {Fore.YELLOW}{experiment.parent_hypothesis.name}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} File: {Style.DIM}{experiment.file_path}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'═'*70}{Style.RESET_ALL}\n")

            experiment.status = "running"
            self.store_hypothesis_tree()

            # STEP 1: PARSE AND EXTRACT
            print(f"{Fore.BLUE}◆ Parsing experiment structure...{Style.RESET_ALL}")
            with open(experiment.file_path, "r") as f:
                file_content = f.read()

            # Setup Python executable and base directory
            python_exe = "python3"
            if self.environment_path:
                python_exe = os.path.join(self.environment_path, "bin", "python")
                print(f"  {Style.DIM}Using environment: {self.environment_path}{Style.RESET_ALL}")

            base_dir = os.path.dirname(os.path.abspath(experiment.file_path))
            temp_files: list[str] = []  # Track temp files for cleanup (extra safety)

            # Extract components from file
            trial_funcs, verify_funcs, import_lines, other_code = self.extract_file_components(file_content)

            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Found {len(trial_funcs)} trial(s) and {len(verify_funcs)} verification(s)")
            print(f"  {Style.DIM}Helper code segments: {len(other_code)}{Style.RESET_ALL}\n")

            # STEP 2: INITIALIZE RUN
            trial_names = [name for _, name, _, _ in trial_funcs]
            trial_run = TrialRun(trial_names)
            trial_run.status = "running"

            print(f"{Fore.BLUE}◆ Starting trial run {Style.BRIGHT}{run_id}{Style.RESET_ALL}\n")

            # STEP 3: EXECUTE TRIALS
            print(f"{Fore.MAGENTA}{'─'*70}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}TRIALS{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'─'*70}{Style.RESET_ALL}\n")

            for order, func_name, values, func_body in trial_funcs:
                arg_names = list(values.keys())
                arg_values = list(values.values())
                cartesian_product = list(it.product(*arg_values))
                trial_run.results[func_name] = []

                print(f"{Fore.GREEN}▶ Trial {order}: {Style.BRIGHT}{func_name}{Style.RESET_ALL}")
                print(f"  {Style.DIM}Parameter combinations: {len(cartesian_product)}{Style.RESET_ALL}")

                # Setup progress bar for multiple combinations
                if len(cartesian_product) > 1:
                    progress_bar = tqdm(
                        total=len(cartesian_product),
                        bar_format='{l_bar}{bar:30}{r_bar}',
                        colour='green',
                        leave=False,
                    )

                start_time = time.time()
                successful_runs = 0
                failed_runs = 0

                for i, combination in enumerate(cartesian_product):
                    # Create temporary file for this trial run
                    temp_file_path = self.create_trial_temp_file(
                        func_name,
                        func_body,
                        arg_names,
                        combination,
                        import_lines,
                        other_code,
                        base_dir,
                        temp_dir=run_temp_dir,
                    )
                    temp_files.append(temp_file_path)

                    # Show parameters
                    param_str = ", ".join([f"{k}={v}" for k, v in zip(arg_names, combination)])
                    if len(cartesian_product) > 1:
                        progress_bar.set_description(f"  Running with {param_str}")
                    else:
                        print(f"  {Style.DIM}Parameters: {param_str}{Style.RESET_ALL}")

                    # Execute the temp file
                    stream_output = len(cartesian_product) == 1
                    result = self.execute_temp_file(temp_file_path, python_exe, base_dir, stream_output)

                    if not result.get("success", False):
                        # Handle failure
                        if stream_output:
                            error_msg = result.get("error", "Unknown error")
                            error_type = result.get("type", "Error")
                            print(f"    {Fore.RED}✗ {error_type}: {error_msg}{Style.RESET_ALL}")

                        trial_run.results[func_name].append(result)
                        failed_runs += 1
                    else:
                        # Handle success
                        if stream_output:
                            result_str = str(result.get("result", "No result"))
                            if len(result_str) > 100:
                                result_str = result_str[:97] + "..."
                            print(f"    {Fore.GREEN}✓ Result: {result_str}{Style.RESET_ALL}")

                        trial_run.results[func_name].append(result)
                        successful_runs += 1

                    if len(cartesian_product) > 1:
                        progress_bar.update(1)

                if len(cartesian_product) > 1:
                    progress_bar.close()

                elapsed = time.time() - start_time
                print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Completed in {elapsed:.3f}s")
                print(
                    f"  {Style.DIM}Successful: {successful_runs}/{len(cartesian_product)}, "
                    f"Failed: {failed_runs}/{len(cartesian_product)}{Style.RESET_ALL}\n"
                )

            # STEP 4: EXECUTE VERIFICATIONS
            if verify_funcs:
                print(f"{Fore.YELLOW}{'─'*70}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}VERIFICATIONS{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{'─'*70}{Style.RESET_ALL}\n")

                for verify_name, trial_names, verify_node in verify_funcs:
                    # Check if all required trials are present
                    missing_trials = [name for name in trial_names if name not in trial_run.results]

                    if missing_trials:
                        print(f"{Fore.RED}▶ Verification: {verify_name}")
                        print(f"  ✗ Skipped - missing trials: {', '.join(missing_trials)}{Style.RESET_ALL}\n")
                        trial_run.verification_results[verify_name] = VerificationResult.INVALID
                        continue

                    print(f"{Fore.YELLOW}▶ Verification: {Style.BRIGHT}{verify_name}{Style.RESET_ALL}")
                    print(f"  {Style.DIM}Verifying trials: {', '.join(trial_names)}{Style.RESET_ALL}")

                    # Create temporary file for verification
                    temp_file_path = self.create_verify_temp_file(
                        verify_name,
                        verify_node,
                        trial_run.results,
                        import_lines,
                        other_code,
                        base_dir,
                        temp_dir=run_temp_dir,
                    )
                    temp_files.append(temp_file_path)

                    start_time = time.time()
                    result = self.execute_temp_file(temp_file_path, python_exe, base_dir, True)
                    elapsed = time.time() - start_time

                    if not result.get("success", False):
                        error_msg = result.get("error", "Unknown error")
                        error_type = result.get("type", "Error")
                        print(f"    {Fore.RED}✗ {error_type}: {error_msg}{Style.RESET_ALL}")
                        trial_run.verification_results[verify_name] = VerificationResult.INVALID
                        status_symbol = f"{Fore.RED}⚠ INVALID{Style.RESET_ALL}"
                    else:
                        verif_result = result.get("result")

                        # Parse verification result
                        if isinstance(verif_result, str):
                            valid_results = [
                                "supports",
                                "refutes",
                                "inconclusive",
                                "invalid",
                                "continue",
                            ]
                            if verif_result.lower() in valid_results:
                                verif_enum = VerificationResult(verif_result.lower())
                            else:
                                verif_enum = VerificationResult.INVALID
                        elif isinstance(verif_result, bool):
                            verif_enum = (
                                VerificationResult.SUPPORTS if verif_result else VerificationResult.REFUTES
                            )
                        elif verif_result is None:
                            verif_enum = VerificationResult.INCONCLUSIVE
                        else:
                            verif_enum = VerificationResult.INVALID

                        trial_run.verification_results[verify_name] = verif_enum

                        status_symbol = {
                            VerificationResult.SUPPORTS: f"{Fore.GREEN}✓ SUPPORTS{Style.RESET_ALL}",
                            VerificationResult.REFUTES: f"{Fore.RED}✗ REFUTES{Style.RESET_ALL}",
                            VerificationResult.INCONCLUSIVE: f"{Fore.YELLOW}? INCONCLUSIVE{Style.RESET_ALL}",
                            VerificationResult.INVALID: f"{Fore.RED}⚠ INVALID{Style.RESET_ALL}",
                            VerificationResult.CONTINUE: f"{Fore.BLUE}→ CONTINUE{Style.RESET_ALL}",
                        }[verif_enum]

                    print(f"    {Style.DIM}└─{Style.RESET_ALL} Result: {status_symbol} ({elapsed:.3f}s)")
                    print()

            # STEP 5: DETERMINE STATUS
            verification_values = list(trial_run.verification_results.values())

            if not verification_values:
                trial_run.status = "inconclusive"
                status_color = Fore.YELLOW
                status_symbol = "?"
            elif all(v == VerificationResult.SUPPORTS for v in verification_values):
                trial_run.status = "validated"
                status_color = Fore.GREEN
                status_symbol = "✓"
            elif any(v == VerificationResult.REFUTES for v in verification_values):
                trial_run.status = "invalidated"
                status_color = Fore.RED
                status_symbol = "✗"
            elif any(v == VerificationResult.INVALID for v in verification_values):
                trial_run.status = "invalid"
                status_color = Fore.RED
                status_symbol = "⚠"
            elif any(v == VerificationResult.INCONCLUSIVE for v in verification_values):
                trial_run.status = "inconclusive"
                status_color = Fore.YELLOW
                status_symbol = "?"
            else:
                trial_run.status = "inconclusive"
                status_color = Fore.YELLOW
                status_symbol = "?"

            # STEP 6: STORE AND SUMMARIZE
            experiment.runs[run_id] = trial_run
            experiment.status = trial_run.status
            self.store_hypothesis_tree()

            print(f"\n{Fore.CYAN}{'═'*70}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Style.BRIGHT}RUN COMPLETE{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} Run ID: {Style.BRIGHT}{run_id}{Style.RESET_ALL}")
            print(
                f"{Fore.CYAN}║{Style.RESET_ALL} Status: {status_color}{status_symbol} {trial_run.status.upper()}{Style.RESET_ALL}"
            )
            print(f"{Fore.CYAN}║{Style.RESET_ALL} Trials executed: {len(trial_run.results)}")
            print(
                f"{Fore.CYAN}║{Style.RESET_ALL} Verifications completed: {len(trial_run.verification_results)}"
            )
            print(f"{Fore.CYAN}{'═'*70}{Style.RESET_ALL}\n")

            return run_id, trial_run

        finally:
            # CLEANUP
            try:
                _cleanup()
            except Exception:
                pass
            # Restore previous signal handlers
            try:
                signal.signal(signal.SIGINT, prev_sigint)
                signal.signal(signal.SIGTERM, prev_sigterm)
            except Exception:
                pass

    # ---------- Status Query ----------
    def get_experiment_status(self, experiment_id: str):
        experiment = self.find_experiment_by_id(experiment_id)
        if not experiment:
            print(f"Experiment with ID starting '{experiment_id}' not found.")
            return

        print(f"Experiment '{experiment.name}' <{str(experiment.id)[:8]}> status: {experiment.status}")

        # No runs yet
        if not getattr(experiment, "runs", None):
            print("No runs recorded yet.")
            return experiment.status, {}, {}

        # Latest run is the most recently inserted (dicts preserve insertion order)
        latest_run_id = list(experiment.runs.keys())[-1]
        latest_run = experiment.runs[latest_run_id]

        # Summaries
        trial_func_count = len(latest_run.results)
        trial_exec_count = sum(len(v) for v in latest_run.results.values())
        verification_count = len(latest_run.verification_results)

        print(f"Latest run: {latest_run_id} -> {latest_run.status}")
        print(f"Trials: {trial_func_count} function(s), {trial_exec_count} execution(s)")
        print(f"Verifications: {verification_count}")

        if latest_run.verification_results:
            print("Verification results:")
            for name, v in latest_run.verification_results.items():
                label = v.value if hasattr(v, "value") else str(v)
                print(f"  - {name}: {label}")

        return experiment.status, latest_run.results, latest_run.verification_results