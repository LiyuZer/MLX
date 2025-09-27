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

    # ---------- Deletion (Hypotheses / Experiments) ----------
    def _find_parent_of_hypothesis(self, current: Hypothesis, target: Hypothesis) -> Hypothesis | None:
        for child in current.children:
            if child.id == target.id:
                return current
            found = self._find_parent_of_hypothesis(child, target)
            if found:
                return found
        return None

    def delete_hypothesis(self, id_prefix: str, force: bool = False) -> bool:
        """Delete a hypothesis by ID prefix. Prevent deleting non-empty nodes unless force=True.

        Behavior:
        - If the hypothesis has children or experiments, require force=True.
        - If deleting the root hypothesis, clear the tree (respecting force rules).
        - If the deleted hypothesis is the head, move head to its parent (or None if root).
        - Persist changes via store_hypothesis_tree and print a status line.
        """
        if self.root_hypothesis is None:
            print(f"{Fore.RED}No hypothesis tree initialized.{Style.RESET_ALL}")
            return False

        target = self.find_hypothesis_by_prefix(id_prefix)
        if not target:
            print(f"{Fore.RED}Hypothesis with ID starting '{id_prefix}' not found.{Style.RESET_ALL}")
            return False

        has_children_or_exps = bool(target.children) or bool(target.experiments)
        if has_children_or_exps and not force:
            print(
                f"{Fore.RED}Cannot delete hypothesis with children or experiments. "
                f"Use --force to override.{Style.RESET_ALL}"
            )
            return False

        # Find parent (None if target is root)
        parent: Hypothesis | None = None
        if self.root_hypothesis.id != target.id:
            parent = self._find_parent_of_hypothesis(self.root_hypothesis, target)

        # Perform deletion
        if parent is None:
            # Deleting root
            self.root_hypothesis = None
            self.head_hypothesis = None
        else:
            parent.children = [c for c in parent.children if c.id != target.id]
            # Update head if needed
            if self.head_hypothesis and self.head_hypothesis.id == target.id:
                self.head_hypothesis = parent

        self.store_hypothesis_tree()
        print(
            f"{Fore.GREEN}✓{Style.RESET_ALL} Deleted hypothesis {target.name} <{str(target.id)[:8]}>"
        )
        return True

    def delete_experiment(self, id_prefix: str) -> bool:
        """Delete an experiment by ID prefix from wherever it resides in the tree."""
        if self.root_hypothesis is None:
            print(f"{Fore.RED}No hypothesis tree initialized.{Style.RESET_ALL}")
            return False

        def find_parent_and_exp(h: Hypothesis):
            # Search experiments in this node
            for exp in h.experiments:
                if str(exp.id).startswith(id_prefix):
                    return h, exp
            # Recurse into children
            for c in h.children:
                found = find_parent_and_exp(c)
                if found:
                    return found
            return None

        found = find_parent_and_exp(self.root_hypothesis)
        if not found:
            print(f"{Fore.RED}Experiment with ID starting '{id_prefix}' not found.{Style.RESET_ALL}")
            return False

        parent_h, exp = found
        parent_h.experiments = [e for e in parent_h.experiments if e.id != exp.id]
        self.store_hypothesis_tree()
        print(
            f"{Fore.GREEN}✓{Style.RESET_ALL} Deleted experiment {exp.name} <{str(exp.id)[:8]}>"
        )
        return True

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
        """Delegate run orchestration to runner.run_experiment_impl to slim core.

        Returns:
            tuple[str, TrialRun] | None: (run_id, trial_run) on success, otherwise None.
        """
        experiment = self.find_experiment_by_id(experiment_id)
        if not experiment:
            print(f"{Fore.RED}✗ Experiment with ID '{experiment_id}' not found.{Style.RESET_ALL}")
            return

        # Lazy import to avoid circular dependencies
        try:
            from runner import run_experiment_impl
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to import run implementation: {e}{Style.RESET_ALL}")
            return

        return run_experiment_impl(self, experiment)
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