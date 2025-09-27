from __future__ import annotations

import time
import uuid
from typing import Any, Dict

from decorators import VerificationResult


class TrialRun:
    def __init__(self, trial_names):
        self.trials = trial_names
        self.results = {}  # key: trial_name, value: list of results
        self.verification_results = {}  # key: verification_name, value: enum or str
        self.status = "pending"  # 'pending' | 'running' | 'validated' | 'invalidated' | 'inconclusive' | 'invalid'


class Experiment:
    def __init__(self, name, parent_hypothesis, file_path: str | None = None):
        self.name = name
        self.parent_hypothesis = parent_hypothesis
        self.created_at = time.time()
        self.id = uuid.uuid4()
        self.description = ""
        self.file_path = file_path
        self.status = "pending"  # 'pending' | 'running' | 'validated' | 'invalidated' | 'inconclusive' | 'invalid'
        self.runs: Dict[str, TrialRun] = {}


class Hypothesis:
    def __init__(self, name):
        self.name = name
        self.created_at = time.time()
        self.id = uuid.uuid4()
        self.description = ""  # long message / hypothesis statement
        # Conclusion metadata
        self.conclusion_status = None  # 'supports' | 'refutes' | 'inconclusive' | None
        self.conclusion_notes = ""
        self.concluded_at = None  # timestamp or None
        # Tree
        self.children: list[Hypothesis] = []
        self.experiments: list[Experiment] = []

    def __repr__(self):
        return (
            f"Hypothesis(name={self.name}, created_at={self.created_at}, "
            f"id={self.id}, description={self.description})"
        )


# ---- Serialization helpers ----

def serialize_trial_run(trial_run: TrialRun) -> Dict[str, Any]:
    return {
        "trials": trial_run.trials,
        "results": trial_run.results,
        "verification_results": {
            k: (v.value if isinstance(v, VerificationResult) else v)
            for k, v in trial_run.verification_results.items()
        },
        "status": trial_run.status,
    }


def deserialize_trial_run(data: Dict[str, Any]) -> TrialRun:
    tr = TrialRun(data["trials"])
    tr.results = data.get("results", {})
    tr.verification_results = {
        k: (VerificationResult(v) if isinstance(v, str) else v)
        for k, v in data.get("verification_results", {}).items()
    }
    tr.status = data.get("status", "pending")
    return tr


def serialize_experiment(experiment: Experiment) -> Dict[str, Any]:
    return {
        "name": experiment.name,
        "created_at": experiment.created_at,
        "id": str(experiment.id),
        "description": experiment.description,
        "file_path": experiment.file_path,
        "status": experiment.status,
        "runs": {run_id: serialize_trial_run(run) for run_id, run in experiment.runs.items()},
    }


def deserialize_experiment(data: Dict[str, Any]) -> Experiment:
    exp = Experiment(data["name"], None, data.get("file_path"))
    exp.created_at = data["created_at"]
    exp.id = uuid.UUID(data["id"])  # type: ignore[arg-type]
    exp.description = data.get("description", "")
    exp.status = data.get("status", "pending")
    exp.runs = {
        run_id: deserialize_trial_run(run_data)
        for run_id, run_data in data.get("runs", {}).items()
    }
    return exp


def serialize_hypothesis(hypothesis: Hypothesis) -> Dict[str, Any]:
    return {
        "name": hypothesis.name,
        "created_at": hypothesis.created_at,
        "id": str(hypothesis.id),
        "description": hypothesis.description,
        "conclusion_status": hypothesis.conclusion_status,
        "conclusion_notes": hypothesis.conclusion_notes,
        "concluded_at": hypothesis.concluded_at,
        "children": [serialize_hypothesis(child) for child in hypothesis.children],
        "experiments": [serialize_experiment(exp) for exp in hypothesis.experiments],
    }


def deserialize_hypothesis(data: Dict[str, Any]) -> Hypothesis:
    hypo = Hypothesis(data["name"])
    hypo.created_at = data["created_at"]
    hypo.id = uuid.UUID(data["id"])  # type: ignore[arg-type]
    hypo.description = data.get("description", "")
    hypo.conclusion_status = data.get("conclusion_status", None)
    hypo.conclusion_notes = data.get("conclusion_notes", "")
    hypo.concluded_at = data.get("concluded_at", None)
    hypo.children = [deserialize_hypothesis(child) for child in data.get("children", [])]
    hypo.experiments = [deserialize_experiment(exp) for exp in data.get("experiments", [])]
    for exp in hypo.experiments:
        exp.parent_hypothesis = hypo
    return hypo


__all__ = [
    "TrialRun",
    "Experiment",
    "Hypothesis",
    "serialize_trial_run",
    "deserialize_trial_run",
    "serialize_experiment",
    "deserialize_experiment",
    "serialize_hypothesis",
    "deserialize_hypothesis",
]