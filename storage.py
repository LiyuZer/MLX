from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from typing import Optional, Tuple

from models import (
    Hypothesis,
    serialize_hypothesis,
    deserialize_hypothesis,
)


def _find_hypothesis_by_id(hypothesis: Hypothesis, hypo_id: uuid.UUID) -> Optional[Hypothesis]:
    if hypothesis.id == hypo_id:
        return hypothesis
    for child in hypothesis.children:
        result = _find_hypothesis_by_id(child, hypo_id)
        if result:
            return result
    return None


def store_hypothesis_tree(
    mlx_dir: str,
    root_hypothesis: Optional[Hypothesis],
    head_hypothesis: Optional[Hypothesis],
    environment_path: Optional[str],
) -> None:
    os.makedirs(mlx_dir, exist_ok=True)
    tree_path = os.path.join(mlx_dir, "mlx.tree")
    data: dict = {}
    if root_hypothesis is not None:
        data["hypothesis_tree"] = serialize_hypothesis(root_hypothesis)
        data["head_hypothesis_id"] = str(head_hypothesis.id) if head_hypothesis else None
    else:
        data["hypothesis_tree"] = None
        data["head_hypothesis_id"] = None
    data["environment_path"] = environment_path
    with open(tree_path, "w") as f:
        json.dump(data, f, indent=4)


def load_hypothesis_tree(
    mlx_dir: str,
) -> Tuple[Optional[Hypothesis], Optional[Hypothesis], Optional[str]]:
    tree_path = os.path.join(mlx_dir, "mlx.tree")
    if not os.path.exists(tree_path):
        return None, None, None

    with open(tree_path, "r") as f:
        data = json.load(f)

    # Handle empty/uninitialized tree gracefully
    if not data.get("hypothesis_tree"):
        return None, None, data.get("environment_path", None)

    root = deserialize_hypothesis(data["hypothesis_tree"])  # type: ignore[index]
    env_path = data.get("environment_path", None)

    head_h: Optional[Hypothesis] = None
    head_id_str = data.get("head_hypothesis_id")
    if head_id_str:
        try:
            head_id = uuid.UUID(head_id_str)
            head_h = _find_hypothesis_by_id(root, head_id)
        except Exception:
            head_h = None
    return root, head_h, env_path


def setup_environment(mlx_dir: str, env_path: Optional[str] = None) -> Optional[str]:
    """Update the persisted environment_path in .mlx/mlx.tree.

    Behavior matches MLXCore.setup_environment prior to refactor:
    - If env_path is a valid directory, persist it.
    - If invalid or None, keep the existing persisted value and print a notice.
    - Preserve hypothesis_tree and head_hypothesis_id as-is.

    Returns the environment_path value that was persisted.
    """
    os.makedirs(mlx_dir, exist_ok=True)
    tree_path = os.path.join(mlx_dir, "mlx.tree")

    data: dict = {}
    if os.path.exists(tree_path):
        try:
            with open(tree_path, "r") as f:
                data = json.load(f)
        except Exception:
            # If the file is corrupted, fall back to an empty structure
            data = {}

    existing_env = data.get("environment_path", None)

    # Decide the new environment value
    if env_path and os.path.isdir(env_path):
        new_env = env_path
    else:
        # Mirror previous user feedback behavior
        if env_path is not None and not os.path.isdir(env_path):
            print("No valid environment path provided, using default.")
        new_env = existing_env

    # Ensure keys exist and preserve existing hypothesis data
    data.setdefault("hypothesis_tree", None)
    data.setdefault("head_hypothesis_id", None)
    data["environment_path"] = new_env

    with open(tree_path, "w") as f:
        json.dump(data, f, indent=4)

    return new_env


def cleanup_orphan_temp_dirs(temp_root_dir: str, max_age_hours: int = 24) -> None:
    """Remove leftover run directories older than max_age_hours under temp_root_dir.
    Best-effort cleanup to prevent disk clutter from unexpected terminations.
    """
    if not os.path.isdir(temp_root_dir):
        return
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    for name in os.listdir(temp_root_dir):
        path = os.path.join(temp_root_dir, name)
        try:
            if os.path.isdir(path):
                mtime = os.path.getmtime(path)
                if now - mtime > max_age_seconds:
                    shutil.rmtree(path, ignore_errors=True)
        except Exception:
            # Best-effort; ignore errors to avoid disrupting runs
            pass


__all__ = [
    "store_hypothesis_tree",
    "load_hypothesis_tree",
    "setup_environment",
    "cleanup_orphan_temp_dirs",
]