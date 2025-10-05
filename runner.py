from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
import shutil
import atexit
import signal
import itertools as it
from typing import Any, Dict, List, Tuple, Optional

from colorama import Fore, Style
from tqdm import tqdm

# Note: This module extracts heavy helper logic from MLXCore to keep mlx_core.py slim.
# Public functions mirror former MLXCore methods so MLXCore can delegate to them.

from decorators import VerificationResult
from models import TrialRun


# ---------------------------------------
# Parsing helpers
# ---------------------------------------

def make_function(
    func_name: str,
    func_body: str,
    arg_names: List[str],
    arg_values: List[Any],
    import_lines: str,
) -> str:
    """
    Build a small Python program string that defines a function and invokes it,
    printing a JSON result. Preserved for API parity (not used in current flow).
    """
    func_string = f"{import_lines}\n"
    func_string += "import json\n"
    func_string += f"def {func_name}({', '.join(arg_names)}):\n"
    for line in func_body.splitlines():
        func_string += "    " + line + "\n"

    func_wrapper_name = f"wrapper_{func_name}"
    func_string += f"def {func_wrapper_name}():\n"
    function_inside_args = [f"{name}={repr(value)}" for name, value in zip(arg_names, arg_values)]

    func_string += f"    print(json.dumps({{ 'result': {func_name}({', '.join(function_inside_args)}) }}))\n"
    func_string += f"    return\n"
    func_string += f"{func_wrapper_name}()\n"
    return func_string


def extract_file_components(file_content: str):
    """
    Extract trials, verifications, setup functions, import lines, and other code segments from a file.

    Returns:
      trial_funcs: List[Tuple[int, str, Dict[str, List[Any]], str]]
      verify_funcs: List[Tuple[str, List[str], ast.FunctionDef]]
      setup_funcs: List[Tuple[Optional[int], str, ast.FunctionDef]]
      import_lines: List[str]
      other_code: List[str]
    """
    tree = ast.parse(file_content)
    trial_funcs: List[Tuple[int, str, Dict[str, List[Any]], str]] = []
    verify_funcs: List[Tuple[str, List[str], ast.FunctionDef]] = []
    setup_funcs: List[Tuple[Optional[int], str, ast.FunctionDef]] = []
    import_lines: List[str] = []
    other_code: List[str] = []  # Classes, helper functions, constants

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(file_content, node)
            if seg is not None:
                import_lines.append(seg)
        elif isinstance(node, ast.FunctionDef):
            # Check decorators
            has_trial = False
            has_verify = False
            has_setup = False

            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    func_id = getattr(decorator.func, 'id', None)
                    if func_id == 'trial':
                        order = None
                        values = None
                        # Keyword args
                        for kw in decorator.keywords:
                            if kw.arg == 'order':
                                order = ast.literal_eval(kw.value)
                            elif kw.arg == 'values':
                                values = ast.literal_eval(kw.value)
                        # Positional args fallback: trial(order, values)
                        if (order is None or values is None) and getattr(decorator, 'args', None):
                            if len(decorator.args) >= 1 and order is None:
                                order = ast.literal_eval(decorator.args[0])
                            if len(decorator.args) >= 2 and values is None:
                                values = ast.literal_eval(decorator.args[1])
                        if order is not None and values is not None:
                            func_body = "\n".join([ast.unparse(stmt) for stmt in node.body])
                            # Normalize order to int for sorting (fallback to large number)
                            order_val = int(order)
                            trial_funcs.append((order_val, node.name, values, func_body))
                            has_trial = True
                    elif func_id == 'verify':
                        trial_names: List[str] = []
                        # Keyword arg: verify(trial_names=[...])
                        for kw in decorator.keywords:
                            if kw.arg == 'trial_names':
                                trial_names = ast.literal_eval(kw.value)
                        # Positional arg: verify([ ... ])
                        if not trial_names and getattr(decorator, 'args', None):
                            if len(decorator.args) >= 1:
                                trial_names = ast.literal_eval(decorator.args[0])
                        verify_funcs.append((node.name, trial_names, node))
                        has_verify = True
                    elif func_id == 'setup':
                        order: Optional[int] = None
                        # Keyword arg: setup(order=...)
                        for kw in decorator.keywords:
                            if kw.arg == 'order':
                                order = ast.literal_eval(kw.value)
                        # Positional arg: setup(order)
                        if order is None and getattr(decorator, 'args', None):
                            if len(decorator.args) >= 1:
                                order = ast.literal_eval(decorator.args[0])
                        setup_funcs.append((order, node.name, node))
                        has_setup = True

            # If function doesn't have trial/verify/setup decorators, include it as helper code
            if not has_trial and not has_verify and not has_setup:
                seg = ast.get_source_segment(file_content, node)
                if seg is not None:
                    other_code.append(seg)
        else:
            # Include classes, variables, and other top-level code
            seg = ast.get_source_segment(file_content, node)
            if seg is not None:
                other_code.append(seg)

    # Sort trials by order
    trial_funcs.sort(key=lambda x: x[0])
    # Sort setup funcs: those with order first by order, otherwise preserve discovery order
    setup_with_order = [(o, n, nd) for (o, n, nd) in setup_funcs if o is not None]
    setup_no_order = [(o, n, nd) for (o, n, nd) in setup_funcs if o is None]
    setup_with_order.sort(key=lambda x: x[0])
    setup_funcs_sorted: List[Tuple[Optional[int], str, ast.FunctionDef]] = setup_with_order + setup_no_order

    return trial_funcs, verify_funcs, setup_funcs_sorted, import_lines, other_code


# ---------------------------------------
# Temp file builders (trial / verify / setup)
# ---------------------------------------

def _seed_block_lines() -> List[str]:
    """Generate seeding code lines that use a global SETUP dict if present."""
    lines: List[str] = []
    # Safe best-effort seeding based on SETUP
    lines.append("# --- MLX auto-seeding (best-effort) ---")
    lines.append("seed = (SETUP.get('seed') if isinstance(SETUP, dict) else None)")
    lines.append("numpy_seed = SETUP.get('numpy_seed', seed) if isinstance(SETUP, dict) else None")
    lines.append("torch_seed = SETUP.get('torch_seed', seed) if isinstance(SETUP, dict) else None")
    lines.append("pythonhashseed = SETUP.get('pythonhashseed', seed) if isinstance(SETUP, dict) else None")
    lines.append("try:\n    import os as _os_seed\n    import random as _random_seed\n    if pythonhashseed is not None:\n        _os_seed.environ['PYTHONHASHSEED'] = str(int(pythonhashseed))\n    if seed is not None:\n        _random_seed.seed(int(seed))\nexcept Exception:\n    pass")
    lines.append("try:\n    import numpy as _np_seed\n    if numpy_seed is not None:\n        _np_seed.random.seed(int(numpy_seed))\nexcept Exception:\n    pass")
    lines.append("try:\n    import torch as _torch_seed\n    if torch_seed is not None:\n        _torch_seed.manual_seed(int(torch_seed))\n        if hasattr(_torch_seed, 'cuda') and _torch_seed.cuda.is_available():\n            _torch_seed.cuda.manual_seed_all(int(torch_seed))\n        try:\n            import torch.backends.cudnn as _cudnn\n            _cudnn.deterministic = True\n            _cudnn.benchmark = False\n        except Exception:\n            pass\nexcept Exception:\n    pass")
    lines.append("# --- end auto-seeding ---")
    return lines


def create_trial_temp_file(
    func_name: str,
    func_body: str,
    arg_names: List[str],
    arg_values: List[Any],
    import_lines: List[str],
    other_code: List[str],
    base_dir: str,
    run_context: Dict[str, Any],
    temp_dir: str | None = None,
) -> str:
    """Create a temporary file for executing a single trial."""
    # Build the complete file content
    file_content: List[str] = []

    # Add imports
    file_content.extend(import_lines)
    file_content.append("import json")
    file_content.append("import sys")
    file_content.append("import os")
    file_content.append("")

    # Inject setup context
    setup_json = json.dumps(run_context or {})
    file_content.append(f"SETUP = {setup_json}")
    file_content.extend(_seed_block_lines())
    file_content.append("")

    # Add other code (classes, helper functions, constants)
    file_content.extend(other_code)
    file_content.append("")

    # Add the trial function
    file_content.append(f"def {func_name}({', '.join(arg_names)}):")
    for line in func_body.splitlines():
        file_content.append("    " + line)
    file_content.append("")

    # Add wrapper function
    function_inside_args = [f"{name}={repr(value)}" for name, value in zip(arg_names, arg_values)]
    file_content.append(f"def wrapper_{func_name}():")
    file_content.append("    try:")
    file_content.append(f"        result = {func_name}({', '.join(function_inside_args)})")
    file_content.append("        print(json.dumps({\"result\": result}))")
    file_content.append("    except Exception as e:")
    file_content.append("        print(json.dumps({\"error\": str(e), \"type\": type(e).__name__}), file=sys.stderr)")
    file_content.append("        sys.exit(1)")
    file_content.append("")
    file_content.append(f"wrapper_{func_name}()")

    # Create temporary file
    directory = temp_dir if temp_dir else base_dir
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=directory, delete=False)
    temp_file.write('\n'.join(file_content))
    temp_file.close()

    return temp_file.name


def create_verify_temp_file(
    verify_name: str,
    verify_node: ast.FunctionDef,
    trial_results: Dict[str, Any],
    import_lines: List[str],
    other_code: List[str],
    base_dir: str,
    run_context: Dict[str, Any],
    temp_dir: str | None = None,
) -> str:
    """Create a temporary file for executing a verification."""
    file_content: List[str] = []

    # Add imports
    file_content.extend(import_lines)
    file_content.append("import json")
    file_content.append("import sys")
    file_content.append("from enum import Enum")
    file_content.append("")

    # Inject setup context
    setup_json = json.dumps(run_context or {})
    file_content.append(f"SETUP = {setup_json}")
    # (Seeding not strictly necessary here, but harmless)
    file_content.extend(_seed_block_lines())
    file_content.append("")

    # Add VerificationResult enum (self-contained)
    file_content.append("class VerificationResult(Enum):")
    file_content.append("    SUPPORTS = 'supports'")
    file_content.append("    REFUTES = 'refutes'")
    file_content.append("    INCONCLUSIVE = 'inconclusive'")
    file_content.append("    INVALID = 'invalid'")
    file_content.append("    CONTINUE = 'continue'")
    file_content.append("")

    # Add other code
    file_content.extend(other_code)
    file_content.append("")

    # Add verification function
    func_body = "\n".join([ast.unparse(stmt) for stmt in verify_node.body])
    file_content.append(f"def {verify_name}(results):")
    for line in func_body.splitlines():
        file_content.append("    " + line)
    file_content.append("")

    # Add wrapper
    file_content.append(f"def wrapper_{verify_name}():")
    file_content.append("    try:")
    file_content.append(f"        result = {verify_name}({repr(trial_results)})")
    file_content.append("        if hasattr(result, 'value'):")
    file_content.append("            result = result.value")
    file_content.append("        print(json.dumps({\"result\": result}))")
    file_content.append("    except Exception as e:")
    file_content.append("        print(json.dumps({\"error\": str(e), \"type\": type(e).__name__}), file=sys.stderr)")
    file_content.append("        sys.exit(1)")
    file_content.append("")
    file_content.append(f"wrapper_{verify_name}()")

    # Create temporary file
    directory = temp_dir if temp_dir else base_dir
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=directory, delete=False)
    temp_file.write('\n'.join(file_content))
    temp_file.close()

    return temp_file.name


def create_setup_temp_file(
    setup_name: str,
    setup_node: ast.FunctionDef,
    import_lines: List[str],
    other_code: List[str],
    base_dir: str,
    run_context: Dict[str, Any],
    temp_dir: str | None = None,
) -> str:
    """Create a temporary file for executing a setup function once to produce run context.

    We do NOT auto-seed before setup; users typically return seeds here.
    """
    file_content: List[str] = []

    # Add imports
    file_content.extend(import_lines)
    file_content.append("import json")
    file_content.append("import sys")
    file_content.append("")

    # Current (possibly empty) context is provided as SETUP for convenience
    setup_json = json.dumps(run_context or {})
    file_content.append(f"SETUP = {setup_json}")
    file_content.append("")

    # Add other code
    file_content.extend(other_code)
    file_content.append("")

    # Add setup function
    func_body = "\n".join([ast.unparse(stmt) for stmt in setup_node.body])
    file_content.append(f"def {setup_name}():")
    for line in func_body.splitlines():
        file_content.append("    " + line)
    file_content.append("")

    # Add wrapper
    file_content.append(f"def wrapper_{setup_name}():")
    file_content.append("    try:")
    file_content.append(f"        result = {setup_name}()")
    file_content.append("        print(json.dumps({\"result\": result}))")
    file_content.append("    except Exception as e:")
    file_content.append("        print(json.dumps({\"error\": str(e), \"type\": type(e).__name__}), file=sys.stderr)")
    file_content.append("        sys.exit(1)")
    file_content.append("")
    file_content.append(f"wrapper_{setup_name}()")

    # Create temporary file
    directory = temp_dir if temp_dir else base_dir
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=directory, delete=False)
    temp_file.write('\n'.join(file_content))
    temp_file.close()

    return temp_file.name


# ---------------------------------------
# Execution helper
# ---------------------------------------

def execute_temp_file(
    temp_file_path: str,
    python_exe: str,
    base_dir: str,
    stream_output: bool = True,
):
    """Execute a temporary file and return results with proper error handling"""
    try:
        import threading

        # Build PYTHONPATH to include base_dir, repo_root, lib_root, and existing PYTHONPATH
        repo_root = os.getcwd()
        lib_root = os.path.dirname(os.path.abspath(__file__))
        existing_pp = os.environ.get('PYTHONPATH', '')
        path_sep = os.pathsep
        pythonpath_parts = [p for p in [base_dir, repo_root, lib_root] if p]
        pythonpath = path_sep.join(pythonpath_parts)
        if existing_pp:
            pythonpath = pythonpath + path_sep + existing_pp
        env_vars = {**os.environ, 'PYTHONPATH': pythonpath, 'PYTHONUNBUFFERED': '1'}  # ensure child prints flush

        process = subprocess.Popen(
            [python_exe, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=base_dir,  # Set working directory
            env=env_vars,
        )

        output_lines: List[str] = []
        error_lines: List[str] = []

        def _reader(stream, sink: List[str], is_err: bool = False):
            try:
                for line in iter(stream.readline, ''):
                    if line:
                        # Stream line-by-line for live feedback
                        if stream_output:
                            if not is_err:
                                # Hide the final JSON result line from display
                                if not line.startswith('{"result":'):
                                    print(f"    {Style.DIM}│ {line.rstrip()}{Style.RESET_ALL}", flush=True)
                            else:
                                # Show stderr explicitly
                                print(f"    {Style.DIM}│ [stderr] {line.rstrip()}{Style.RESET_ALL}", flush=True)
                        sink.append(line)
            except Exception:
                # Best effort; avoid breaking on stream read errors
                pass

        if stream_output:
            # Read stdout and stderr concurrently to avoid deadlocks and to stream both
            t_out = threading.Thread(target=_reader, args=(process.stdout, output_lines, False), daemon=True)
            t_err = threading.Thread(target=_reader, args=(process.stderr, error_lines, True), daemon=True)
            t_out.start()
            t_err.start()
            t_out.join()
            t_err.join()
            # Ensure process has exited to get a valid return code
            process.wait()
        else:
            # Capture all output for batch runs
            stdout, stderr = process.communicate()
            output_lines = [stdout] if stdout else []
            error_lines = [stderr] if stderr else []

        # Close pipes explicitly to avoid resource warnings
        try:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        except Exception:
            pass

        # Parse results
        if process.returncode != 0:
            # Extract error information
            error_info = {'success': False, 'return_code': process.returncode}

            # Try to parse JSON error from stderr
            stderr_content = ''.join(error_lines).strip()
            if stderr_content:
                try:
                    error_json = json.loads(stderr_content)
                    error_info.update(error_json)
                except json.JSONDecodeError:
                    error_info['error'] = stderr_content

            return error_info
        else:
            # Success - parse output
            output = ''.join(output_lines).strip()
            try:
                # Get the last line which should contain our JSON result
                last_line = output.split('\n')[-1] if '\n' in output else output
                result_json = json.loads(last_line)
                result_json['success'] = True
                return result_json
            except json.JSONDecodeError:
                return {'success': True, 'result': output}

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }


# ---------------------------------------
# Orchestration
# ---------------------------------------

def run_experiment_impl(core: Any, experiment: Any):
    """
    Orchestrate an experiment run with support for:
      - @setup run-level context (JSON-serializable)
      - Automatic seeding (random/numpy/torch) per combination based on context
      - Parallel execution per-trial via reserved "__parallel__" values key (env MLX_PARALLEL)
      - Streaming stdout+stderr with optional MLX_STREAM_ALL toggle
    """
    # STEP 0: VALIDATION
    if not experiment:
        print(f"{Fore.RED}✗ Experiment not found.{Style.RESET_ALL}")
        return

    if not experiment.file_path or not os.path.isfile(experiment.file_path):
        print(f"{Fore.RED}✗ Experiment file '{experiment.file_path}' not found.{Style.RESET_ALL}")
        return

    # Prepare temp environment (cleanup orphans, create run dir, register cleanup handlers)
    core.cleanup_orphan_temp_dirs(max_age_hours=24)

    # Generate run id early and create a dedicated temp directory
    run_id = str(uuid.uuid4())[:8]
    run_temp_dir = os.path.join(core.temp_root_dir, f"run_{run_id}")
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
        core.store_hypothesis_tree()

        # STEP 1: PARSE AND EXTRACT
        print(f"{Fore.BLUE}◆ Parsing experiment structure...{Style.RESET_ALL}")
        with open(experiment.file_path, "r") as f:
            file_content = f.read()

        # Setup Python executable and base directory
        python_exe = "python3"
        if getattr(core, 'environment_path', None):
            python_exe = os.path.join(core.environment_path, "bin", "python")
            print(f"  {Style.DIM}Using environment: {core.environment_path}{Style.RESET_ALL}")

        base_dir = os.path.dirname(os.path.abspath(experiment.file_path))
        temp_files: list[str] = []  # Track temp files for cleanup (extra safety)

        # Extract components from file
        trial_funcs, verify_funcs, setup_funcs, import_lines, other_code = extract_file_components(file_content)

        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Found {len(trial_funcs)} trial(s), {len(verify_funcs)} verification(s)")
        if setup_funcs:
            print(f"  {Style.DIM}Setup functions: {len(setup_funcs)}{Style.RESET_ALL}")
        print(f"  {Style.DIM}Helper code segments: {len(other_code)}{Style.RESET_ALL}\n")

        # STEP 1b: RUN SETUP (once) to produce run-level context
        run_context: Dict[str, Any] = {}
        if setup_funcs:
            print(f"{Fore.BLUE}◆ Running setup...{Style.RESET_ALL}")
            for order, setup_name, setup_node in setup_funcs:
                temp_file_path = create_setup_temp_file(
                    setup_name,
                    setup_node,
                    import_lines,
                    other_code,
                    base_dir,
                    run_context,
                    temp_dir=run_temp_dir,
                )
                temp_files.append(temp_file_path)
                start_time = time.time()
                res = execute_temp_file(temp_file_path, python_exe, base_dir, True)
                elapsed = time.time() - start_time
                if not res.get("success", False):
                    error_msg = res.get("error", "Unknown error")
                    error_type = res.get("type", "Error")
                    print(f"    {Fore.RED}✗ {error_type}: {error_msg}{Style.RESET_ALL}")
                    print(f"    {Style.DIM}Setup '{setup_name}' failed; aborting run.{Style.RESET_ALL}")
                    return
                # Merge JSON-serializable context
                ctx = res.get("result")
                # Allow non-dict returns by wrapping under function name
                if isinstance(ctx, dict):
                    merge_candidate = ctx
                else:
                    merge_candidate = {setup_name: ctx}
                # Ensure serializable
                try:
                    json.dumps(merge_candidate)
                except Exception:
                    print(f"    {Fore.RED}✗ Setup '{setup_name}' returned a non-serializable object{Style.RESET_ALL}")
                    return
                run_context.update(merge_candidate)
                print(f"    {Fore.GREEN}✓ Setup '{setup_name}' completed in {elapsed:.3f}s{Style.RESET_ALL}")
            print()

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
            # Determine requested parallelism (reserved key) with env fallback
            parallel = 1
            env_parallel = os.environ.get("MLX_PARALLEL")
            if env_parallel:
                try:
                    parallel = int(env_parallel)
                except Exception:
                    parallel = 1
            if "__parallel__" in values:
                try:
                    parallel = int(values.pop("__parallel__"))
                except Exception:
                    pass
            max_workers = max(1, min(parallel, (os.cpu_count() or 1)))

            # Disallow reserved name 'setup' inside values
            if "setup" in values:
                print(f"  {Fore.RED}✗ Parameter name 'setup' is reserved and cannot be used in trial values.{Style.RESET_ALL}")
                return

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

            # Streaming mode note
            stream_all = (os.environ.get("MLX_STREAM_ALL") == "1")

            start_time = time.time()
            successful_runs = 0
            failed_runs = 0

            if max_workers > 1 and len(cartesian_product) > 0:
                # Parallel execution: disable per-combo streaming to keep logs readable
                print(f"  {Style.DIM}Parallel: {max_workers} worker(s); streaming: disabled (parallel){Style.RESET_ALL}")
                try:
                    import concurrent.futures as cf
                except Exception:
                    cf = None

                # Prepare tasks
                results_buffer: list[dict | None] = [None] * len(cartesian_product)

                if cf is None:
                    # Fallback to sequential if futures unavailable
                    for i, combination in enumerate(cartesian_product):
                        temp_file_path = create_trial_temp_file(
                            func_name,
                            func_body,
                            arg_names,
                            combination,
                            import_lines,
                            other_code,
                            base_dir,
                            run_context,
                            temp_dir=run_temp_dir,
                        )
                        temp_files.append(temp_file_path)

                        param_str = ", ".join([f"{k}={v}" for k, v in zip(arg_names, combination)])
                        if len(cartesian_product) > 1:
                            progress_bar.set_description(f"  Running with {param_str}")
                        else:
                            print(f"  {Style.DIM}Parameters: {param_str}{Style.RESET_ALL}")

                        result = execute_temp_file(temp_file_path, python_exe, base_dir, False)
                        results_buffer[i] = result
                        if len(cartesian_product) > 1:
                            progress_bar.update(1)
                else:
                    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_index = {}
                        for i, combination in enumerate(cartesian_product):
                            temp_file_path = create_trial_temp_file(
                                func_name,
                                func_body,
                                arg_names,
                                combination,
                                import_lines,
                                other_code,
                                base_dir,
                                run_context,
                                temp_dir=run_temp_dir,
                            )
                            temp_files.append(temp_file_path)
                            # No streaming per task in parallel mode
                            if len(cartesian_product) > 1:
                                param_str = ", ".join([f"{k}={v}" for k, v in zip(arg_names, combination)])
                                progress_bar.set_description(f"  Running with {param_str}")
                            fut = executor.submit(execute_temp_file, temp_file_path, python_exe, base_dir, False)
                            future_to_index[fut] = i

                        for fut in cf.as_completed(future_to_index):
                            i = future_to_index[fut]
                            try:
                                res = fut.result()
                            except Exception as e:
                                res = {"success": False, "error": str(e), "type": type(e).__name__}
                            results_buffer[i] = res
                            if len(cartesian_product) > 1:
                                progress_bar.update(1)

                # Collate results in order, merging any context updates post-batch
                for res in results_buffer:
                    res = res or {"success": False, "error": "Unknown error", "type": "Error"}
                    if not res.get("success", False):
                        trial_run.results[func_name].append(res)
                        failed_runs += 1
                    else:
                        # Handle success
                        trial_run.results[func_name].append(res)
                        successful_runs += 1
                        # Merge context update if provided
                        r = res.get("result")
                        if isinstance(r, dict) and isinstance(r.get("context_update"), dict):
                            try:
                                json.dumps(r["context_update"])  # ensure serializable
                                run_context.update(r["context_update"])  # post-batch merge
                            except Exception:
                                pass

            else:
                # Sequential execution (original behavior, with improved streaming)
                if max_workers > 1:
                    print(f"  {Style.DIM}Parallel requested: {parallel}, clamped to {max_workers}{Style.RESET_ALL}")
                if stream_all and len(cartesian_product) > 1:
                    print(f"  {Style.DIM}Streaming: all combinations (MLX_STREAM_ALL=1){Style.RESET_ALL}")
                elif len(cartesian_product) == 1:
                    print(f"  {Style.DIM}Streaming: single combination{Style.RESET_ALL}")

                for i, combination in enumerate(cartesian_product):
                    # Create temporary file for this trial run
                    temp_file_path = create_trial_temp_file(
                        func_name,
                        func_body,
                        arg_names,
                        combination,
                        import_lines,
                        other_code,
                        base_dir,
                        run_context,
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
                    stream_output = stream_all or (len(cartesian_product) == 1)
                    result = execute_temp_file(temp_file_path, python_exe, base_dir, stream_output)

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
                        # Merge context update immediately (sequential)
                        r = result.get("result")
                        if isinstance(r, dict) and isinstance(r.get("context_update"), dict):
                            try:
                                json.dumps(r["context_update"])  # ensure serializable
                                run_context.update(r["context_update"])  # immediate merge
                            except Exception:
                                pass

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
                temp_file_path = create_verify_temp_file(
                    verify_name,
                    verify_node,
                    trial_run.results,
                    import_lines,
                    other_code,
                    base_dir,
                    run_context,
                    temp_dir=run_temp_dir,
                )
                temp_files.append(temp_file_path)

                start_time = time.time()
                result = execute_temp_file(temp_file_path, python_exe, base_dir, True)
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
        core.store_hypothesis_tree()

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