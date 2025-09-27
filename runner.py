from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

from colorama import Style


# Note: This module extracts heavy helper logic from MLXCore to keep mlx_core.py slim.
# Public functions mirror former MLXCore methods so MLXCore can delegate to them.


def make_function(
    func_name: str,
    func_body: str,
    arg_names: List[str],
    arg_values: List[Any],
    import_lines: str,
) -> str:
    """
    Build a small Python program string that defines a function and invokes it,
    printing a JSON result. Preserved from original implementation for API parity.
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
    """Extract trials, verifications, import lines, and other code segments from a file."""
    tree = ast.parse(file_content)
    trial_funcs: List[Tuple[int, str, Dict[str, List[Any]], str]] = []
    verify_funcs = []
    import_lines: List[str] = []
    other_code: List[str] = []  # Classes, helper functions, constants

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_lines.append(ast.get_source_segment(file_content, node))
        elif isinstance(node, ast.FunctionDef):
            # Check if this function has trial or verify decorators
            has_trial = False
            has_verify = False

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
                            trial_funcs.append((order, node.name, values, func_body))
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

            # If function doesn't have trial/verify decorators, include it as helper code
            if not has_trial and not has_verify:
                seg = ast.get_source_segment(file_content, node)
                if seg is not None:
                    other_code.append(seg)
        else:
            # Include classes, variables, and other top-level code
            seg = ast.get_source_segment(file_content, node)
            if seg is not None:
                other_code.append(seg)

    trial_funcs.sort(key=lambda x: x[0])  # Sort by order
    return trial_funcs, verify_funcs, import_lines, other_code


def create_trial_temp_file(
    func_name: str,
    func_body: str,
    arg_names: List[str],
    arg_values: List[Any],
    import_lines: List[str],
    other_code: List[str],
    base_dir: str,
    temp_dir: str | None = None,
) -> str:
    """Create a temporary file for executing a single trial"""
    # Build the complete file content
    file_content: List[str] = []

    # Add imports
    file_content.extend(import_lines)
    file_content.append("import json")
    file_content.append("import sys")
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
    temp_dir: str | None = None,
) -> str:
    """Create a temporary file for executing a verification"""
    file_content: List[str] = []

    # Add imports
    file_content.extend(import_lines)
    file_content.append("import json")
    file_content.append("import sys")
    file_content.append("from enum import Enum")
    file_content.append("")

    # Add VerificationResult enum
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


def execute_temp_file(
    temp_file_path: str,
    python_exe: str,
    base_dir: str,
    stream_output: bool = True,
):
    """Execute a temporary file and return results with proper error handling"""
    try:
        # Build PYTHONPATH to include base_dir, repo_root, lib_root, and existing PYTHONPATH
        repo_root = os.getcwd()
        lib_root = os.path.dirname(os.path.abspath(__file__))
        existing_pp = os.environ.get('PYTHONPATH', '')
        path_sep = os.pathsep
        pythonpath_parts = [p for p in [base_dir, repo_root, lib_root] if p]
        pythonpath = path_sep.join(pythonpath_parts)
        if existing_pp:
            pythonpath = pythonpath + path_sep + existing_pp
        env_vars = {**os.environ, 'PYTHONPATH': pythonpath}

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

        if stream_output:
            # Stream stdout for single runs
            for line in iter(process.stdout.readline, ''):
                if line and not line.startswith('{"result":'):
                    print(f"    {Style.DIM}│ {line.rstrip()}{Style.RESET_ALL}")
                output_lines.append(line)
            # Drain stderr after stdout completes
            stderr = process.stderr.read()
            if stderr:
                error_lines.append(stderr)
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
