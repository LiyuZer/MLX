import argparse
import os
import sys
import time

# Ensure local import works regardless of CWD or execution method
try:
    from mlx_core import *
except ModuleNotFoundError:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from mlx_core import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Initialize MLX environment.")
    subparsers = parser.add_subparsers(dest="command")

    # init
    subparsers.add_parser("init", help="Initialize a new MLX")

    # hypothesis group
    hypo_parser = subparsers.add_parser("hypothesis", help="Manage hypotheses")
    # Back-compat: allow `mlx hypothesis -m ...` to create
    hypo_parser.add_argument("-m", "--message", type=str, help="Message for the hypothesis (create if no subcommand)")
    hypo_parser.add_argument("-F", "--file", type=str, help="Path to a file containing the hypothesis message")
    hypo_sub = hypo_parser.add_subparsers(dest="hypo_cmd")

    hypo_create = hypo_sub.add_parser("create", help="Create a new hypothesis")
    hypo_create.add_argument("-m", "--message", type=str, help="Hypothesis message")
    hypo_create.add_argument("-F", "--file", type=str, help="Path to a file with the message")

    hypo_sub.add_parser("list", help="List hypotheses (tree)")

    hypo_show = hypo_sub.add_parser("show", help="Show a hypothesis detail")
    hypo_show.add_argument("id_prefix", type=str, help="Hypothesis ID prefix")

    hypo_switch = hypo_sub.add_parser("switch", help="Switch active head hypothesis")
    hypo_switch.add_argument("id_prefix", type=str, help="Hypothesis ID prefix")

    hypo_setdesc = hypo_sub.add_parser("set-desc", help="Update hypothesis description")
    hypo_setdesc.add_argument("id_prefix", type=str, help="Hypothesis ID prefix")
    hypo_setdesc.add_argument("-m", "--message", type=str, help="New description")
    hypo_setdesc.add_argument("-F", "--file", type=str, help="File containing new description")

    hypo_conclude = hypo_sub.add_parser("conclude", help="Conclude a hypothesis with a status and notes")
    hypo_conclude.add_argument("id_prefix", type=str, help="Hypothesis ID prefix")
    hypo_conclude.add_argument("--status", type=str, choices=["supports", "refutes", "inconclusive"], default=None,
                               help="Conclusion status")
    hypo_conclude.add_argument("-m", "--notes", type=str, help="Conclusion notes/message")
    hypo_conclude.add_argument("-F", "--file", type=str, help="File containing conclusion notes")

    # NEW: delete hypothesis
    hypo_delete = hypo_sub.add_parser("delete", help="Delete a hypothesis")
    hypo_delete.add_argument("id_prefix", type=str, help="Hypothesis ID prefix")
    hypo_delete.add_argument("--force", action="store_true",
                             help="Force deletion even if it has children/experiments")

    # experiment group (no subparsers — we parse remainder ourselves to support aliases/legacy)
    exp_parser = subparsers.add_parser("exp", help="Manage experiments")
    exp_parser.add_argument("rest", nargs=argparse.REMAINDER, help="exp subcommand and arguments")

    # run
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("experiment_id", type=str, help="ID of the experiment to run")

    # setup
    setup_parser = subparsers.add_parser("setup", help="Setup the MLX environment")
    setup_parser.add_argument("--env", type=str, help="Path to the environment setup folder")

    # status
    status_parser = subparsers.add_parser("status", help="Show status of experiments")
    status_parser.add_argument("experiment_id", type=str, help="ID of the experiment to check status")

    # log
    subparsers.add_parser("log", help="Show hypothesis log")

    return parser.parse_args()


def _read_file_text(path):
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"File '{path}' does not exist.")
        return None
    with open(path, "r") as f:
        return f.read()


def _resolve_message(args, msg_attr: str, file_attr: str):
    msg = getattr(args, msg_attr, None)
    if msg:
        return msg
    file_path = getattr(args, file_attr, None)
    if file_path:
        file_text = _read_file_text(file_path)
        if file_text is not None:
            return file_text
    return None


def _print_hypothesis_details(h):
    print(f"Name: {h.name}")
    print(f"ID: {str(h.id)}")
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h.created_at))
    print(f"Created: {ts}")
    desc = (h.description or '').strip()
    if desc:
        print("\nDescription:\n" + desc)
    if getattr(h, 'conclusion_status', None):
        print("\nConclusion:")
        print(f"  Status: {h.conclusion_status}")
        if h.concluded_at:
            cts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h.concluded_at))
            print(f"  At: {cts}")
        notes = (getattr(h, 'conclusion_notes', '') or '').strip()
        if notes:
            print("  Notes:\n" + notes)


def _list_all_experiments(core):
    items = []
    def walk(h):
        for e in h.experiments:
            items.append((h, e))
        for c in h.children:
            walk(c)
    if core.root_hypothesis:
        walk(core.root_hypothesis)
    return items


def _print_experiment_details(exp):
    print(f"Name: {exp.name}")
    print(f"ID: {str(exp.id)}")
    cts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp.created_at))
    print(f"Created: {cts}")
    print(f"File: {exp.file_path}")
    print(f"Status: {exp.status}")
    if exp.runs:
        latest = list(exp.runs.keys())[-1]
        print(f"Runs: {len(exp.runs)} (latest: {latest})")
    else:
        print("Runs: 0")


def main():
    args = parse_arguments()
    mlx_dir = ".mlx"

    if args.command == "init":
        if os.path.exists(mlx_dir):
            print(f"MLX environment already initialized in '{mlx_dir}'.")
            print("Use 'mlx hypothesis create -m " + '"Your hypothesis"' + "' to start.")
            return
        os.makedirs(mlx_dir, exist_ok=True)
        MLXCore(mlx_dir)
        print(f"Initialized new MLX environment in '{mlx_dir}'.")

    elif args.command == "hypothesis":
        if not os.path.exists(mlx_dir):
            print("MLX environment not initialized. Please run 'mlx init' first.")
            sys.exit(1)
        core = MLXCore(mlx_dir)

        # Back-compat: `mlx hypothesis -m "..."` creates a hypothesis
        if getattr(args, 'hypo_cmd', None) is None and (getattr(args, 'message', None) or getattr(args, 'file', None)):
            text = _resolve_message(args, 'message', 'file')
            new_hypo = core.create_new_hypothesis(text.splitlines()[0] if text else "New Hypothesis")
            if text:
                new_hypo.description = text
                core.store_hypothesis_tree()
            print(f"Created new hypothesis: {new_hypo.name} with ID {str(new_hypo.id)[:8]}")
            return

        cmd = getattr(args, 'hypo_cmd', None)
        if cmd == "create":
            text = _resolve_message(args, 'message', 'file')
            title = text.splitlines()[0] if text else "New Hypothesis"
            new_hypo = core.create_new_hypothesis(title)
            if text:
                new_hypo.description = text
                core.store_hypothesis_tree()
            print(f"Created new hypothesis: {new_hypo.name} with ID {str(new_hypo.id)[:8]}")

        elif cmd == "list":
            core.print_log()

        elif cmd == "show":
            h = core.find_hypothesis_by_prefix(args.id_prefix)
            if not h:
                print(f"Hypothesis with ID starting '{args.id_prefix}' not found.")
                sys.exit(1)
            _print_hypothesis_details(h)

        elif cmd == "switch":
            ok = core.set_head(args.id_prefix)
            if not ok:
                sys.exit(1)

        elif cmd == "set-desc":
            h = core.find_hypothesis_by_prefix(args.id_prefix)
            if not h:
                print(f"Hypothesis with ID starting '{args.id_prefix}' not found.")
                sys.exit(1)
            text = _resolve_message(args, 'message', 'file')
            if not text:
                print("No description provided (-m or -F).")
                sys.exit(1)
            h.description = text
            core.store_hypothesis_tree()
            print(f"Updated description for {h.name} <{str(h.id)[:8]}>")

        elif cmd == "conclude":
            notes = _resolve_message(args, 'notes', 'file')
            ok = core.conclude_hypothesis(args.id_prefix, args.status, notes)
            if not ok:
                sys.exit(1)

        elif cmd == "delete":
            ok = core.delete_hypothesis(args.id_prefix, args.force)
            if not ok:
                sys.exit(1)

        else:
            print("No hypothesis action specified. Use one of: create/list/show/switch/set-desc/conclude/delete")
            sys.exit(1)

    elif args.command == "exp":
        if not os.path.exists(mlx_dir):
            print("MLX environment not initialized. Please run 'mlx init' first.")
            sys.exit(1)
        core = MLXCore(mlx_dir)

        rest = [r for r in (args.rest or []) if r != '--']  # drop argparse '--' separator if present
        # Handle: add/list/show/delete explicitly
        if not rest:
            print("Usage:\n  mlx exp list\n  mlx exp show <id_prefix>\n  mlx exp <id_prefix>\n  mlx exp add <file_path> <name>\n  mlx exp delete <id_prefix>\n  mlx exp <file_path> <name>\n")
            sys.exit(1)

        cmd = rest[0]
        if cmd == 'list':
            items = _list_all_experiments(core)
            if not items:
                print("No experiments found.")
                return
            for h, e in items:
                print(f"{e.name} <{str(e.id)[:8]}> [{e.status}] under hypothesis: {h.name} <{str(h.id)[:8]}>")
            return

        if cmd == 'show':
            if len(rest) < 2:
                print("Usage: mlx exp show <id_prefix>")
                sys.exit(1)
            exp = core.find_experiment_by_id(rest[1])
            if not exp:
                print(f"Experiment with ID starting '{rest[1]}' not found.")
                sys.exit(1)
            _print_experiment_details(exp)
            return

        if cmd == 'add':
            if len(rest) < 3:
                print("Usage: mlx exp add <file_path> <name>")
                sys.exit(1)
            file_path = rest[1]
            name = " ".join(rest[2:]) if len(rest) > 2 else rest[2]
            if not os.path.isfile(file_path):
                print(f"File '{file_path}' does not exist.")
                sys.exit(1)
            new_exp = core.add_experiment(name, file_path)
            print(f"Added new experiment: {new_exp.name} with ID {str(new_exp.id)[:8]}")
            return

        if cmd == 'delete':
            if len(rest) < 2:
                print("Usage: mlx exp delete <id_prefix>")
                sys.exit(1)
            ok = core.delete_experiment(rest[1])
            if not ok:
                sys.exit(1)
            return

        # Shorthand show: mlx exp <id_prefix>
        if len(rest) == 1 and not cmd.startswith('-'):
            exp = core.find_experiment_by_id(cmd)
            if not exp:
                print(f"Experiment with ID starting '{cmd}' not found.")
                sys.exit(1)
            _print_experiment_details(exp)
            return

        # Legacy add: mlx exp <file_path> <name>
        if len(rest) >= 2 and os.path.isfile(rest[0]):
            file_path = rest[0]
            name = " ".join(rest[1:])
            new_exp = core.add_experiment(name, file_path)
            print(f"Added new experiment: {new_exp.name} with ID {str(new_exp.id)[:8]}")
            return

        print("Invalid exp usage. Try:\n  mlx exp list\n  mlx exp show <id_prefix>\n  mlx exp <id_prefix>\n  mlx exp add <file_path> <name>\n  mlx exp delete <id_prefix>\n  mlx exp <file_path> <name>")
        sys.exit(1)

    elif args.command == "setup":
        if not os.path.exists(mlx_dir):
            print("MLX environment not initialized. Please run 'mlx init' first.")
            sys.exit(1)
        core = MLXCore(mlx_dir)
        env_path = args.env if args.env else None
        core.setup_environment(env_path)
        print("Environment setup completed.")

    elif args.command == "run":
        if not os.path.exists(mlx_dir):
            print("MLX environment not initialized. Please run 'mlx init' first.")
            sys.exit(1)
        core = MLXCore(mlx_dir)
        core.run_experiment(args.experiment_id)

    elif args.command == "status":
        if not os.path.exists(mlx_dir):
            print("MLX environment not initialized. Please run 'mlx init' first.")
            sys.exit(1)
        core = MLXCore(mlx_dir)
        core.get_experiment_status(args.experiment_id)

    elif args.command == "log":
        if not os.path.exists(mlx_dir):
            print("MLX environment not initialized. Please run 'mlx init' first.")
            sys.exit(1)
        core = MLXCore(mlx_dir)
        core.print_log()

    else:
        print("No command specified. Use one of: init, hypothesis, exp, run, setup, status, log")
        sys.exit(1)


if __name__ == "__main__":
    main()