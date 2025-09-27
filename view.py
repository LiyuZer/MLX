from __future__ import annotations

import sys
from colorama import Fore, Style
from models import Hypothesis


def print_hypothesis_tree(root_hypothesis: Hypothesis | None, head_hypothesis: Hypothesis | None) -> None:
    """
    Render the hypothesis tree and legend to the console.
    Mirrors previous MLXCore.print_log output exactly to preserve behavior.
    """

    def supports_unicode() -> bool:
        enc = getattr(sys.stdout, "encoding", "") or ""
        return "utf" in enc.lower()

    if root_hypothesis is None:
        print(f"{Fore.RED}No hypothesis tree initialized.{Style.RESET_ALL}")
        return

    uni = supports_unicode()

    # Symbol set depending on console capabilities
    branch_mid = "├── " if uni else "|-- "
    branch_last = "└── " if uni else "`-- "
    vert = "│   " if uni else "|   "
    bullet = "▪" if uni else "*"
    current_sym = "●" if uni else "*"
    inactive_sym = "○" if uni else "o"
    chk = "✓" if uni else "OK"
    cross = "✗" if uni else "X"
    warn = "⚠" if uni else "!"
    q = "?"

    def print_hypo(hypo: Hypothesis, level: int = 0, is_last: bool = False, parent_lines: str = ""):
        # Determine the branch characters
        if level == 0:
            branch = ""
            current_lines = ""
        else:
            branch = branch_last if is_last else branch_mid
            current_lines = parent_lines + ("    " if is_last else vert)

        # Determine hypothesis color
        if head_hypothesis and hypo.id == head_hypothesis.id:
            color = Fore.GREEN
            symbol = current_sym  # Current/active
        else:
            color = Fore.YELLOW
            symbol = inactive_sym  # Inactive

        # Hypothesis main line
        print(
            f"{parent_lines}{branch}{color}{symbol} {hypo.name}{Style.RESET_ALL} "
            f"{Style.DIM}<{str(hypo.id)[:8]}>{Style.RESET_ALL}"
        )

        # Optional description snippet
        if getattr(hypo, "description", None):
            snippet = hypo.description.strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            print(f"{current_lines}{Style.DIM}   {snippet}{Style.RESET_ALL}")

        # Optional conclusion
        if getattr(hypo, "conclusion_status", None):
            s = hypo.conclusion_status
            label_map = {
                "supports": (Fore.GREEN, f"{chk} SUPPORTS"),
                "refutes": (Fore.RED, f"{cross} REFUTES"),
                "inconclusive": (Fore.YELLOW, f"{q} INCONCLUSIVE"),
            }
            c, lab = label_map.get(s, (Fore.BLUE, f"• {s}"))
            print(f"{current_lines}Conclusion: {c}{lab}{Style.RESET_ALL}")

        # Print experiments under this hypothesis
        for i, exp in enumerate(hypo.experiments):
            is_last_exp = (i == len(hypo.experiments) - 1) and (len(hypo.children) == 0)
            exp_branch = branch_last if is_last_exp else branch_mid

            # Status symbols and colors
            status_symbols = {
                "pending": (Fore.YELLOW, "⧗" if uni else "."),
                "running": (Fore.CYAN, "⟳" if uni else "~"),
                "validated": (Fore.GREEN, chk),
                "invalidated": (Fore.RED, cross),
                "inconclusive": (Fore.YELLOW, q),
                "invalid": (Fore.RED, warn),
            }

            status_color, status_symbol = status_symbols.get(
                exp.status, (Fore.WHITE, "•" if uni else "*")
            )

            # Print experiment with status
            print(
                f"{current_lines}{exp_branch}{Fore.BLUE}{bullet} {exp.name}{Style.RESET_ALL} "
                f"{Style.DIM}<{str(exp.id)[:8]}>{Style.RESET_ALL} "
                f"{status_color}[{status_symbol} {exp.status}]{Style.RESET_ALL}"
            )

            # If experiment has runs, show count
            if hasattr(exp, "runs") and exp.runs:
                run_count = len(exp.runs)
                latest_run = list(exp.runs.keys())[-1]
                spacer = "    " if is_last_exp else vert
                print(
                    f"{current_lines}{spacer}{Style.DIM}  {run_count} run(s), latest: {latest_run}{Style.RESET_ALL}"
                )

        # Print child hypotheses
        for i, child in enumerate(hypo.children):
            is_last_child = i == len(hypo.children) - 1
            print_hypo(child, level + 1, is_last_child, current_lines)

    # Print header
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}HYPOTHESIS TREE{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    print_hypo(root_hypothesis)

    # Print legend
    print(f"\n{Style.DIM}{'-'*60}{Style.RESET_ALL}")
    print(f"{Style.DIM}Legend:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}{current_sym}{Style.RESET_ALL} Current hypothesis")
    print(f"  {Fore.YELLOW}{inactive_sym}{Style.RESET_ALL} Inactive hypothesis")
    print(f"  {Fore.BLUE}{bullet}{Style.RESET_ALL} Experiment")
    print(
        f"  {Fore.GREEN}{chk}{Style.RESET_ALL} Validated  "
        f"{Fore.RED}{cross}{Style.RESET_ALL} Invalidated  "
        f"{Fore.YELLOW}{q}{Style.RESET_ALL} Inconclusive"
    )
    print(f"{Style.DIM}{'-'*60}{Style.RESET_ALL}\n")


__all__ = ["print_hypothesis_tree"]
