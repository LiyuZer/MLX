# CLI Usage

Initialize a project:
- mlx init

Hypotheses:
- Create: mlx hypothesis create -m "Title\nDetailed description"
- List:   mlx hypothesis list
- Show:   mlx hypothesis show <id_prefix>
- Switch: mlx hypothesis switch <id_prefix>
- Set description: mlx hypothesis set-desc <id_prefix> -m "..."
- Conclude: mlx hypothesis conclude <id_prefix> --status supports|refutes|inconclusive -m "notes"
- Delete: mlx hypothesis delete <id_prefix> [--force]
- Back-compat: mlx hypothesis -m "..." creates a new hypothesis if no subcommand specified.

Experiments:
- List:   mlx exp list
- Show:   mlx exp show <id_prefix>
- Add:    mlx exp add <file_path> <name>
- Delete: mlx exp delete <id_prefix>
- Shorthand show: mlx exp <id_prefix>
- Legacy add:     mlx exp <file_path> <name>

Run, reproduce, status, log:
- Run a single experiment:
  - mlx run <experiment_id>
  - Options:
    - --stream-all        Stream output for every parameter combination during trials (sets MLX_STREAM_ALL=1)
    - --parallel N        Run N combinations in parallel per trial (sets MLX_PARALLEL=N). Defaults to 1 (serial)
  - Notes: If a trial declares "__parallel__" in its values (or MLX_PARALLEL is set), it will run combinations in parallel. Per-combination streaming is disabled in parallel mode to keep logs readable; progress updates when combinations complete.

- Reproduce the whole tree (sequentially by experiment):
  - mlx reproduce [--dry-run] [--fail-fast] [--filter-status any|pending|running|validated|invalidated|inconclusive|invalid] [--stream-all]
  - Behavior:
    - Walks the hypothesis tree in creation order and runs each experiment sequentially.
    - Inside each experiment, per-trial parallelism declared via "__parallel__" or MLX_PARALLEL is preserved.
    - --stream-all enables per-combination streaming for sequential trial execution; in parallel trials, streaming per combination is disabled (progress updates per completion).
    - --dry-run lists the plan without executing.
    - --fail-fast stops on first failure.
    - --filter-status limits which experiments are run by their current status (default: any).

- Status: mlx status <experiment_id>
- Log:    mlx log

Environment notes:
- By default, the runner uses your current Python (python3). If you configured an environment path in MLXCore, it uses that interpreter instead.