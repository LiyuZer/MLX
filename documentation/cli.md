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

Run, status, log:
- Run:    mlx run <experiment_id>
- Status: mlx status <experiment_id>
- Log:    mlx log

Environment notes:
- By default, uses your current Python (python3). If you configured an environment path in MLXCore, it uses that.
