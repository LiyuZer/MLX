# Troubleshooting

- Running by ID shows a different file name than expected
  - IDs map to experiments stored in .mlx. Use `mlx exp show <id>` to inspect the mapping.
  - To run a new local file, add it via `mlx exp add` and run the new ID.

- No prints during run
  - For multiple parameter combinations, enable streaming with `MLX_STREAM_ALL=1`.
  - tqdm writes to stderr by default; stderr is streamed. If you still don't see updates, add newline prints.

- Progress bar at 0% for a while
  - The bar advances after each combination completes. Long combos may hold the bar. Use `MLX_STREAM_ALL=1` to watch intermediate prints.

- Packaged binary fails with ModuleNotFoundError: runner
  - Rebuild using the provided spec (mlx_init.spec) and ensure hiddenimports include local modules.

- Environment mismatch
  - Verify which Python is used. If MLXCore.environment_path is set, the runner uses that interpreter.
