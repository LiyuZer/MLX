# Running and Streaming

Run an experiment by ID:
- mlx run <id_prefix>

Live output behavior:
- Single parameter combination: stdout and stderr lines are streamed live by default.
- Multiple combinations: enable live streaming for every combo by setting:
  MLX_STREAM_ALL=1 mlx run <id>

Notes:
- The child Python process is unbuffered, so print() appears immediately.
- tqdm defaults to stderr; stderr is streamed as well. Carriage-return animations may appear as periodic lines. For clearer logs, use `tqdm(..., file=sys.stdout)` or `tqdm.write()`.

Environment selection:
- The runner uses python3 by default.
- If MLXCore.environment_path is configured, it uses that environment's Python (path/bin/python).

ID vs file:
- Running by ID uses the experiment recorded in .mlx. Use `mlx exp show <id>` to view its file path.
- To run a new file, add it first: `mlx exp add path/to/file.py "Name"` then run its ID.
