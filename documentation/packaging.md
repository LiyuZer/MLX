# Packaging with PyInstaller

A spec file is provided: mlx_init.spec. It includes local modules via hiddenimports.

Steps:
1) pip install pyinstaller
2) rm -rf build dist
3) pyinstaller mlx_init.spec
4) Test: ./dist/mlx_init -h
5) Run:  MLX_STREAM_ALL=1 ./dist/mlx_init run <id>

If you see ModuleNotFoundError for local modules (e.g., runner):
- Ensure you're building from the repository root.
- Confirm hiddenimports in mlx_init.spec include: runner, models, decorators, storage, view, cli (and any third-party modules like tqdm, colorama).
