---
name: vscode-debug-launch
description: Create or update VS Code launch.json debug configurations when the user mentions "vscode debug". Mirror what a .sh script runs (python or torchrun) into a proper Python debug config, with matching default args/env. Add a launch item named after the script (e.g., train.sh -> train).
---

# VS Code debug launch items for shell scripts

## Workflow

1) Locate the target .sh file path (usually in the workspace root). If multiple .sh files are mentioned, handle each one separately.
2) Open `.vscode/launch.json`. If it does not exist, create it with a valid `version` and `configurations` array.
3) Parse the .sh script:
   - Collect `export KEY=VALUE` lines into `env`.
   - Identify the actual execution command (usually `python ...` or `torchrun ...`).
   - Prefer the "interactive/local" branch if the script has SLURM logic; mirror the defaults from that branch.
4) Ensure there is a configuration object per script with:
   - `name`: the script base name without `.sh`.
   - `type`: `debugpy`.
   - `request`: `launch`.
5) If the script runs **python directly**:
   - Use `program` with the python entrypoint (e.g., `train.py`).
   - Put the remaining CLI args into `args`.
6) If the script runs **torchrun / distributed**:
   - Use `module`: `torch.distributed.run`.
   - Convert torchrun flags to `args` in order, followed by the python entrypoint and its args.
7) Always include:
   - `env` from the script exports.
   - `cwd`: `${workspaceFolder}`.
   - `console`: `integratedTerminal`.
   - `justMyCode`: `false`.

## Editing rules

- Preserve existing configurations; append new items if missing.
- If a config with the same `name` already exists, update its `program`/`module` + `args` + `env` to match the latest request.
- Keep JSON formatting consistent with the existing file.

## Minimal template if missing

If `.vscode/launch.json` is missing, create:

```
{
  "version": "0.2.0",
  "configurations": []
}
```
