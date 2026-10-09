# SWE Agent Environment

A small Python environment for running agent-supplied commands inside an existing software repository. It is a lightweight scaffold, not a sandbox or a task-completion evaluator.

## Current behavior

- `SWEEnv(repo_path)` validates and records an existing directory.
- `reset()` returns a `SWEObservation` containing a sorted top-level file listing and an info dictionary.
- `step(command)` or `step(SWEAction(command))` runs that shell command in the repository and returns a `StepResult` with the observation, reward, termination, truncation, and info fields.
- The observation contains actual stdout, stderr, exit code, and—inside a Git work tree—a read-only diff of tracked staged and unstaged changes.
- Commands have a timeout and episodes have a configurable step limit.

The reward is `0.0` for exit code zero and `-1.0` otherwise. It reflects command execution only; it does **not** indicate that a coding task was solved. No task-specific evaluator is configured. Untracked file contents are not included in Git diffs.

## Safety and limitations

Commands run with the current user's permissions and can modify the repository. There is no container, filesystem isolation, automatic cleanup, or rollback. The environment does not run destructive Git commands itself and never resets or cleans the working tree. Review commands before executing them.

Gymnasium compatibility, task-specific rewards, and stronger isolation are not implemented. Git support uses the `git` executable when available; Git is not a Python dependency.

## Example

Run from the project root with the package importable:

```bash
python3 examples/basic.py
```

The example uses the project directory itself and runs the current Python interpreter with `-m pytest -q` there.

## Tests

```bash
python -m pytest
```
