# SWE Agent Environment

A small Python environment for running agent-supplied commands in a temporary copy of an existing repository. It is a lightweight scaffold, not a sandbox or a task-completion evaluator.

## Current behavior

- `SWEEnv(repo_path)` resolves and validates the source directory without copying it yet; directory symlinks resolve to their target.
- `reset()` creates a fresh temporary workspace, returns its sorted top-level listing and info with `repo_path` (source) and `workspace_path` (copy). A previous episode's changes do not survive reset.
- `step(command)` or `step(SWEAction(command))` runs that shell command in the active workspace and returns a `StepResult` with the observation, reward, termination, truncation, and info fields. Call `reset()` before stepping, including after `close()` or a failed reset.
- `close()` removes the workspace and is safe to call repeatedly; `SWEEnv` also supports `with SWEEnv(...) as env:` for explicit cleanup.
- The observation contains actual stdout, stderr, exit code, and—when the source has root `.git` metadata—a Git diff of tracked staged and unstaged changes against a private workspace snapshot.
- Commands have a timeout and episodes have a configurable step limit. Rejected actions do not consume a step; valid commands consume one step even if launch fails or they time out.

`SWEEnv(..., command_timeout=30.0, output_limit_bytes=1_000_000)` limits **each** of stdout and stderr independently to the first configured number of bytes while reading; both pipes are drained concurrently. `step().info` includes `stdout_truncated`, `stderr_truncated`, `output_limit_bytes`, `timed_out`, and `error`. Captured bytes are decoded as UTF-8 with replacement for invalid or split characters. A timeout reports exit code 124 and preserves captured output; a process-launch failure reports exit code 127 and a descriptive `error`. Shell commands naming a missing program typically return the shell's nonzero status (usually 127) with its message on stderr. The episode's `truncated` field still means the *step limit*, not output truncation. The reward is `0.0` for exit code zero and `-1.0` otherwise. It reflects command execution only; it does **not** indicate that a coding task was solved. No task-specific evaluator is configured. Untracked file contents created during an episode are not included in Git diffs. Initial uncommitted and untracked files are included in the snapshot, so the initial diff is empty rather than showing the source's pre-existing changes. Original Git history, branches, ignores, and configuration are not carried over; Git must be installed for Git-backed source snapshots.

## Safety and limitations

The complete source tree is copied, including untracked files, ignored files, virtual environments, caches, and build artifacts; large trees therefore take time and disk space. Root `.git` metadata is excluded and Git-backed sources get a new private Git baseline. Ordinary relative-path edits stay in the copy. Regular file permissions and timestamps are copied where supported; symlinks are copied as links without traversing their targets, including links pointing outside the source. Special filesystem objects (such as FIFOs and sockets) cause reset to fail rather than creating an incomplete copy. Filesystem permission, symlink, and Git behavior can vary by platform. A Git working tree whose root has no `.git` entry is treated as a plain directory.

On POSIX, a timed-out command starts in its own session: the environment signals its process group, escalates to SIGKILL and reaps the direct process with bounded waits. Descendants that create a new session/group can escape. On Windows, timeout cleanup attempts `taskkill /T /F` and then kills the direct process; `taskkill` availability and child termination are not guaranteed. Output readers and cleanup use bounded waits; a process that evades termination or keeps pipes open may cause incomplete capture and an `error` rather than a full transcript. Git diff collection and reset listings are not subject to the command output limit.

**Copying and command safeguards are not a security sandbox.** Commands run with the current user's permissions and can follow symlinks at runtime, access absolute paths, environment variables, other accessible files, and external resources. Review commands before executing them; call `close()` or use a context manager to release temporary disk space. Gymnasium compatibility, task-specific rewards, and stronger isolation are not implemented.

## Example

Run from the project root with the package importable:

```bash
python3 -m examples.basic
```

The example copies the project directory into a temporary episode workspace and runs the current Python interpreter with `-m pytest -q` in that copy.

## Tests

```bash
python3 -m pytest
```
