"""A small command-execution environment for an existing software repository."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Union

from .executor import execute
from .types import StepResult, SWEAction, SWEObservation
from .workspace import EpisodeWorkspace


class SWEEnv:
    """Run commands in a disposable copy of a repository, not a security sandbox."""

    def __init__(
        self,
        repo_path: Union[str, Path],
        command_timeout: float = 30.0,
        max_steps: int = 20,
        output_limit_bytes: int = 1_000_000,
    ) -> None:
        workspace = EpisodeWorkspace(repo_path)
        if (isinstance(command_timeout, bool) or not isinstance(command_timeout, (int, float))
                or not math.isfinite(command_timeout) or command_timeout <= 0):
            raise ValueError("command_timeout must be a finite positive number")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if (isinstance(output_limit_bytes, bool) or not isinstance(output_limit_bytes, int)
                or output_limit_bytes < 1):
            raise ValueError("output_limit_bytes must be a positive integer")

        self.repo_path = workspace.source
        self._workspace = workspace
        self.command_timeout = command_timeout
        self.max_steps = max_steps
        self.output_limit_bytes = output_limit_bytes
        self._steps = 0
        self._truncated = False

    def reset(self) -> tuple[SWEObservation, dict]:
        """Start a new episode and return a top-level workspace listing."""
        path = self._workspace.reset()
        self._steps = 0
        self._truncated = False
        try:
            names = sorted(entry.name for entry in os.scandir(path))
            stdout = "\n".join(names)
            stderr = ""
            exit_code = 0
        except OSError as exc:
            stdout = ""
            stderr = str(exc)
            exit_code = 1

        git_available, diff = self._git_changes()
        observation = SWEObservation(stdout, stderr, exit_code, diff)
        info = {
            "repo_path": str(self.repo_path),
            "workspace_path": str(path),
            "step": 0,
            "git_available": git_available,
            "diff_scope": "tracked staged and unstaged changes; untracked files are omitted",
            "task_solved": None,
        }
        return observation, info

    def step(self, action: Union[SWEAction, str]) -> StepResult:
        """Execute one command and return the documented five-field result.

        Reward is 0 for an exit status of 0 and -1 otherwise. This only scores
        command execution; it is not a task-completion signal.
        """
        if self._workspace.path is None:
            raise RuntimeError("No active episode; call reset() before stepping")
        if self._truncated:
            raise RuntimeError("Episode step limit reached; call reset() before stepping again")
        if isinstance(action, SWEAction):
            command = action.command
        elif isinstance(action, str):
            command = action
        else:
            raise TypeError("action must be a command string or SWEAction")
        if not isinstance(command, str):
            raise TypeError("command must be a string")
        if not command.strip():
            raise ValueError("command must not be empty")

        self._steps += 1
        execution = execute(command, self._workspace.path, self.command_timeout, self.output_limit_bytes)
        git_available, diff = self._git_changes()
        observation = SWEObservation(execution.stdout, execution.stderr, execution.exit_code, diff)
        terminated = False
        self._truncated = self._steps >= self.max_steps
        reward = 0.0 if execution.exit_code == 0 else -1.0
        info = {
            "command": command,
            "step": self._steps,
            "max_steps": self.max_steps,
            "timed_out": execution.timed_out,
            "error": execution.error,
            "stdout_truncated": execution.stdout_truncated,
            "stderr_truncated": execution.stderr_truncated,
            "output_limit_bytes": self.output_limit_bytes,
            "git_available": git_available,
            "diff_scope": "tracked staged and unstaged changes; untracked files are omitted",
            "task_solved": None,
            "reward_basis": "command exit status only; no task-completion evaluator is configured",
        }
        return StepResult(observation, reward, terminated, self._truncated, info)

    def close(self) -> None:
        """Release the active episode; reset() is required before another step."""
        self._workspace.close()

    def __enter__(self) -> SWEEnv:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except OSError:
                pass  # Do not hide an exception raised inside the episode.

    def _git_changes(self) -> tuple[bool, str]:
        probe = self._run_git(["git", "rev-parse", "--is-inside-work-tree"])
        if probe is None or probe.returncode != 0:
            return False, ""

        diff = self._run_git(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"]
        )
        if diff is not None and diff.returncode == 0:
            return True, diff.stdout or ""

        # An unborn repository has no HEAD; report its staged and unstaged tracked diffs.
        unstaged = self._run_git(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "--"]
        )
        staged = self._run_git(
            ["git", "diff", "--cached", "--no-ext-diff", "--no-textconv", "--binary"]
        )
        parts = []
        if staged is not None and staged.returncode == 0 and staged.stdout:
            parts.append(staged.stdout)
        if unstaged is not None and unstaged.returncode == 0 and unstaged.stdout:
            parts.append(unstaged.stdout)
        return True, "\n".join(parts)

    def _run_git(self, command: list[str]) -> Union[subprocess.CompletedProcess[str], None]:
        try:
            return subprocess.run(
                command,
                cwd=self._workspace.path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(self.command_timeout, 5.0),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
