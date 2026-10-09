"""A small command-execution environment for an existing software repository."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Union

from .types import StepResult, SWEAction, SWEObservation


class SWEEnv:
    """Run explicitly supplied shell commands with a repository as their cwd.

    Commands run with the current user's permissions. This class does not
    sandbox commands, reset the repository, or decide whether a task is solved.
    """

    def __init__(
        self,
        repo_path: Union[str, Path],
        command_timeout: float = 30.0,
        max_steps: int = 20,
    ) -> None:
        path = Path(repo_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Repository path must be a directory: {path}")
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.repo_path = path.resolve()
        self.command_timeout = command_timeout
        self.max_steps = max_steps
        self._steps = 0
        self._truncated = False

    def reset(self) -> tuple[SWEObservation, dict]:
        """Start a new episode and return a top-level workspace listing."""
        self._steps = 0
        self._truncated = False
        try:
            names = sorted(entry.name for entry in os.scandir(self.repo_path))
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
        if self._truncated:
            raise RuntimeError("Episode step limit reached; call reset() before stepping again")
        if isinstance(action, SWEAction):
            command = action.command
        elif isinstance(action, str):
            command = action
        else:
            raise TypeError("action must be a command string or SWEAction")
        if not command.strip():
            raise ValueError("command must not be empty")

        self._steps += 1
        timed_out = False
        error = None
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                check=False,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = self._decode_output(exc.stdout)
            stderr = self._decode_output(exc.stderr)
            exit_code = 124
            error = f"Command timed out after {self.command_timeout:g} seconds"
        except OSError as exc:
            stdout = ""
            stderr = str(exc)
            exit_code = 127
            error = str(exc)

        git_available, diff = self._git_changes()
        observation = SWEObservation(stdout, stderr, exit_code, diff)
        terminated = False
        self._truncated = self._steps >= self.max_steps
        reward = 0.0 if exit_code == 0 else -1.0
        info = {
            "command": command,
            "step": self._steps,
            "max_steps": self.max_steps,
            "timed_out": timed_out,
            "error": error,
            "git_available": git_available,
            "diff_scope": "tracked staged and unstaged changes; untracked files are omitted",
            "task_solved": None,
            "reward_basis": "command exit status only; no task-completion evaluator is configured",
        }
        return StepResult(observation, reward, terminated, self._truncated, info)

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
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=min(self.command_timeout, 5.0),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _decode_output(output: Union[str, bytes, None]) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode(errors="replace")
        return output
