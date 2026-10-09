from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from swe_env import SWEEnv
from swe_env.types import StepResult, SWEAction, SWEObservation


def _python(command: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(command)}"


def test_reset_returns_workspace_listing_and_info(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    env = SWEEnv(tmp_path)

    observation, info = env.reset()

    assert isinstance(observation, SWEObservation)
    assert "hello.txt" in observation.stdout
    assert observation.exit_code == 0
    assert observation.diff == ""
    assert info["repo_path"] == str(tmp_path.resolve())
    assert info["git_available"] is False
    assert info["task_solved"] is None


def test_step_runs_in_repo_and_returns_step_result(tmp_path: Path) -> None:
    env = SWEEnv(tmp_path)
    env.reset()
    command = _python("import os; print(os.getcwd())")

    result = env.step(SWEAction(command))

    assert isinstance(result, StepResult)
    assert result.observation.stdout.strip() == str(env._workspace.path)
    assert env._workspace.path != tmp_path.resolve()
    assert result.observation.exit_code == 0
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["task_solved"] is None
    assert result.info["reward_basis"].startswith("command exit status only")


def test_nonzero_exit_captures_output_and_has_failure_reward(tmp_path: Path) -> None:
    env = SWEEnv(tmp_path)

    env.reset()
    result = env.step(_python("import sys; print('failed', file=sys.stderr); sys.exit(3)"))

    assert result.observation.stderr.strip() == "failed"
    assert result.observation.exit_code == 3
    assert result.reward == -1.0


def test_timeout_is_reported_as_command_failure(tmp_path: Path) -> None:
    env = SWEEnv(tmp_path, command_timeout=0.05)

    env.reset()
    result = env.step(_python("import time; time.sleep(2)"))

    assert result.observation.exit_code == 124
    assert result.info["timed_out"] is True
    assert "timed out" in result.info["error"]
    assert result.reward == -1.0


def test_git_diff_reports_tracked_changes_without_resetting_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "SWEEnv Test"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "sweenv-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "tracked.txt"
    source.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=tmp_path, check=True)
    env = SWEEnv(tmp_path)

    env.reset()
    result = env.step(
        _python("from pathlib import Path; Path('tracked.txt').write_text('after\\n')")
    )

    assert result.info["git_available"] is True
    assert "tracked.txt" in result.observation.diff
    assert "+after" in result.observation.diff
    assert source.read_text(encoding="utf-8") == "before\n"


def test_episode_limit_truncates_until_reset(tmp_path: Path) -> None:
    env = SWEEnv(tmp_path, max_steps=1)

    env.reset()
    result = env.step("true")

    assert result.truncated is True
    with pytest.raises(RuntimeError, match="reset"):
        env.step("true")
    env.reset()
    assert env.step("true").truncated is True


def test_invalid_path_and_action_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SWEEnv(tmp_path / "missing")
    env = SWEEnv(tmp_path)
    env.reset()
    with pytest.raises(ValueError, match="empty"):
        env.step("  ")
