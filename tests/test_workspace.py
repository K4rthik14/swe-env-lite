import glob
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from swe_env import SWEEnv



def command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _swe_temp_dirs() -> set[str]:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "swe-env-*")))


def test_source_preserved_and_reset_recreates_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=t@invalid", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=source, check=True)
    (source / "tracked.txt").write_text("tracked\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=t@invalid", "commit", "--quiet", "-m", "tracked"], cwd=source, check=True)
    (source / "tracked.txt").write_text("uncommitted\n")
    (source / "untracked.txt").write_text("untracked\n")
    (source / "nested").mkdir()
    (source / "nested" / "data.txt").write_text("nested\n")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=source, capture_output=True, text=True, check=True).stdout

    env = SWEEnv(source)
    observation, info = env.reset()
    first = Path(info["workspace_path"])
    assert observation.exit_code == 0
    assert (first / "tracked.txt").read_text() == "uncommitted\n"
    assert (first / "untracked.txt").read_text() == "untracked\n"
    assert (first / "nested" / "data.txt").read_text() == "nested\n"
    assert env.step(command("from pathlib import Path; Path('tracked.txt').write_text('changed\\n'); Path('untracked.txt').unlink(); Path('new.txt').write_text('new')")).observation.exit_code == 0
    assert (first / "tracked.txt").read_text() == "changed\n"
    assert (first / "new.txt").exists()
    assert (source / "tracked.txt").read_text() == "uncommitted\n"
    assert (source / "untracked.txt").read_text() == "untracked\n"
    assert (source / "nested" / "data.txt").read_text() == "nested\n"
    assert subprocess.run(["git", "status", "--porcelain"], cwd=source, capture_output=True, text=True, check=True).stdout == status

    _, info = env.reset()
    second = Path(info["workspace_path"])
    assert second != first and not first.exists()
    assert (second / "tracked.txt").read_text() == "uncommitted\n"
    assert (second / "untracked.txt").read_text() == "untracked\n"
    assert not (second / "new.txt").exists()
    env.close()
    env.close()
    assert not second.exists()
    assert source.exists()
    with pytest.raises(RuntimeError, match="reset"):
        env.step("echo nope")


def test_failed_copy_cleans_up_and_never_steps_in_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    env = SWEEnv(source)
    env.reset()
    previous = env._workspace.path
    with patch("swe_env.workspace.shutil.copytree", side_effect=OSError("copy failed")):
        with pytest.raises(OSError, match="copy failed"):
            env.reset()
    assert previous is not None and not previous.exists()
    assert env._workspace.path is None
    with pytest.raises(RuntimeError, match="reset"):
        env.step(command("open('source_modified', 'w').close()"))
    assert not (source / "source_modified").exists()
    env.close()


def test_invalid_paths_and_special_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SWEEnv(tmp_path / "missing")
    file = tmp_path / "file"
    file.write_text("data")
    with pytest.raises(ValueError, match="directory"):
        SWEEnv(file)
    if hasattr(os, "mkfifo"):
        source = tmp_path / "source"
        source.mkdir()
        os.mkfifo(source / "pipe")
        env = SWEEnv(source)
        with pytest.raises((ValueError, shutil.Error), match="special filesystem object"):
            env.reset()
        assert env._workspace.path is None


def test_symlink_is_copied_without_following(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data").write_text("outside")
    try:
        (source / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")
    with SWEEnv(source) as env:
        _, info = env.reset()
        link = Path(info["workspace_path"]) / "link"
        assert link.is_symlink()
        assert os.readlink(link) == str(outside)


def test_empty_repo_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        SWEEnv("")


def test_partial_copy_failure_leaves_no_temp_dirs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "ok.txt").write_text("ok")
    (source / "sub" / "good.txt").write_text("good")
    if not hasattr(os, "mkfifo"):
        pytest.skip("requires fifos to force a mid-copy failure")
    os.mkfifo(source / "sub" / "pipe")
    env = SWEEnv(source)
    before = _swe_temp_dirs()
    with pytest.raises(ValueError, match="special filesystem object"):
        env.reset()
    assert env._workspace.path is None
    assert env._workspace._temporary is None
    assert _swe_temp_dirs() == before
    with pytest.raises(RuntimeError, match="reset"):
        env.step("echo unsafe")
    assert (source / "ok.txt").read_text() == "ok"
    assert (source / "sub" / "good.txt").read_text() == "good"

    (source / "sub" / "pipe").unlink()
    _, info = env.reset()
    workspace = Path(info["workspace_path"])
    assert (workspace / "ok.txt").read_text() == "ok"
    assert (workspace / "sub" / "good.txt").read_text() == "good"
    env.close()


def test_git_init_failure_cleans_up_and_recovers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=t@invalid", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=source, check=True)
    env = SWEEnv(source)
    before = _swe_temp_dirs()
    _, info = env.reset()
    first = Path(info["workspace_path"])
    assert first.is_dir()
    with patch("swe_env.workspace.subprocess.run", side_effect=FileNotFoundError("git missing")):
        with pytest.raises(FileNotFoundError, match="git missing"):
            env.reset()
    assert not first.exists()
    assert env._workspace.path is None
    assert env._workspace._temporary is None
    assert _swe_temp_dirs() == before
    with pytest.raises(RuntimeError, match="reset"):
        env.step("echo unsafe")

    _, info = env.reset()
    assert Path(info["workspace_path"]).is_dir()
    assert env.step("echo ok").observation.stdout.strip() == "ok"
    env.close()


def test_nested_git_metadata_is_excluded_from_baseline(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=t@invalid", "commit", "--quiet", "--allow-empty", "-m", "initial"], cwd=source, check=True)
    (source / "top.txt").write_text("top\n")
    nested = source / "sub"
    nested.mkdir()
    # An unborn nested repository used to abort the private baseline with
    # `git add` exit status 128 ("does not have a commit checked out").
    subprocess.run(["git", "init", "--quiet"], cwd=nested, check=True)
    (nested / "inner.txt").write_text("inner\n")
    (source / ".gitignore").write_text("*.log\n")
    (source / "debug.log").write_text("logs\n")
    env = SWEEnv(source)

    observation, info = env.reset()
    workspace = Path(info["workspace_path"])

    assert info["git_available"] is True
    assert observation.diff == ""
    # Root metadata is the fresh private baseline, not the source's history.
    assert (workspace / ".git" / "HEAD").exists()
    assert "initial" not in subprocess.run(
        ["git", "log", "--format=%s"], cwd=workspace, capture_output=True, text=True, check=True
    ).stdout
    # Nested metadata is excluded at every level; nested working files join
    # the baseline as ordinary tracked files instead.
    assert not (workspace / "sub" / ".git").exists()
    assert (workspace / "sub" / "inner.txt").read_text() == "inner\n"
    assert (workspace / "debug.log").read_text() == "logs\n"
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=workspace, capture_output=True, text=True, check=True
    ).stdout.split()
    assert "sub/inner.txt" in tracked
    result = env.step("echo mod >> top.txt")
    assert "top.txt" in result.observation.diff
    env.close()
