"""Temporary, per-episode copies of a source tree."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union


class EpisodeWorkspace:
    def __init__(self, source: Union[str, Path]) -> None:
        if isinstance(source, str) and source == "":
            raise ValueError("Repository path must not be empty")
        path = Path(source).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FileNotFoundError(f"Repository path cannot be resolved: {path}") from exc
        if not resolved.is_dir():
            raise ValueError(f"Repository path must be a directory: {path}")
        self.source = resolved
        self.path: Optional[Path] = None
        self._temporary: Optional[tempfile.TemporaryDirectory[str]] = None

    @staticmethod
    def _copy_file(source: str, destination: str) -> str:
        if not stat.S_ISREG(os.lstat(source).st_mode):
            raise ValueError(f"Cannot copy special filesystem object: {source}")
        return shutil.copy2(source, destination)

    def reset(self) -> Path:
        self.close()
        temporary = tempfile.TemporaryDirectory(prefix="swe-env-")
        destination = Path(temporary.name) / "repo"
        try:
            # Git metadata at any level may reference external worktrees or
            # object stores, and nested repositories break the private
            # baseline below. The copy therefore never contains `.git` entries.
            shutil.copytree(
                self.source,
                destination,
                symlinks=True,
                ignore=lambda directory, names: {".git"} if ".git" in names else set(),
                copy_function=self._copy_file,
            )
            if (self.source / ".git").exists() or (self.source / ".git").is_symlink():
                self._initialize_git(destination)
        except BaseException:
            try:
                temporary.cleanup()
            except OSError:
                pass  # Preserve the original copy or Git initialization failure.
            raise
        self._temporary = temporary
        self.path = destination
        return destination

    @staticmethod
    def _initialize_git(path: Path) -> None:
        commands = [
            ["git", "init", "--quiet"],
            ["git", "add", "-f", "--all"],
            [
                "git", "-c", "user.name=SWEEnv", "-c", "user.email=swe-env@invalid",
                "commit", "--quiet", "--allow-empty", "-m", "Episode baseline",
            ],
        ]
        for command in commands:
            subprocess.run(command, cwd=path, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def close(self) -> None:
        temporary = self._temporary
        self._temporary = None
        self.path = None
        if temporary is not None:
            temporary.cleanup()
