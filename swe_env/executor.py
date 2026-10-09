"""Bounded capture and best-effort process-tree cleanup for shell actions."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Execution:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 127
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error: Optional[str] = None


class _Capture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self.error: Optional[OSError] = None

    def read(self, pipe: object) -> None:
        try:
            while True:
                chunk = os.read(pipe.fileno(), 8192)  # type: ignore[attr-defined]
                if not chunk:
                    break
                remaining = self.limit - len(self.data)
                self.data.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except OSError as exc:
            self.error = exc

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def _terminate(process: subprocess.Popen[bytes]) -> Optional[str]:
    """Signal the shell's group, then escalate even if the shell exits first."""
    errors = []
    if os.name == "posix":
        group = process.pid  # start_new_session=True makes this the process group ID.
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(str(exc))
        try:
            process.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
        # Children may still be alive after their shell has exited.
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(str(exc))
    else:
        # Windows has no stdlib equivalent of POSIX killpg. taskkill can
        # terminate a process tree, but may be unavailable or denied.
        try:
            cleanup = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=1, check=False,
            )
            if cleanup.returncode != 0:
                errors.append(f"taskkill exited with status {cleanup.returncode}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(str(exc))
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        errors.append("Process did not exit after forced termination")
    return "; ".join(errors) or None


def execute(command: str, cwd: Path, timeout: float, output_limit: int) -> Execution:
    result = Execution()
    try:
        process = subprocess.Popen(
            command, cwd=cwd, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        result.error = f"Process launch failed: {exc}"
        result.stderr = result.error
        return result

    stdout = _Capture(output_limit)
    stderr = _Capture(output_limit)
    readers = [
        threading.Thread(target=stdout.read, args=(process.stdout,), daemon=True),
        threading.Thread(target=stderr.read, args=(process.stderr,), daemon=True),
    ]
    try:
        for reader in readers:
            reader.start()
        try:
            result.exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.exit_code = 124
            result.error = f"Command timed out after {timeout:g} seconds"
            cleanup_error = _terminate(process)
            if cleanup_error:
                result.error += f"; cleanup failed: {cleanup_error}"
    finally:
        if process.poll() is None:
            cleanup_error = _terminate(process)
            result.error = f"{result.error or 'Execution failed'}; cleanup failed: {cleanup_error}" if cleanup_error else result.error
        for reader in readers:
            reader.join(timeout=1)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        for reader in readers:
            reader.join(timeout=0.1)
            if reader.is_alive():
                result.error = f"{result.error or 'Capture failed'}; output reader did not exit"
    result.stdout = stdout.text()
    result.stderr = stderr.text()
    result.stdout_truncated = stdout.truncated
    result.stderr_truncated = stderr.truncated
    for capture in (stdout, stderr):
        if capture.error is not None:
            result.error = f"{result.error or 'Capture failed'}; {capture.error}"
    return result
