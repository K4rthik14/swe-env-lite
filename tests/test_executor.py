"""Behavioral tests for bounded shell execution in an episode workspace."""

import os
import shlex
import signal
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from swe_env import SWEEnv
from swe_env.executor import _Capture
from swe_env.types import SWEAction


def py(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


class _FdPipe:
    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


class _ClosedPipe:
    def fileno(self) -> int:
        raise ValueError("I/O operation on closed file.")


def test_capture_records_closed_pipe_without_raising() -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"partial")
    finally:
        os.close(write_fd)
    capture = _Capture(1024)
    capture.read(_FdPipe(read_fd))
    os.close(read_fd)
    assert capture.text() == "partial"
    assert capture.error is None

    # execute() closes pipes after a bounded join; a live reader observes this.
    capture.read(_ClosedPipe())
    assert capture.text() == "partial"
    assert isinstance(capture.error, ValueError)


def test_capture_closed_pipe_in_thread_reports_no_thread_exception() -> None:
    capture = _Capture(1024)
    seen: list = []
    previous = threading.excepthook
    threading.excepthook = lambda args: seen.append(args.exc_value)  # noqa: E731
    try:
        worker = threading.Thread(target=capture.read, args=(_ClosedPipe(),), daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
    finally:
        threading.excepthook = previous
    assert seen == []
    assert isinstance(capture.error, ValueError)


def test_bounded_concurrent_output_and_unicode(tmp_path: Path) -> None:
    with SWEEnv(tmp_path, output_limit_bytes=7) as env:
        env.reset()
        result = env.step(py("import os; os.write(1, 'éééé'.encode()); os.write(2, 'λ' .encode() * 4)"))
        assert result.observation.stdout == "ééé�"
        assert result.observation.stderr == "λλλ�"
        assert result.info["stdout_truncated"] is True
        assert result.info["stderr_truncated"] is True
        assert result.info["output_limit_bytes"] == 7
        assert result.observation.exit_code == 0
        within = env.step(py("import sys; sys.stdout.write('ok'); sys.stderr.write('err')"))
        assert (within.observation.stdout, within.observation.stderr) == ("ok", "err")
        assert within.info["stdout_truncated"] is False
        assert within.info["stderr_truncated"] is False


def test_large_streams_and_nonzero_exit(tmp_path: Path) -> None:
    with SWEEnv(tmp_path, output_limit_bytes=1024) as env:
        env.reset()
        result = env.step(py("import os,sys,threading; a=lambda fd: [os.write(fd,b'x'*8192) for _ in range(256)]; t=threading.Thread(target=a,args=(2,)); t.start(); a(1); t.join(); sys.exit(3)"))
        assert result.observation.stdout == "x" * 1024
        assert result.observation.stderr == "x" * 1024
        assert result.info["stdout_truncated"] and result.info["stderr_truncated"]
        assert result.observation.exit_code == 3
        assert result.reward == -1.0


def test_timeout_preserves_output_and_next_step(tmp_path: Path) -> None:
    with SWEEnv(tmp_path, command_timeout=0.3, output_limit_bytes=4, max_steps=2) as env:
        env.reset()
        result = env.step(py("import os,time; os.write(1,b'123456'); os.write(2,b'error'); time.sleep(30)"))
        assert result.info["timed_out"] is True
        assert result.observation.exit_code == 124
        assert result.observation.stdout == "1234"
        assert result.observation.stderr == "erro"
        assert result.info["stdout_truncated"] and result.info["stderr_truncated"]
        assert result.reward == -1.0
        next_result = env.step(py("print('ok')"))
        assert next_result.observation.stdout.strip() == "ok"
        assert next_result.info["timed_out"] is False
        assert next_result.info["step"] == 2
        assert next_result.truncated is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups and /proc test")
def test_timeout_kills_shell_descendant(tmp_path: Path) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("requires Linux /proc to distinguish dead from zombie child")

    child = "import os,time; open('child.pid','w').write(str(os.getpid())); time.sleep(30)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    with SWEEnv(tmp_path, command_timeout=0.8) as env:
        env.reset()
        result = env.step(py(parent))
        workspace = Path(env.repo_path)  # The source must not receive the child PID file.
        assert not (workspace / "child.pid").exists()
        pidfile = Path(env._workspace.path) / "child.pid"
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        assert result.info["timed_out"] is True
        for _ in range(30):
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("child still running after timeout")


def test_timeout_captures_output_from_inherited_descendant(tmp_path: Path) -> None:
    child = "import sys,time; sys.stdout.write('child-marker\\n'); sys.stdout.flush(); time.sleep(30)"
    parent = f"import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',{child!r}]); p.wait()"
    with SWEEnv(tmp_path, command_timeout=1.0) as env:
        env.reset()
        result = env.step(py(parent))
    assert result.info["timed_out"] is True
    assert result.observation.exit_code == 124
    assert "child-marker" in result.observation.stdout
    assert result.info["error"] == "Command timed out after 1 seconds"


@pytest.mark.skipif(os.name != "posix", reason="POSIX sessions and /proc test")
def test_escaped_descendant_stays_bounded_and_reports_reader_pressure(tmp_path: Path) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("requires Linux /proc to reap the escaped child")
    child = "import os,sys,time; os.setsid(); open('esc.pid','w').write(str(os.getpid())); sys.stdout.write('early\\n'); sys.stdout.flush(); time.sleep(30)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    thread_errors: list = []
    lock = threading.Lock()
    previous = threading.excepthook

    def hook(args) -> None:
        with lock:
            thread_errors.append(args.exc_value)

    threading.excepthook = hook
    pid = None
    try:
        with SWEEnv(tmp_path, command_timeout=0.5) as env:
            env.reset()
            started = time.monotonic()
            result = env.step(py(parent))
            elapsed = time.monotonic() - started
            pidfile = Path(env._workspace.path) / "esc.pid"
            assert pidfile.exists()
            pid = int(pidfile.read_text())
        assert result.info["timed_out"] is True
        assert result.observation.exit_code == 124
        assert elapsed < 15
        assert "early" in result.observation.stdout
        assert result.info["error"] is not None
        assert result.info["error"].count("output reader did not exit") == 1
    finally:
        threading.excepthook = previous
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            for _ in range(200):
                try:
                    if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
                        break
                except FileNotFoundError:
                    break
                time.sleep(0.05)
            time.sleep(0.5)
    with lock:
        assert thread_errors == []


def test_repeated_timeouts_do_not_accumulate_threads(tmp_path: Path) -> None:
    with SWEEnv(tmp_path, command_timeout=0.2, max_steps=10) as env:
        env.reset()
        baseline = threading.active_count()
        for _ in range(3):
            result = env.step(py("import time; time.sleep(30)"))
            assert result.info["timed_out"] is True
            assert result.observation.exit_code == 124
        for _ in range(100):
            if threading.active_count() <= baseline:
                break
            time.sleep(0.05)
        assert threading.active_count() <= baseline
        assert env.step(py("print('ok')")).observation.stdout.strip() == "ok"


def test_cleanup_failure_disables_episode(tmp_path: Path) -> None:
    env = SWEEnv(tmp_path)
    _, info = env.reset()
    temporary = env._workspace._temporary
    assert temporary is not None
    with patch.object(temporary, "cleanup", side_effect=OSError("cleanup blocked")):
        with pytest.raises(OSError, match="cleanup blocked"):
            env.close()
    assert tmp_path.exists()
    with pytest.raises(RuntimeError, match="reset"):
        env.step("echo unsafe")
    temporary.cleanup()
    assert not Path(info["workspace_path"]).exists()


def test_launch_failure_missing_command_and_accounting(tmp_path: Path) -> None:
    with SWEEnv(tmp_path, max_steps=3) as env:
        env.reset()
        with pytest.raises(TypeError, match="string"):
            env.step(SWEAction(None))
        with pytest.raises(TypeError, match="action"):
            env.step(42)
        with pytest.raises(ValueError, match="empty"):
            env.step("  ")
        missing = env.step("swe_env_executable_that_does_not_exist_8592")
        assert missing.observation.exit_code != 0
        assert missing.reward == -1.0
        assert missing.info["step"] == 1
        with patch("swe_env.executor.subprocess.Popen", side_effect=OSError("cannot launch")):
            failed = env.step("echo test")
        assert failed.observation.exit_code == 127
        assert failed.info["error"] == "Process launch failed: cannot launch"
        assert failed.info["timed_out"] is False
        assert failed.info["step"] == 2
        assert env.step("echo ok").info["step"] == 3


@pytest.mark.parametrize("argument,value", [
    ("output_limit_bytes", 0), ("output_limit_bytes", 1.5),
    ("output_limit_bytes", True), ("command_timeout", float("inf")),
    ("command_timeout", float("nan")), ("command_timeout", "invalid"),
    ("max_steps", 1.5),
])
def test_invalid_configuration(tmp_path: Path, argument: str, value: object) -> None:
    with pytest.raises(ValueError):
        SWEEnv(tmp_path, **{argument: value})
