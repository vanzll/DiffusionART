"""Bounded single-request JSON transport for isolated reward interpreters."""

import json
import os
import selectors
import signal
import subprocess
import threading
import time
import weakref


def _cleanup(process, response_fd):
    if process.stdin is not None:
        process.stdin.close()
    os.close(response_fd)
    # The worker owns a new session; include DataLoader children even when the
    # direct interpreter has already exited. Never target the trainer's group.
    def signal_group(sig):
        retry_deadline = time.monotonic() + 0.2
        while True:
            try:
                os.killpg(process.pid, sig)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # Darwin can report EPERM during group exit before waitpid
                # observes the leader's exit. Reap/retry briefly, not indefinitely.
                process.poll()
                if time.monotonic() >= retry_deadline:
                    raise
                time.sleep(0.01)

    process.poll()
    signal_group(signal.SIGTERM)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        process.poll()
        if not signal_group(0):
            break
        time.sleep(0.01)
    if signal_group(0):
        signal_group(signal.SIGKILL)
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


class JsonWorker:
    """One owner and private response FD; SDK stdout cannot corrupt replies.

    The command receives the response FD followed by explicit arguments. Engine
    wrappers validate their own payloads and close the worker on unknown state.
    No automatic restart/replay occurs after a timeout or transport failure.
    """

    def __init__(self, command, arguments, *, startup_timeout, request_timeout,
                 error_type=RuntimeError, label="Reward", response_limit=2 * 1024 * 1024):
        if any(
            not isinstance(t, (int, float)) or isinstance(t, bool) or not 0 < t <= 600
            for t in (startup_timeout, request_timeout)
        ):
            raise ValueError(f"{label} deadlines must be positive and at most 600 seconds")
        self.timeout, self.error_type, self.label = request_timeout, error_type, label
        self.response_limit = response_limit
        self._lock = threading.Lock()
        read_fd, write_fd = os.pipe()
        try:
            process = subprocess.Popen(
                [*command, str(write_fd), *arguments], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,), bufsize=0, start_new_session=True,
            )
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        self.process, self.response_fd = process, read_fd
        self._finalizer = weakref.finalize(self, _cleanup, process, read_fd)
        try:
            os.set_blocking(read_fd, False)
            os.set_blocking(process.stdin.fileno(), False)
            reply = self._exchange(b"", startup_timeout)
            if reply != {"status": "ready"}:
                raise error_type(
                    f"{label} worker initialization failed: {reply.get('error_type', 'protocol')}"
                    + (f": {reply['message']}" if isinstance(reply.get("message"), str) else "")
                )
        except BaseException:
            self.close()
            raise

    def _exchange(self, payload, timeout):
        if not self._finalizer.alive or self.process.poll() is not None:
            raise self.error_type(f"{self.label} worker is not alive")
        deadline, offset, response = time.monotonic() + timeout, 0, bytearray()
        with selectors.DefaultSelector() as poll:
            poll.register(self.response_fd, selectors.EVENT_READ)
            if payload:
                poll.register(self.process.stdin.fileno(), selectors.EVENT_WRITE)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self.error_type(f"{self.label} worker deadline exceeded")
                for key, _ in poll.select(remaining):
                    if key.fd == self.response_fd:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            raise self.error_type(f"{self.label} worker response pipe closed")
                        response.extend(chunk)
                        if len(response) > self.response_limit:
                            raise self.error_type(f"{self.label} worker response exceeds byte limit")
                        if b"\n" in response:
                            if (offset != len(payload) or not response.endswith(b"\n")
                                    or response.count(b"\n") != 1):
                                raise self.error_type(f"{self.label} worker response framing differs")
                            result = json.loads(response)
                            if not isinstance(result, dict):
                                raise self.error_type(f"invalid {self.label} worker response")
                            return result
                    else:
                        offset += os.write(key.fd, memoryview(payload)[offset:offset + 65536])
                        if offset == len(payload):
                            poll.unregister(key.fd)

    def close(self):
        self._finalizer()
