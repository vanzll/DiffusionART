"""Publish owned failure before process-group teardown, with bounded Ceph I/O."""

import json
import os
import runpy
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


def publish_owned_failure(sender, start_ticks, exception_type):
    root = Path(os.environ["H3_ACCEPTANCE_ROOT"]).resolve(strict=True)
    attempt = Path(os.environ["H3_ATTEMPT_DIR"]).resolve(strict=True)
    nonce = os.environ["H3_ATTEMPT_NONCE"]
    if attempt.parent != root / "attempts" or attempt.name != nonce:
        raise RuntimeError("Failure attempt identity changed")
    control = runpy.run_path(str(Path(os.environ["H3_CHECKOUT"]) / "scripts/minimax_h3_process_control.py"))
    identity = control["identity"]
    process = identity(sender)
    if (sender != os.getppid() or process is None or process["start_ticks"] != start_ticks):
        raise RuntimeError("Failure sender identity changed")
    node = str(int(os.environ["H3_NODE_RANK"]))
    owner = dict(line.split("=", 1) for line in (attempt / "nodes" / node / "owner.identity").read_text().splitlines())
    expected = {"GPU_LEASE_PID": owner["pid"], "GPU_LEASE_START_TICKS": owner["start_ticks"],
                "GPU_LEASE_GENERATION": owner["generation"], "GPU_LEASE_NONCE": nonce}
    lease = Path("/m2v_intern2/zhenglin/paper_coordination/GPU_LEASES") / owner["host"] / "owner.env"
    control["validate_lease"](lease, expected)
    control["supervisor_identities"](expected)
    current = process
    for _ in range(128):
        if current["pid"] == int(owner["pid"]):
            break
        current = identity(current["parent_pid"])
        if current is None:
            raise RuntimeError("Failure sender is outside the live owner tree")
    else:
        raise RuntimeError("Unbounded failure sender ancestry")
    control["validate_lease"](lease, expected)
    metadata = {"exception_type": exception_type, "pid": sender, "start_ticks": start_ticks,
                "nonce": nonce, "rank": os.environ.get("RANK")}
    try:
        with (attempt / f"failure.{node}").open("x") as stream:
            json.dump(metadata, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass


def report_failure(exception_type, timeout=2):
    if not os.environ.get("H3_ATTEMPT_NONCE"):
        return
    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    child = subprocess.Popen([sys.executable, __file__, str(os.getpid()),
                              fields[19], exception_type], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Do not wait on a child blocked in filesystem I/O; nonce owner drains it.
        child.kill()


@contextmanager
def native_failure_fence():
    try:
        yield
    except BaseException as error:
        name = type(error).__name__
        print(json.dumps({"stage": "h3_failure_before_teardown", "exception_type": name,
                          "rank": os.environ.get("RANK"), "pid": os.getpid()}), flush=True)
        try:
            report_failure(name)
        except BaseException as reporting_error:
            print(json.dumps({"stage": "h3_failure_reporting_failed",
                              "exception_type": type(reporting_error).__name__}), flush=True)
        raise


if __name__ == "__main__":
    publish_owned_failure(int(sys.argv[1]), sys.argv[2], sys.argv[3])
