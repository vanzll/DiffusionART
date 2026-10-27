"""Publish an owned local exception before NCCL process-group teardown can wait."""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .minimax_h3_checkpoint_io import CheckpointIO


@contextmanager
def native_failure_fence(*, io_factory=CheckpointIO):
    try:
        yield
    except BaseException as exc:
        root = os.environ.get("H3_ACCEPTANCE_ROOT")
        attempt = os.environ.get("H3_ATTEMPT_DIR")
        nonce = os.environ.get("H3_ATTEMPT_NONCE")
        if root and attempt and nonce and Path(attempt).parent == Path(root) / "attempts" and Path(attempt).name == nonce:
            # Never log arbitrary exception messages: SDK errors can contain credentials.
            record = dict(status="native_local_runtime_exception", exception_type=type(exc).__name__,
                          attempt=nonce, rank=int(os.environ.get("RANK", "0")),
                          pid=os.getpid(), source_commit=os.environ.get("H3_RUNTIME_COMMIT"),
                          unix=time.time())
            print("H3_RUNTIME_FAILURE " + json.dumps(record), file=sys.stderr, flush=True)
            client = None
            try:
                client = io_factory()
                client.request("runtime_failure", root=root, attempt=attempt, nonce=nonce, record=record)
            except Exception as reporting_error:
                print("H3_RUNTIME_FAILURE_REPORT_ERROR " + type(reporting_error).__name__,
                      file=sys.stderr, flush=True)
            finally:
                if client is not None:
                    client.close()
        raise
