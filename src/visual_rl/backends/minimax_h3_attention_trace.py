"""Opt-in one-call CPU evidence of SDPA's post-autocast types and backend.

No CUDA timing, extra attention call, or device binding is performed here.
Install after native binding; CPU conditioning is ignored by default.
"""
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
import torch.nn.functional as functional

from .minimax_h3_observability import event


class OneShotAttentionTrace:
    def __init__(self, root, *, enabled=False, device_type="cuda"):
        self.root = Path(root)
        self.enabled = enabled
        self.device_type = device_type
        self.original = self.wrapper = None
        self.used = False

    @staticmethod
    def notice(stage, **values):
        try:
            event(stage, **values)
        except Exception:
            pass

    def install(self):
        if not self.enabled or self.wrapper is not None:
            return
        self.original = functional.scaled_dot_product_attention

        def observe(*args, **kwargs):
            query = args[0] if args else kwargs.get("query")
            if (self.used or not isinstance(query, torch.Tensor)
                    or query.device.type != self.device_type):
                return self.original(*args, **kwargs)
            self.used = True
            self.close()
            nonce = os.environ.get("H3_ATTEMPT_NONCE", "")
            if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", nonce)
                    or torch.autograd._profiler_enabled()):
                self.notice("attention_trace_skipped", reason="unsafe_nonce_or_existing_profiler")
                return self.original(*args, **kwargs)
            started = time.monotonic()
            try:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True,
                    with_stack=False, profile_memory=False)
                profiler.__enter__()
            except Exception as error:
                self.notice("attention_trace_unavailable", error_type=type(error).__name__)
                return self.original(*args, **kwargs)
            try:
                result = self.original(*args, **kwargs)
            except BaseException:
                try:
                    profiler.__exit__(*sys.exc_info())
                except Exception:
                    pass
                raise
            try:
                profiler.__exit__(None, None, None)
                rank = int(os.environ.get("RANK", "0"))
                directory = self.root / "attention-traces" / nonce
                directory.mkdir(parents=True, exist_ok=True)
                trace = directory / f"rank-{rank:05d}-first-sdpa.trace.json"
                profiler.export_chrome_trace(str(trace))
                document = json.loads(trace.read_text())
                operators = [dict(name=record["name"],
                    input_types=record.get("args", {}).get("Input type"),
                    input_shapes=record.get("args", {}).get("Input Dims"))
                    for record in document["traceEvents"]
                    if "scaled_dot_product" in record.get("name", "")]
                receipt = dict(schema_version=1, rank=rank, attempt=nonce,
                    status="one_native_sdpa_cpu_operator_trace", trace=str(trace),
                    attention_operators=operators, output_dtype=str(result.dtype),
                    device_type=self.device_type, cuda_activity_enabled=False,
                    operation_replayed=False, host_capture_seconds=time.monotonic()-started,
                    scope="Dispatched operator/input types; no CUDA timing or exact kernel attribution")
                path = directory / f"rank-{rank:05d}-first-sdpa.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(receipt, indent=2)+"\n")
                temporary.replace(path)
                self.notice("attention_trace_complete", receipt=str(path),
                            output_dtype=str(result.dtype), operation_replayed=False)
            except Exception as error:
                self.notice("attention_trace_export_failed", error_type=type(error).__name__)
            return result

        self.wrapper = observe
        functional.scaled_dot_product_attention = observe

    def close(self):
        if self.wrapper is not None and functional.scaled_dot_product_attention is self.wrapper:
            functional.scaled_dot_product_attention = self.original
