"""H3-only execution markers and finite, removable batch-shape observation."""

import json
import os
import time
from collections import Counter

import torch


def event(stage, **values):
    record = {"schema_version": 1, "rank": int(os.environ.get("RANK", "0")),
              "attempt": os.environ.get("H3_ATTEMPT_NONCE"),
              "commit": os.environ.get("H3_RUNTIME_COMMIT"),
              "stage": stage, "unix": time.time(), **values}
    if torch.cuda.is_initialized():
        record.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print("H3_PROGRESS " + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


class BatchTrace:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.handles = []
        self.counts = Counter()
        self.seen = set()

    def install(self, model):
        if not self.enabled:
            return
        modules = {"transformer": model}
        modules.update({name: getattr(model, name) for name in
                        ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out")
                        if hasattr(model, name)})
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._hook(name), with_kwargs=True))

    def _hook(self, name):
        def observe(module, arguments, keywords):
            inputs = {f"arg_{index}": value for index, value in enumerate(arguments)}
            inputs.update(keywords)
            shapes, dtypes = {}, {}
            def inspect(key, value):
                if isinstance(value, torch.Tensor):
                    shapes[key] = list(value.shape)
                    dtypes[key] = str(value.dtype)
                elif isinstance(value, dict):
                    for child, nested in value.items():
                        inspect(f"{key}/{child}", nested)
            for key, value in inputs.items():
                inspect(key, value)
            mode = "update" if torch.is_grad_enabled() else "inference"
            signature = json.dumps(shapes, sort_keys=True)
            self.counts[(name, mode, signature)] += 1
            identity = name, mode, signature
            if identity not in self.seen:
                self.seen.add(identity)
                event("physical_batch_shape", module=name, mode=mode, shapes=shapes,
                      dtypes=dtypes, cuda_autocast=torch.is_autocast_enabled())
        return observe

    def close(self):
        if not self.handles:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for (name, mode, shapes), count in sorted(self.counts.items()):
            event("physical_batch_call_count", module=name, mode=mode,
                  shapes=json.loads(shapes), count=count)
        event("physical_batch_trace_complete", hooks_removed=True,
              observed_groups=len(self.counts), observed_module_calls=sum(self.counts.values()))
