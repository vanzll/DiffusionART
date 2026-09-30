"""One configuration-based entrypoint; no implicit model download or fallback."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from .config import ConfigError, load_config
from .registry import get_backend, list_backends


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--validate", action="store_true", help="check config and installed backend; no training"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="print resolved config; no training"
    )
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--output-dir", help="override only the artifact output directory")
    parser.add_argument(
        "--resume", help="explicit checkpoint receipt; never auto-resume by run name"
    )
    args = parser.parse_args(argv)
    if args.list_backends:
        for backend in list_backends():
            print(f"{backend.name}\t{backend.status}\t{','.join(sorted(backend.tasks))}")
        return 0
    if not args.config:
        parser.error("--config is required")
    try:
        overrides = {"runtime": {"output_dir": args.output_dir}} if args.output_dir else {}
        config = load_config(args.config, overrides)
        if args.print_config:
            print(json.dumps(config.to_dict(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        # Reject accidental independent single-device jobs under torchrun before
        # any backend creates an output directory, loads assets or touches GPUs.
        if config.runtime.distributed == "none":
            from .training.distributed import launch_identity

            launch_identity("none")
        backend = get_backend(config)
        if config.runtime.log == "wandb" and importlib.util.find_spec("wandb") is None:
            raise ConfigError("W&B logging requires installing visual-diffusion-rl[wandb]")
        if args.validate:
            print(
                json.dumps(
                    {
                        "valid": True,
                        "name": config.name,
                        "config_sha256": config.fingerprint(),
                        "backend_status": backend.status,
                    }
                )
            )
            return 0
        backend.load()(config, resume=args.resume)
        return 0
    except (ConfigError, FileNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(
            f"output already exists: {exc.filename}; choose a new --output-dir or explicit --resume",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
