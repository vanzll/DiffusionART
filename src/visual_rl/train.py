"""One config-based entrypoint with declared asset preparation and no scientific fallback."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

from .config import ConfigError, load_config
from .registry import get_backend, list_backends
from .training.no_signal import NoSignalRolloutLimit


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
    parser.add_argument("--nproc-per-node", type=int,
                        help="prepare assets, then launch this node with torchrun")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--prepare-only", action="store_true", help="prepare assets without starting training")
    parser.add_argument("--max-updates", type=int, help="override total optimizer-update budget")
    parser.add_argument(
        "--resume", help="latest, a run/checkpoint directory, or a complete checkpoint receipt"
    )
    parser.add_argument("--resume-rewind", action="store_true",
                        help="archive history after the checkpoint before resuming")
    parser.add_argument("--eval-only", action="store_true",
                        help="load a complete checkpoint and evaluate without training")
    args = parser.parse_args(argv)
    if args.nproc_per_node is not None and args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be positive")
    if (args.nnodes < 1 or not 0 <= args.node_rank < args.nnodes
            or not 1 <= args.master_port <= 65535):
        parser.error("invalid node count/rank or master port")
    if args.nnodes > 1 and (args.nproc_per_node is None or not args.master_addr):
        parser.error("multi-node launch requires --nproc-per-node and --master-addr")
    if (args.resume_rewind or args.eval_only) and not args.resume:
        parser.error("--resume-rewind and --eval-only require --resume")
    if args.resume_rewind and args.eval_only:
        parser.error("evaluation must not rewind training history")
    if args.list_backends:
        for backend in list_backends():
            print(f"{backend.name}\t{backend.status}\t{','.join(sorted(backend.tasks))}")
        return 0
    if not args.config:
        parser.error("--config is required")
    try:
        from .prepare_assets import plan_environment, prepare_assets, validate_bindings

        # Planning is network-free so validation/printing never downloads assets.
        plan = plan_environment(args.config)
        overrides = {"runtime": {"output_dir": args.output_dir}} if args.output_dir else {}
        if args.max_updates is not None:
            overrides["training"] = {"max_updates": args.max_updates}
        config = load_config(args.config, overrides)
        validate_bindings(config, plan)
        if args.print_config:
            print(json.dumps(config.to_dict(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        # Reject accidental independent single-device jobs under torchrun before
        # any backend creates an output directory, loads assets or touches GPUs.
        if config.runtime.distributed == "none":
            from .training.distributed import launch_identity

            launch_identity("none")
        backend = get_backend(config)
        if args.eval_only and not config.evaluation.enabled:
            raise ConfigError("--eval-only requires an enabled evaluation manifest")
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
        if config.runtime.log == "wandb" and not args.prepare_only:
            from .wandb_export import validate_credentials

            validate_credentials(mode=config.runtime.wandb.get("mode"))
        if args.nproc_per_node is not None:
            if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
                raise ConfigError("do not nest --nproc-per-node inside torchrun")
            if config.runtime.distributed == "none":
                raise ConfigError("--nproc-per-node requires a distributed config")
            workers = args.nproc_per_node * args.nnodes
            if (config.sampling.prompts_per_rollout < workers
                    or config.sampling.prompts_per_rollout % workers):
                raise ConfigError("global prompt count must divide evenly over the requested workers")
        elif int(os.environ.get("WORLD_SIZE", "1")) > 1 and any(plan):
            raise ConfigError("prepare assets before torchrun; use visual-rl --nproc-per-node N "
                              "or supply the configured asset paths and SHA256 values")
        if args.nnodes > 1:
            from .prepare_distributed import prepare_shared_assets

            prepare_shared_assets(args.config, plan, prepare_assets)
        else:
            prepare_assets(plan)
        config = load_config(args.config, overrides)
        validate_bindings(config, plan)
        backend = get_backend(config)
        if args.prepare_only:
            print(json.dumps({"prepared": True, "name": config.name}, sort_keys=True))
            return 0
        if args.nproc_per_node is not None:
            command = [sys.executable, "-m", "torch.distributed.run"]
            if args.nnodes == 1:
                command.append("--standalone")
            else:
                command.extend([f"--nnodes={args.nnodes}", f"--node-rank={args.node_rank}",
                                f"--master-addr={args.master_addr}", f"--master-port={args.master_port}"])
            command.extend([f"--nproc-per-node={args.nproc_per_node}", "-m", "visual_rl.train",
                            "--config", args.config])
            for option, value in (("--output-dir", args.output_dir), ("--resume", args.resume)):
                if value is not None:
                    command.extend([option, value])
            if args.resume_rewind:
                command.append("--resume-rewind")
            if args.eval_only:
                command.append("--eval-only")
            if args.max_updates is not None:
                command.extend(["--max-updates", str(args.max_updates)])
            os.execv(sys.executable, command)
        options = {"resume": args.resume}
        if args.resume_rewind:
            options["resume_rewind"] = True
        if args.eval_only:
            options["evaluate_only"] = True
        backend.load()(config, **options)
        return 0
    except NoSignalRolloutLimit as exc:
        print(f"training stopped: {exc}", file=sys.stderr)
        return 1
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
