"""Evaluate any hashed held-out manifest without advancing a training budget."""

import argparse

from .config import load_config
from .data import PromptDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="complete resumable checkpoint receipt")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    from .backends.small_video_runner import run_native

    dataset = PromptDataset.load(args.manifest, expected_sha256=args.sha256, task="t2v")
    return run_native(
        config,
        resume=args.checkpoint,
        evaluate_only=True,
        evaluation_dataset=dataset,
        evaluation_label=args.label,
    )


if __name__ == "__main__":
    main()
