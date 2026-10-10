# Development

Run commands from the repository root. Keep mathematical changes separate from
runtime refactors, and test losses, gradients, update schedules, and checkpoint
state when modifying an algorithm.

## Install & Test

```bash
uv sync --locked --python 3.10 --extra dev
uv run --locked --extra dev pytest -q
```

Optional model-adapter tests use tiny, randomly initialized Diffusers models,
not downloaded pretrained weights:

```bash
uv sync --locked --python 3.10 --extra dev --extra diffusers
uv run --locked --extra dev --extra diffusers pytest -q tests/test_models_diffusers.py
uv run --locked --extra dev --extra diffusers pytest -q tests/test_sd3_text.py tests/test_pickscore.py tests/test_image_training.py
```

Pass the same extras to `uv run` as to `uv sync`; otherwise uv can remove the
optional dependencies. Gallery JavaScript tests need Node.js. Review skipped
tests alongside the pass count; CPU tests do not replace pretrained GPU checks.

## Add an Experiment

1. Extend a complete config in `configs/examples/` or `configs/paper/`.
2. Set a unique name and output directory; bind required asset variables.
3. Run the config with `--validate`, then a short training check.
4. Inspect optimizer progress, finite metrics, evaluation artifacts, and resume.

Configs reject unknown options, duplicate YAML keys, and missing environment
variables. Algorithm, model, or reward name changes reset the component's old
options; reward weights replace the entire map.

## Add a Component

- **Algorithm:** preserve its sampler, objective, loss denominator, update
  partition, and policy-state rules. Start with [algorithm contracts](algorithm-provenance.md).
- **Model:** implement conditioning, scheduling, prediction, and decoding through
  the [model interface](model-contract.md).
- **Reward:** implement scalar or differentiable scoring through the
  [reward interface](reward-contract.md); keep reporting scales explicit.

Keep credentials, machine addresses, and cluster-specific launchers out of
configs. Model weights and datasets are prepared separately under their own
terms; retain source attribution in [third-party notices](../THIRD_PARTY_NOTICES.md).
