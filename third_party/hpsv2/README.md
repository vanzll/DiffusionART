# HPSv2 Packaging Repair

This is a source-preserving local redistribution of the official PyPI
`hpsv2==1.2.0` wheel, with one explicitly audited Python-source exception.
The upstream wheel and source distribution both omit the BPE vocabulary needed
by import-time `SimpleTokenizer()` initialization. The pinned official Git tree
has the vocabulary but no `MANIFEST.in`; its unchanged package-data glob is not
an accepted packaging fix.

## Changes

- Add official `bpe_simple_vocab_16e6.txt.gz`, pinned by Git revision and SHA256.
- Remove exactly the unused `from turtle import forward` line in `factory.py`.
  The actual PyPI source imports turtle. An AST audit confirms the imported name
  is unused. No other Python source is changed; model/tokenizer code is preserved.
- Insert wheel build tag 2 and regenerate all `RECORD` hashes and lengths.
- Retain original dependency metadata and the Apache-2.0 license byte-for-byte.

`NOTICE` records the modification and upstream attribution. The wheel contains
the original license at `hpsv2-1.2.0.dist-info/LICENSE`. `LICENSE` is a readable
copy normalized to LF; copyright comments in all upstream Python files remain.

## Install

From the repository root, either installer can use this standard wheel:

```sh
python -m pip install third_party/hpsv2/hpsv2-1.2.0-2-py3-none-any.whl
uv pip install third_party/hpsv2/hpsv2-1.2.0-2-py3-none-any.whl
```

These commands resolve the preserved upstream dependencies. They do not repair
an installed package or download HPS model weights. For an environment whose
runtime dependencies are already supplied, use `--no-deps` with either command.
The upstream metadata still pins `pytest==7.2.0` and `pytest-split==0.8.0`;
pip users combining newer developer tools must manage that conflict explicitly,
or install this wheel with `--no-deps` only after providing its runtime dependencies.
The repository's uv overrides already handle these upstream test-tool pins.

The repository's `[tool.uv.sources]` is already configured to use this relative wheel path.
Do not publish or describe this artifact as an unchanged official release.

## Rebuild and Verify

No build tool dependency is needed beyond Python's standard library. The script
downloads only the bounded official wheel and BPE inputs and verifies their
fixed SHA256 values before parsing them. It never fetches model weights.

```sh
python third_party/hpsv2/repack.py
python third_party/hpsv2/repack.py --verify-only
```

For an offline rebuild, provide `--upstream-wheel /path/to/original.whl` and
`--bpe /path/to/official/bpe_simple_vocab_16e6.txt.gz`. Input hashes are still
mandatory and built in. ZIP timestamps, entry order, permissions, and compression
level are fixed. The recorded byte-for-byte rebuild was performed with Python
3.10.20 and zlib 1.2.12; different zlib implementations can change compressed
archive bytes, so consumers must check the recorded artifact SHA256 rather than
silently replacing the pinned artifact with a different build.

`repack.py --verify-only` compares the entire wheel to a deterministic rebuild,
checks every RECORD entry, permits only the declared member changes, checks all
other Python files and dependency metadata byte-for-byte, and writes
`verification.json`. `provenance.json` identifies the released local artifact.

## Host Validation Boundary

Acceptance checks run on macOS CPU, not Linux or a GPU. The repaired vendor API
is also imported with imports of `turtle`, `tkinter`, and `_tkinter` blocked; no
Tk dependency is allowed to be hidden by the Mac host installation. The original
wheel fails on a headless Python lacking Tk, typically with
`ModuleNotFoundError: No module named 'tkinter'`; use this build-2 wheel instead
of changing installed site-packages. This audit does not claim full headless
Linux acceptance, CUDA numerical parity, or acceptance of the 1.97 GB checkpoint.
See `validation.json` for the actual commands and results.
