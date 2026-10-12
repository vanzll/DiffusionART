# Decord wheel tag repair

The official PyPI `decord==0.6.0` Linux wheel has filename tag
`py3-none-manylinux2010_x86_64` but internal `WHEEL` tag
`cp36-cp36m-manylinux2010_x86_64`. This makes a modern `pip check` reject
Python 3.10 after installation, although the original ctypes runtime imports
and decodes video on that interpreter.

This build changes only the internal tag to match the published filename and
regenerates `RECORD`. Every runtime source, shared library, dependency metadata
and bundled license byte is preserved. It is a repaired artifact, not an
unchanged official release. The source and output SHA256 and exact changed
members are recorded in `provenance.json`.

Generate from the pinned public artifact using `repack.py --source CACHE_WHEEL
--output-dir third_party/decord`. The native video installer applies this
standard wheel after the locked dependencies and keeps its strict dependency,
import and CUDA kernel checks. Decoder output parity requires actual decoding
and is recorded separately by the acceptance task.
