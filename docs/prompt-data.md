# Prompt Datasets

Native image runners consume UTF-8 JSONL with one sample per line. Each row has
a unique, nonempty `sample_id` and nonempty `prompt`. Keep repeated benchmark
prompts as separate sample IDs when they represent separate evaluation samples.

Convert a downloaded text prompt list or annotated GenEval JSONL without writing
custom code:

```bash
python -m visual_rl.prepare_data --source /path/to/train.txt --format text \
  --prefix train --output /path/to/train.jsonl
python -m visual_rl.prepare_data --source /path/to/test_metadata.jsonl --format geneval \
  --prefix eval --output /path/to/eval.jsonl
```

The command prints the manifest path, SHA256 and sample count. It preserves row
order, repeated prompts and complete GenEval annotations, and refuses to
overwrite an existing output. Text-line surrounding whitespace is stripped,
matching the reference text loader. Keep the source digest in the run record;
converting an arbitrary prompt list does not establish paper-dataset identity.

```json
{"sample_id":"validation-0001","prompt":"a photo of two apples"}
```

For custom data, set only `data.manifest` and `evaluation.manifest`; fingerprints
are recorded automatically. Optional `data.sha256` and
`evaluation.manifest_sha256` enforce a known dataset version. Paper recipes
keep their fixed pins. Hashes include annotations, row order and whitespace;
changed data is not compatible with an exact old resume.

## Scorer Annotations

An optional `metadata` object preserves benchmark annotations, for example:

```json
{"sample_id":"validation-0001","prompt":"a photo of two apples","metadata":{"prompt":"a photo of two apples","tag":"counting","include":[{"class":"apple","count":2}]}}
```

The loader checks JSON object shape and finite JSON values, not a particular
benchmark's semantics. An annotation-dependent scorer must additionally validate
its required fields, class vocabulary and relation references. This example
documents the transport format; the [GenEval guide](geneval.md) describes its
isolated native loader and remaining pretrained acceptance requirements.

Annotations are part of the immutable prompt snapshot. Every `Prompt.metadata`
access returns a new tree. Native training repeats annotations in prompt-major
order, just like its K generated images; evaluation preserves manifest order.
Decoding batches slice images, text and annotations together. Scorers declaring
`requires_metadata = True` receive paired annotations through the `metadata`
keyword. Missing annotations fail before decoding; scorers without this
capability keep their existing call signature. Composite scoring passes a fresh
copy to each annotation-dependent component.

Evaluation media use sample IDs to link back to this hashed manifest, not sorted
filenames or guessed text matches. Keep the source manifest with the run config.

## Image-to-Video References

The dataset loader also supports `image` and `image_sha256` for I2V. Relative
image paths resolve against the manifest's directory. Each referenced image is
hashed before use. For custom images, `image_sha256` may be omitted; their
contents become part of the automatically recorded dataset identity. A supplied
image hash remains strict. Text-only tasks reject image fields rather than ignoring them.
The native Wan runner accepts fixed, hashed images for evaluation; training uses
its explicitly configured online FLUX reference-image path. See the
[Wan guide](wan-runner.md) for preparation and launch commands.
