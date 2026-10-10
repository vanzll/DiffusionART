# Reward and evaluation contract

The scalar contracts, manifests, and HTML rendering are model-independent and
use the Python standard library. Their tensor checks import PyTorch lazily.
The optional native image adapters below have separate model dependencies;
their loaders default to offline access. No service, credential storage or
pairwise-kernel copy is introduced.

## Native PickScore

`visual_rl.rewards.pickscore.PickScore` accepts paired RGB PIL images or floating
`[B,3,H,W]` pixels in `[0,1]`. The tensor path uses
`round(pixels * 255).clamp(0,255).uint8()` before PIL conversion, matching the
research reward wrapper; it does not truncate or score continuous pixels.
Prompts and images stay in input order, including repeated prompts.

The adapter uses the [official PickScore inference equation](https://github.com/yuvalkirstain/PickScore#inference-with-pickscore):
unit-L2 text/image features and `exp(logit_scale)` times their dot product.
For paired batches, it takes the matrix diagonal. The official demonstration's
optional multi-image softmax is not a scalar training reward and is not used.

Two score spaces are explicit:

- `space="raw"`: native PickScore logits.
- `space="fixed_normalized"`: historical training logits divided by 26, without
  clipping to `[0,1]`, batch normalization, centering or a sigmoid.

Both modes return a `ScoreResult` with raw and fixed-normalized components. The
division is represented as fixed mean 0 / std 26 with a named provenance tag;
it is a historical scale convention, not an empirical variance estimate.
Plot labels must identify the selected space. Config `reward.normalization`
uses `raw`/`fixed`; a future runner must map `fixed` to this explicit API mode.

`PickScore.from_pretrained` requires separate model and processor paths and
revisions. Pin them to immutable commit hashes in paper configs. Network access
requires `local_files_only=False`; authentication and cache locations use the
standard Hugging Face environment. The loader selects the PIL image processor
explicitly rather than accepting an environment-dependent fast backend.
The reference assets are `yuvalkirstain/PickScore_v1` and
`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`; weights are not bundled.

Weights remain frozen/eval even under a parent `.train()` call. Inference uses
the model's explicit dtype, without inheriting the policy's autocast context.
Transformers 4 tensor-returning and 5 projected-`pooler_output` feature APIs are
supported; the latter must not be projected a second time. The default loader
uses FP32, as did the inspected research recipe. No differentiable PickScore
or quantization straight-through estimator is advertised by this adapter.

Behavioral references at research commit
`3ca2a36f6b20c53a06813013234666d9c033a660`:

- `flow_grpo_alg/pickscore_scorer.py`, blob
  `048b535d4df297ed4e4514e5ca21296eed286651`.
- `flow_grpo_alg/rewards.py`, blob
  `b8dce5ea5c790e49f98c0eec0ec905a526339cfa`.

Tests cover paired order, uint8 rounding, native/scaled scores, frozen gradients,
both feature APIs, offline revision routing and real tiny CLIP forward parity.
The real pretrained checkpoint, historical dependency environment, and GPU
parity remain required before claiming reproduction.

## Native Aesthetic

`visual_rl.rewards.aesthetic.AestheticScore` uses the same paired scorer API
with a frozen CLIP ViT-L/14 projection and the original 768-dimensional scoring
head. It returns raw `aesthetic` values with no implicit normalization. The
head is an explicit local file whose SHA256 is verified before strict loading;
its bytes are bounded to 16 MiB. Model/processor revisions and the slow PIL
preprocessor are explicit. The [Aesthetic guide](sd3-aesthetic.md) contains the
asset commands, native SD3 config and research/upstream references.

Quantization matches the research wrapper's round-to-uint8 rule. Actual slow
processor tests compare tensor CHW and equivalent PIL inputs; tiny CLIP tests
compare projected/L2-normalized features and head outputs against a direct
reference. Parent `.train()` or policy autocast cannot enable reward-model
dropout or change its configured inference precision. This is a scalar scorer,
not a differentiable pixel reward, and its single-scorer config accepts only
raw unit-weight scoring. The [composition interface](composite-rewards.md)
supports explicit weights and fixed statistics across native image scorers.
ART's optional component group-advantage path is separate from scalar scoring;
the full paper four-reward scorer/profile integration remains unfinished.

## Native CLIPScore

`visual_rl.rewards.clipscore.CLIPScore` preserves the research scorer's
continuous RGB tensor preprocessing: bilinear resize, center crop and channel
normalization. It does not inherit PickScore's uint8/PIL quantization. Its
paired raw logits include CLIP's learned logit scale; fixed scores divide by
30. The native factory keeps the scorer frozen in FP32, and explicitly requires
a boolean antialias setting to avoid torchvision-version-dependent defaults.

The [CLIPScore guide](sd3-clipscore.md) provides pinned assets, commands and
source provenance. In composite scoring, only raw component values enter the
top-level fixed transform. Matching the historical scale therefore requires
mean 0 / std 30 there, not a second normalization inside the child.

## Native OCR

`visual_rl.rewards.ocr.OCRScore` preserves the first-quoted-target, normalized
text, substring and truncated edit-distance rule from the image OCR scorer.
Tensor images use the historical round-to-uint8 RGB path. The native Paddle
loader uses CPU inference with checked local assets and exact engine versions.
An explicit `zero` versus `raise` exception policy separates historical
exception-to-zero behavior from strict operational failure handling.

Both train and evaluation prompts are validated before SD3 models load. Raw
FP32 scores have no implicit divisor; composite normalization, when requested,
must be explicit. See the [OCR guide](sd3-ocr.md) for the exact rule, asset
bundle, supported version candidate and the native inference acceptance gap.

## Reward API

```python
from visual_rl.rewards import FixedNormalization, weighted_score

stats = {
    "VQ": FixedNormalization(mean=1.0, std=2.0, source="checkpoint-id/config-sha256"),
    "MQ": FixedNormalization(mean=0.0, std=4.0, source="checkpoint-id/config-sha256"),
}
scores = weighted_score(
    {"VQ": 5.0, "MQ": 8.0}, {"VQ": 2.0, "MQ": 1.0},
    space="fixed_normalized", normalization=stats,
)
assert scores.weighted == 6.0
```

- `space` is required: `raw` or `fixed_normalized`. There is no ambiguous
  `normalized` mode, batch/group centering, reward clipping, or implicit default.
- Fixed normalization is `(raw - mean) / std`, using supplied finite statistics,
  positive standard deviations, and a nonempty source identifier. Statistics
  must cover every component, including zero-weight components.
- Weights are literal finite coefficients. Signed and all-zero weights are
  allowed; they are never divided by their sum. A configuration layer requiring
  nonnegative or unit-sum weights must validate that explicitly.
- Raw and weight keys must match exactly. Missing/nonfinite components fail even
  when their coefficient is zero. Component arithmetic uses sorted names so
  JSON ordering cannot alter a weighted result.
- `ScoreResult` retains `raw`, `fixed_normalized` (or `None`), `weights`,
  `normalization`, `space`, and `weighted`. One call produces one selected-space
  total; call with each space when both totals are required.
- Real scalars and homogeneous floating PyTorch tensors are supported. Tensor
  components must have the same nonempty shape, dtype, and device. Autograd and
  shape are preserved without broadcasting or reducing the batch.
- `to_dict()` is for a scalar reporting sample, not a tensor batch. Callers must
  explicitly select/detach reporting values. `from_dict()` recomputes and checks
  stored totals and normalization against the raw components and config.

Pairwise preferences/objectives belong to the shared algorithm kernels. This
layer exposes scalar component values and input contracts, not a second
implementation of pairwise loss, comparison probability, or ranking.

## Differentiable Input

`ContinuousRGBInput(frames, prompts, fps, value_range).validate()` requires:

- Nonempty floating `BTCHW` RGB, one nonempty prompt per batch item, positive
  finite FPS, and explicitly `zero_one` or `minus_one_one` pixel range.
- Finite pixels within that range and input `requires_grad=True` by default.
  `validate(require_grad=False)` is available for a non-training validation call.
- `source="continuous_rgb"`. MP4, uint8/PIL quantization, and codec
  straight-through estimators are outside this contract. Validation neither
  copies nor detaches, clamps, resizes, encodes, or changes the tensor.

`DifferentiableReward` is a protocol for future adapters returning `[B]`
`ScoreResult` tensors. The adapter owns frozen RM weights, differentiable
preprocessing, model/tokenizer/checkpoint provenance, and native decoded-pixel
parity tests. Input provenance is a caller assertion: `requires_grad` cannot
prove a tensor has not passed through a custom STE. Continuous-RGB training
scores are **not** declared equivalent to native MP4 reporting scores.

## Evaluation Identity

```python
from visual_rl.evaluation import (
    SampleRecord, EvaluationManifest, audit_manifest, render_gallery,
)

sample = SampleRecord.from_file(
    media_root="/actual/output/root", path="videos/sample.mp4",
    sample_id="validation-0042", prompt="A moving object", seed=42,
    optimizer_step=10, framecount=53, fps=8.0, scores=scores,
)
manifest = EvaluationManifest(
    optimizer_step=10, expected_sample_ids=("validation-0042",), samples=(sample,),
)
manifest_json = manifest.to_json()
restored = EvaluationManifest.from_json(manifest_json)
audit = audit_manifest(restored, media_root="/actual/output/root")
html = render_gallery(restored, audit=audit, gallery_path="gallery/index.html")
```

The caller owns publication of the returned JSON/HTML strings; these functions
do not write to disk. `gallery_path` and sample `path` are canonical decoded
POSIX paths relative to the **same** actual media root. Gallery links are
relative to the HTML location and URL-encoded once, including literal `%`, `?`,
and `#` in filenames. Absolute paths, URLs, traversal, repeated separators,
backslashes, control characters, and symlinks inside the media root are rejected.
The root itself is explicitly resolved; no mount alias is inferred.

Each record stores `sample_id`, `prompt`, `seed`, `optimizer_step`, `framecount`,
`fps`, `path`, `content_sha256`, `prompt_sha256`, and `record_sha256`, plus optional
scalar score provenance. The manifest includes `manifest_sha256`. The record
digest binds all metadata, media hash, and score provenance; the manifest digest
binds all records and expected identities independent of ordering. These are
integrity hashes, not signatures or authentication. Seed/step/count integers
must be nonnegative JavaScript-safe integers (framecount must be positive).

A manifest covers one optimizer step. IDs must be unique and drawn from the
explicit nonempty `expected_sample_ids`; step mismatches fail. Bind scores and
media to a sample when producing its record. Never zip independently sorted
filenames, prompts, or score arrays. FPS/framecount are producer assertions;
the lightweight auditor does not decode video to verify them.

`SampleRecord.from_file()` fails on absent/unreadable files. To record a known
missing output, construct `SampleRecord(..., content_sha256=None)` directly, or
leave the record absent while retaining its expected ID. The audit distinguishes
`missing_record`, `missing_file`, `missing_hash`, `hash_mismatch`, and
`unreadable_file`, with `scores_present` recorded separately. Nothing is silently
skipped. `complete` requires every expected sample to have matching media bytes
and valid scalar scores; `completed_count` counts only such samples. Media
retention or gallery preview success must not be substituted for evaluation
completion. Empty files fail verification.

File hashing is streaming and byte-bounded (256 MiB default), detects ordinary
file replacement/size/mtime changes during the read, and refuses nonregular
files. It is not an adversarial-filesystem sandbox or a wall-clock timeout.
For remote storage, the caller's bounded worker/publisher must isolate stat/open
and reads, select a writable primary or backup root, record the actual absolute
artifact paths, and report failure if neither works. Do not put this synchronous
audit on an unbounded trainer critical path. The renderer itself does no I/O.

## Gallery and Jupyter

`render_gallery()` returns a small static scores table and local-media gallery.
Prompts/IDs/status text are HTML-escaped; embedded JSON escapes `<`, `>`, `&`, and
Unicode line separators. There are no CDN dependencies, binary media embeds,
polling loops, dynamic HTML injection, or downloads before a load-button click.
Missing records, scores, and file statuses remain visible without JavaScript.
The audit must match the manifest hash and all expected IDs.

```python
from visual_rl.evaluation import JupyterContents

contents = JupyterContents(
    base_url="https://existing-host/user/person/",
    contents_root="shared/experiment-run",
)
html = render_gallery(manifest, audit=audit, jupyter=contents)
```

Here `shared/experiment-run` is the explicitly supplied Jupyter Contents path
of the manifest's media root, not its physical mount path. An empty
`contents_root=""` deliberately maps to the Jupyter Contents root. There is
no parent/referrer inspection, guessed alias, URL-token handling, CORS/CSP
change, login flow, or new server. Credentials/query/fragment in `base_url` are
rejected; a base URL must include any existing Jupyter deployment prefix.

Jupyter blob previews work when they retain the existing authenticated page's
origin and allow scripts, same-origin fetch, and blob media under the existing
policy. Requests use that origin only, existing same-origin credentials, and
reject redirects. Null/cross-origin previews fail visibly rather than weakening
security. Contents responses must match the explicit logical path, file type,
base64 format, and manifest SHA-256. WebCrypto must be available; verification
does not silently fall back to unverified playback.

Loads have a 15-second default deadline, a 64-MiB decoded-media limit, at most two
concurrent loads, and at most four retained players/object URLs. Configuration
can lower these limits or raise them only up to hard caps (120 seconds, 256 MiB,
four concurrent loads, four loaded players). Streaming API responses are also
byte-bounded before JSON/base64 decoding. These are byte/concurrency bounds, not
an exact JavaScript heap cap. Release, hidden-page, and pagehide events cancel
loads, clear media sources, and revoke object URLs; stale completions cannot
reattach a cancelled player. A full slot reports a retryable status, not an
unbounded queue.

Without Jupyter, browsers use relative media URLs and generation-time hash
audits only; local `file:` playback requires no fetch service. Native browser
streaming is not subjected to the API's decoded-byte cap. Later file mutations,
codec errors, and access failures are reported by the player; they do not rewrite
the saved audit. A blob preview without explicit Jupyter mapping fails visibly.
Only MP4/WebM are previewed; other verified files still appear in the table.

## Provenance and Verification

The implementation is newly written portable code. Behavioral references were
read without modifying the research repository at `3ca2a36`:

- `flow_grpo_alg/video/eval_gallery_render.py`, introduced at `833659a`, and its
  regression tests: identity checks, escaped script JSON, authenticated blob
  previews, bounded loads, cancellation, and object-URL cleanup.
- Incident ledger entries `INC-20260909-JUPYTER-EVAL-GALLERY` and
  `INC-20260909-EVAL-OPTIONAL-INTERRUPTION-REVIEW` (`0cffe76`): explicit storage
  mappings and optional gallery I/O must not block training/storage failover.
  The ledger's CPU/simulated-browser evidence is not a production/GPU claim.
- `flow_grpo_alg/video/refl_reward.py` and `refl_core.py`, introduced at
  `2d11177`: fixed checkpoint normalization and continuous-RGB/native-MP4
  separation. Their model-specific references include DanceGRPO `15cc71d`,
  Transformers v4.46.2 Qwen2-VL, and UniRL `852f1511`. No implementation or model
  assets from those third parties are copied or bundled here; their license
  requirements remain applicable to any future integration.

No project license or existing third-party notice is changed by these files.
No claim of native VideoReward parity, model quality, remote filesystem
reliability, or production Jupyter compatibility is made by the portable tests.

Run CPU tests from the worktree with an environment containing CPU PyTorch and
Node (Node exercises the actual emitted JavaScript without a browser service):

```sh
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_rewards*.py' -v
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_evaluation*.py' -v
```

Tensor tests explicitly skip without PyTorch; JavaScript tests explicitly skip
without Node. A fully exercised run must have neither skip. Python-only scalar,
manifest, rendering, injection, and path checks require neither dependency.
