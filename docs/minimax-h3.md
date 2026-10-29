# MiniMax-H3 T2AV

This opt-in backend adds text-conditioned audio/video training with ART, NFT
and FlowGRPO Fast. The six recipes are integration examples, not accepted paper
results. CPU tests and upstream runs do not establish real-weight GPU acceptance.
The first version excludes FL2VA, Ref2VA and sequence parallelism. It uses the
guidance-distilled transformer without an unconditional CFG forward.

## Shared inference protocol

Base H3 examples inherit [one inference profile](../configs/inference/minimax_h3.yaml):
**1344×768, 124 frames at 24 fps, 50 sigma boundaries / 49 denoiser forwards**, video
`shift=12` and audio `shift=3`. This matches the unaccelerated workload in the
[reference inference recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-H3), with the
[native Diffusers schedule semantics](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3#two-schedulers).
The released transformer is CFG-distilled: `guidance_scale=1` means one conditional
forward, with no extra unconditional pass or acceleration LoRA.
The current six-reward ART post-training recipe explicitly overrides rollout to
864×480, 124 frames and 11 boundaries / **10 actual denoiser forwards**, with
training-time evaluation disabled. The archived multi-duration recipe retains
the 768p protocol. See the post-training recipes below.

In the base profiles, rollout and evaluation use the same full grid and canvas. Evaluation leaves
`sampling_steps`, `height` and `width` unset so it follows rollout settings,
including explicit overrides; selecting sparse training positions does not
shorten generation. ART and NFT generate deterministic ODE trajectories.
FlowGRPO Fast retains its algorithm's three-position CPS stochastic window
within its rollout; evaluation remains deterministic ODE. Its CPS rollout is
therefore not claimed identical to reference inference. The configured Fast
candidate window remains `[0,8)` within the new 49-transition trajectory; it is
not silently expanded to the entire grid.

Text uses the pinned checkpoint's tokenizer with `add_special_tokens=False`
and the native `get_qwen3vl_prompt_embeds`, taking the unnormalized Qwen3-VL
layer 50 hidden state. No chat template is added. Prompt expansion/Context-IR
is external preprocessing and is not silently applied to the training data.
Video/audio VAEs, latent normalization and 24 fps / 32 kHz stereo output are unchanged.

The previous 384×256/10-boundary profile was a low-cost integration setting.
The new 50-boundary/768p profile changes generation cost and activation memory;
physical microbatches may need hardware-specific tuning. Keep P/K, optimizer
windows and loss denominators fixed when tuning execution chunks. Existing
runs retain their frozen configs; use those configs for checkpoint continuation,
not these updated examples. Matching inference settings does not establish
numerical equivalence of the complete real-weight pipeline by itself.

## Separate native environment

The pinned Diffusers source requires **Python >=3.11**. Keep existing Python3.10
backend environments intact and create an H3 environment separately:

```bash
uv venv --python 3.12 .venv-h3
uv pip install --python .venv-h3/bin/python -r environments/minimax-h3.txt
uv pip install --python .venv-h3/bin/python -e .
.venv-h3/bin/python -m pytest -q tests/test_minimax_h3_model.py tests/test_minimax_h3_assets.py
.venv-h3/bin/python -m visual_rl.models.check_minimax_h3_environment --mode model-only --tiny-forward
```

The environment file records the native CPU oracle versions. Select a PyTorch
build compatible with the target GPU/driver and record actual CUDA/NCCL versions
for GPU acceptance. A CPU dependency lock does not verify CUDA compatibility.
Install `ffmpeg` and `ffprobe` on PATH. Native scorer dependencies and weights
are separate from the model environment and are supplied through a reward manifest.
No vLLM, Ray or verl dependency is introduced.

For Linux x86-64 with Python3.12 and a CUDA12.8-compatible driver, the separate
`environments/minimax-h3-cu128.txt` deployment file pins matched CUDA wheels for
Torch2.11.0, torchaudio2.11.0 and torchvision0.26.0, including their official wheel
SHA256 values. It retains the same Diffusers commit and H3 source checks, and pins
AudioBox source to its0.0.4 release commit plus the native scorer dependencies.
Its companion constraints file fixes the resolved transitive Linux/Python3.12
deployment, including native NCCL2.28.9; it is not the old cluster's NCCL2.29.7.
The original `minimax-h3.txt` remains the CPU oracle inventory, not a promise that
its default CUDA13 wheels work with an older driver. Use a fresh environment:

```bash
uv venv --python /usr/bin/python3.12 .venv-h3-cu128
uv pip install --python .venv-h3-cu128/bin/python -r environments/minimax-h3-cu128.txt
uv pip install --python .venv-h3-cu128/bin/python -c environments/minimax-h3-cu128-constraints.txt -e .
.venv-h3-cu128/bin/python -m visual_rl.models.check_minimax_h3_environment --mode model-only --tiny-forward
```

A deployment still needs full real-weight training/checkpoint/resume/media
acceptance artifacts; an installation lock or collective gate alone is not
acceptance. This target does not change the model, ART objective or precision
settings. Keep only the site's verified
driver compatibility directories when cleaning `LD_LIBRARY_PATH`; do not inherit
Torch libraries or `site-packages` from another training environment. Record the
actual driver/core/plugin versions and run the target's own collective and real
FSDP gates. A fresh environment must also provide `ffmpeg`/`ffprobe`; the dependency
file does not install system executables or prepare licensed scorer assets.

The optional `scripts/minimax_h3_*` KML execution helpers use one MPI node owner
per host and the site's existing generation-fenced lease library, supplied via
`H3_LEASE_LIBRARY`. They are operational helpers, not another training backend.
Each attempt needs its own immutable `launch.env`, nonce, source commit, config,
output and preparation ID. Owners reject foreign GPU work or a dirty/wrong
checkout, propagate any node failure, and verify nonce descendants plus GPU
contexts drained before the current fenced owner restores holding. Do not add
another `gpu_free`, unconditional holding trap or per-node tmux supervisor.
Torchrun's `PET_LOG_DIR`/redirect/tee options retain separate worker logs. All
CUDA/Triton/Torch/Numba and W&B staging caches stay under the selected task root;
setting only `XDG_CACHE_HOME` is not sufficient for every CUDA cache.
For the second clean-environment reproduction, set `H3_KERNEL_CACHE_ROOT` to a
new subdirectory of that task root so compiled kernel caches are independent
as well; verified immutable model/download assets may still be reused.

Initial real32GPU capacity trials on140.40GiB H800s selected the conservative
rollout8/update8 profile: one complete rollout and two finite updates passed on
all32ranks, with51.05GiB worst allocated/96.37GiB reserved and97.79GiB sampled
NVML peak. Update16 also completed, but used132.84GiB reserved/134.26GiB sampled
NVML, leaving only4.37% physical headroom for an observed update-phase change
from146.27s to140.68s. That small single-cycle difference is not a controlled
speedup claim or a universal optimum. Sampling stayed8 in both cases, its real
local trajectory limit. Their five initial reward means/stds matched exactly;
per-window losses/gradient norms differed, so bit-exact training-batch
equivalence is not claimed. Complete four-rollout/checkpoint/continuation
acceptance has a separate evidence ledger in
[the acceptance PR](https://github.com/vanzll/DiffusionART/pull/4).

Independent eight-GPU trials on the same140.40GiB devices selected
rollout4/update2. One complete real rollout and two updates used77.49GiB peak
allocated/118.58GiB reserved and120.10GiB sampled NVML, leaving14.46% observed
physical headroom. A bounded rollout8/update4 trial also completed; its update
phase fell from895.60s to746.41s, but reserved memory reached131.09GiB and
sampled NVML132.61GiB, leaving5.55% before the formal initial evaluation,
checkpoint and four-rollout lifecycle. The conservative profile retains margin
for those phases. This measured tradeoff does not establish a universal optimum
or prove that the larger candidate will OOM.

The two trials' model-collection intervals were174.68s and173.63s. Their total
rollout times282.11s and338.16s include conditioning/preparation intervals of
107.43s and164.53s, respectively. These host phase boundaries do not isolate
GPU kernels or communication. Decode/reward/preparation took619.47s and573.77s.
Both trials had finite five-scorer populations and nonzero gradients, but their
reward values differed; numerical identity across sampling batch sizes is not
claimed. These are capacity trials with evaluation disabled. Their matching
science, actual batch shapes, full online metrics and later formal lifecycle
receipts are recorded in the acceptance PR.

The explicit scorer dependency inventory is
`environments/minimax-h3-scorers.txt`. It is **not a completed installation lock**:
target-compatible torchaudio/torchvision wheels, the exact AudioBox implementation
revision, CLAP audio dependencies and dependencies of the supplied Synchformer
source still need native-load verification. HPSv3/VideoAlign use the in-repository
Qwen2VL reward implementation and pinned transformers/peft/safetensors; installing
an unrelated HPS package does not satisfy that interface. Do not silently
resolve these gaps with latest-version installs or claim five-scorer readiness
from the model-only environment test.

The environment probe runs in a bounded child process (120 seconds by default,
configurable up to 600). It records package versions, actual imported source-file
SHA256 values, exact pinned native Diffusers file hashes and ffmpeg/ffprobe
versions. Missing imports or source drift fail; a matching `__version__` alone
does not pass. It performs no downloads and loads no pretrained weights.
After preparing all scorer assets, test their imports and actual Synchformer
source separately:

```bash
.venv-h3/bin/python -m visual_rl.models.check_minimax_h3_environment \
  --mode all-scorers --reward-manifest "$H3_REWARD_MANIFEST"
```

This does not score media or establish scorer-weight readiness. For a tiny
random-weight CUDA forward/backward, run the following **only on an explicitly
authorized GPU allocation**, never on a development host:

```bash
.venv-h3/bin/python -m visual_rl.models.check_minimax_h3_environment \
  --mode model-only --cuda --authorized-gpu-allocation
```

The explicit allocation flag is required. This toy check does not launch
distributed training, validate communication or load real model weights.

| Source | Immutable identity |
| --- | --- |
| `MiniMaxAI/MiniMax-H3` | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Diffusers | `73faca368c0da77645ea58836aa8780d2d8c5353` |
| verl-omni reference | `ca61e80c42c883e1efa161f6b609cf897fba1787` |
| ART PR744 reference | `93c8a747252d27e4549fba4e4b23abc19d6114bb` |

The last two are provenance references, not runtime dependencies. Model/scorer
licenses remain separate from software licensing. No CLI accepts licenses or
changes account access; obtain any required access directly from the asset owner.
See the [fixed model tree](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe)
and [fixed native Diffusers modules](https://github.com/huggingface/diffusers/tree/73faca368c0da77645ea58836aa8780d2d8c5353/src/diffusers/modular_pipelines/minimax_h3).

## Prepare native model assets before launching ranks

Use writable persistent storage with quota headroom. At the fixed revision,
selected components total approximately **144.05 GB**: transformer 66.28 GB,
conditioner 66.73 GB, video VAE 10.42 GB and audio VAE 0.61 GB. This excludes Ref2VA
weights. Leave additional space for cache, outputs and checkpoints; avoid
container overlay storage. On a system with primary/fallback storage, choose a
writable root first and record the actual resolved path. Do not assume those
roots provide independent failure domains.

Inspect downloads metadata/configuration/tokenizer files, never safetensors:

```bash
export H3_ASSET_CACHE=/absolute/persistent/path/h3-cache
mkdir -p "$H3_ASSET_CACHE"
.venv-h3/bin/python -m visual_rl.models.prepare_minimax_h3 \
  --inspect --cache-dir "$H3_ASSET_CACHE"
```

The inspect receipt has `ready=false` and cannot pass training validation. To
prepare actual weights after storage/access are ready, run once on the coordinator:

```bash
.venv-h3/bin/python -m visual_rl.models.prepare_minimax_h3 \
  --download --cache-dir "$H3_ASSET_CACHE" \
  > "$H3_ASSET_CACHE/h3-prepare-summary.json"
export H3_MODEL="$(.venv-h3/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["root"])' \
  "$H3_ASSET_CACHE/h3-prepare-summary.json")"
.venv-h3/bin/python -m visual_rl.models.prepare_minimax_h3 \
  --validate-only --model-root "$H3_MODEL"
```

Offline validation performs no Hub requests. The atomic `h3_asset_manifest.json`
contains actual root, repository/source revisions, native component identities,
configuration SHA256 values, selected weight filenames/sizes and Hub Git/LFS
identities. Indexed shards must all be present. The allowlist excludes
`transformer_ref`, original checkpoint formats and sample videos. Preserve the
manifest when copying a prepared snapshot. Hub authentication uses existing
credentials or `HF_TOKEN`; never record token values in logs or configs.

Default weight verification is **size only**, explicitly recorded in the receipt.
It cannot detect equal-size weight corruption. Add `--hash-weights` to download
or validation to read every weight byte and compare Hub LFS SHA256 values;
reading 144 GB may take minutes. A previous strong-hash receipt is always rechecked.
Neither asset readiness nor a directory name establishes GPU acceptance.

## Prompt data and native scorer assets

Provide UTF-8 JSONL with unique stable IDs, a generation prompt and explicit
modality reward text. For example, this schema illustration is not a benchmark:

```json
{"sample_id":"example-0001","prompt":"A violinist plays on a small stage with clear violin audio.","metadata":{"reward_text":{"video":"A violinist performs on a small stage.","audio":"Clear solo violin music."}}}
```

Text-only H3 rejects reference-image fields. Preserve dataset bytes and bind SHA256:

```bash
export H3_TRAIN_MANIFEST=/absolute/persistent/path/train.jsonl
export H3_TRAIN_SHA256="$(.venv-h3/bin/python -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$H3_TRAIN_MANIFEST")"
export H3_REWARD_MANIFEST=/absolute/persistent/path/h3-native-reward-manifest.json
```

The five native scorers are VideoAlign, HPSv3, AudioBox, CLAP and DeSync. A model
snapshot is not a ready reward stack. Supply their licensed implementations,
weights, config and tokenizer files; do not replace them with arbitrary functions
bearing the same names. The reward manifest has top-level `schema_version:1`,
`paths`, `sha256`, and `provenance`. Every asset has provenance containing
`repo`, immutable forty-hex `revision`, and `license`. For a directory asset,
`sha256[asset]` is a complete recursive mapping from relative filename to SHA256;
symlinks are forbidden. A single-file asset uses the relative key `"."`.
The inventory covers implementation source and preprocessing/tokenizer files,
not only weights. Paths may be relative to the reward manifest location.

The exact seven asset names in all three mappings are `video_align`, `hpsv3`,
`audiobox`, `clap`, `desync`, `video_align_base`, and `desync_source`.
VideoAlign's base configuration is Qwen2-VL-2B, whereas the released HPSv3
checkpoint has a3584-wide7B backbone. Do not initialize HPS from VideoAlign's
1536-wide2B configuration. Bind `hpsv3` to an explicitly inventoried directory
containing `HPSv3.safetensors` and `base_model/` with the matching Qwen2-VL-7B
configuration, tokenizer and processor files. Base weights are unnecessary because
HPS loads its strict full checkpoint. The bundle overrides the shared fallback
base path; the legacy single-checkpoint interface still accepts an explicit
matching base path. Preserve the original bytes and record both HPS and base-model
immutable repository revisions/licenses in this composite asset's provenance.
The seven top-level asset names and all reward preprocessing/calibration stay
unchanged. A directory existing is not evidence that its real weights load.
Create a paths-only JSON mapping with those names and the actual local file or
directory paths, plus a provenance-only JSON mapping with those same names and
each asset's actual `repo`/`revision`/`license`. Relative input paths resolve
against the paths JSON location. Generate the complete hash manifest offline:

```bash
.venv-h3/bin/python -m visual_rl.models.prepare_minimax_h3 \
  --reward-paths /absolute/persistent/path/reward-paths.json \
  --reward-provenance /absolute/persistent/path/reward-provenance.json \
  --manifest "$H3_REWARD_MANIFEST"
```

This hashes every regular file, including nested source/config/tokenizer files,
and rejects empty directories, symlinks and non-immutable revisions. Hashing
large scorer directories may take minutes. It neither loads scorers nor accepts
their licenses. The result stores absolute paths and the supplied provenance;
hashing proves byte identity, not that the user-entered provenance is correct.

Video routing uses VideoAlign + HPSv3 + DeSync; audio routing uses AudioBox + CLAP
+ DeSync. DeSync appears in both routed losses; the shared Fast scalar counts it
once. Reward transforms, direction and normalization are scientific settings.
Plot individual rewards rather than treating their differently scaled sum as
a universal quality measure.

## Six independent recipes

Validate all bindings before starting training. Validation is network-free:

```bash
for recipe in art_lora art_full nft_lora nft_full fast_lora fast_full; do
  .venv-h3/bin/python -m visual_rl.train \
    --config "configs/examples/minimax_h3_${recipe}.yaml" --validate
done
```

The following are alternative commands, not a simultaneous batch. Use an
authorized allocation; worker count must divide global prompt count. Give each
run an available persistent output directory. The launcher prepares assets
before torchrun; do not nest `--nproc-per-node` inside a distributed process.
Recipes use the source default of 10 schedule boundaries (9 actual transitions).
The software acceptance learning rate is uniformly `1e-5`; this is not a claim
that it is an optimal or matched paper hyperparameter. LoRA uses the source
target suffixes `to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, and `ff.net.2`,
with rank 64 and alpha 128. Formal learning-rate selection remains a separate
scientific decision.

```bash
export H3_NPROC=8
.venv-h3/bin/python -m visual_rl.train --config configs/examples/minimax_h3_art_lora.yaml --nproc-per-node "$H3_NPROC"
.venv-h3/bin/python -m visual_rl.train --config configs/examples/minimax_h3_art_full.yaml --nproc-per-node "$H3_NPROC"
.venv-h3/bin/python -m visual_rl.train --config configs/examples/minimax_h3_nft_lora.yaml --nproc-per-node "$H3_NPROC"
.venv-h3/bin/python -m visual_rl.train --config configs/examples/minimax_h3_nft_full.yaml --nproc-per-node "$H3_NPROC"
.venv-h3/bin/python -m visual_rl.train --config configs/examples/minimax_h3_fast_lora.yaml --nproc-per-node "$H3_NPROC"
.venv-h3/bin/python -m visual_rl.train --config configs/examples/minimax_h3_fast_full.yaml --nproc-per-node "$H3_NPROC"
```

### ART full deployment profiles

Two additional profiles inherit the same ART full scientific recipe. They keep
P32/K8 (256 trajectories per rollout), complete-prompt covariance, two disjoint
updates of 128 trajectories, and 48 nonfinal training positions per trajectory.
Only the name, output directory, physical microbatches and deployment notes
differ. A microbatch is an execution chunk, not an additional optimizer update.

| Profile | Launch topology | Rollout microbatch | Update microbatch |
| --- | --- | --- | --- |
| `minimax_h3_art_full_8gpu.yaml` | 1 node x 8 FSDP ranks | 4 | 2 |
| `minimax_h3_art_full_32gpu.yaml` | 4 nodes x 8 FSDP ranks | 8 | 8 |

These retain the established deployment microbatches. Changing the inference
canvas or grid does not automatically reset them. Previous 256p acceptance does
not establish 768p activation memory: tune rollout and update microbatches only
from actual memory measurements or OOM evidence, retaining P32/K8/U2. The recipe
does not silently reduce physical batches to one.
YAML does not reserve or count GPUs: the launcher determines the world size.
Both profiles use full-world FSDP (`shard_degree=world_size`, `replicate_degree=1`), with no TP/SP;
32 ranks therefore communicate parameter shards across nodes. Extra FSDP ranks
reduce model-state memory, not the activation memory of one physical batch.

For one authorized eight-GPU node:

```bash
.venv-h3/bin/python -m visual_rl.train \
  --config configs/examples/minimax_h3_art_full_8gpu.yaml --nproc-per-node 8
```

For four authorized eight-GPU nodes, use the allocation's persistent node
executor. Run the following once on **each** node, with its own
`H3_NODE_RANK=0,1,2,3`. All nodes must share the exact checkout, asset bindings,
config/output paths, `H3_MASTER_ADDR`, unused `H3_MASTER_PORT`, and a fresh
`H3_PREPARATION_RUN_ID`. The node executor must propagate a failed node to the
whole attempt; do not start four independent standalone jobs. `--nnodes` and
the rendezvous flags are already supported by the public entrypoint:

```bash
.venv-h3/bin/python -m visual_rl.train \
  --config configs/examples/minimax_h3_art_full_32gpu.yaml \
  --nproc-per-node 8 --nnodes 4 --node-rank "$H3_NODE_RANK" \
  --master-addr "$H3_MASTER_ADDR" --master-port "$H3_MASTER_PORT" \
  --preparation-run-id "$H3_PREPARATION_RUN_ID"
```

On sites requiring shell initialization or GPU leases, the executor's final
child shell must initialize that environment and use the existing fenced lease
protocol. The public command does not release GPU holders or allocate nodes.

Tune `sampling.microbatch` and `training.microbatch` independently in an
inherited config, retaining P/K, the time pool, loss denominator and optimizer
boundaries. At world size 32 each rank owns eight rollout trajectories and 32
sample/time items per ART update, so a larger rollout microbatch than eight
does not increase physical concurrency. Measure the worst-rank peak allocated
and reserved memory, device usage, CPU/cgroup peaks and full-phase throughput,
including backward, optimizer step, decode, rewards and checkpointing. Keep
working headroom rather than merely filling the allocator cache. Freeze the
chosen config before checkpoint/resume acceptance; changing microbatches
changes the config fingerprint and must use a separate tuning attempt.

### Physical H3 batching

The H3 runtime enables strict PyTorch deterministic algorithms and requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. The supplied H3
runtime shell sets this environment variable; callers embedding the runtime
must set it before using CUDA. The runtime restores the caller's PyTorch flags
and environment on exit, including failure. Conflicting workspace settings or
late configuration are rejected. BF16 compute, FP32 masters, reward formulas
and optimizer boundaries remain as configured. Unsupported nondeterministic
operations raise an error rather than continuing with a warning.

In a controlled full-model, 32-rank checkpoint replay, fixed cached inputs and
unchanged model/Adam/RNG produced different gradients on three backward passes
without this policy (worst shard relative L2 difference 0.02632). All 32 shards
were bit identical across three passes with the combined policy. This establishes
the tested policy's sufficiency for that replay, not the identity of a faulty
operator or a guarantee of bit equality across hardware, versions or world sizes.
Full training/checkpoint continuation must still be checked independently.

Each replay attempt above regenerated its own cached inputs. The input hashes
differ across the two policies, while model and Adam hashes match on all ranks;
the change in gradient norm is therefore not a matched-input causal comparison.
Use complete continuous/resumed training to validate the chosen policy.

H3 logs decoded/scored sample counts with prompt IDs and trajectory indices,
so a long reward phase has visible local progress before the global phase ends.
For checkpoint continuation diagnosis, optional `H3_INPUT_AUDIT=1` writes
pre-update hashes of cached objective tensors/options, conditioning, prompt
identity, RNG and the sampling generator. Set a fresh safe
`H3_ATTEMPT_NONCE` for every execution. Receipts live under
`input-audits/<nonce>/rollout-*/rank-*.json`; separate nonces preserve both
continuous and resumed observations without replacing earlier receipts.
Derived statistics and mutable diagnostic accumulators are excluded; model and
Adam integrity use the separate restore audit. The input audit checks that RNG
is unchanged and rejects unknown value types or receipt replacement.

Keep these optional audit flags identical across a comparison. Input hashing
adds CPU work and a coordinated phase check before updates; its measured time
is included in reward/preparation timing. Preserve that observation overhead
when comparing stage costs. The supplied H3 runtime shell leaves input auditing
disabled unless explicitly requested.

The adapter executes a true batch through the transformer rather than looping
over requests. Mixed text lengths use masked padding in both the text refiner
and the main attention stack. Each sample retains its original rotary positions
and separate video/audio noise conditions, including training batches that mix
time positions. No attention is shared between different samples. The H3-only
extension retains checkpoint parameter names and the pinned upstream source;
it does not change the other model backends. All ranks follow the same module
call order even when their text lengths and sigmas differ.

Conditioner and transformer cannot both occupy an 80 GB GPU before sharding.
Load on CPU and select trainable parameters before FSDP/shadows. The first
integration explicitly keeps conditioning on CPU (`conditioning_device: cpu`)
and stages decoding separately. At world size8, full FP32 master shards plus
Adam moments already occupy about49.7 GB/rank before gradients or activations;
a66.7 GB conditioner cannot then fit on an80 GB GPU. CPU text encoding can be
slow and requires substantial host memory; record its cost. Native audio decode retains its FP32
modules; applying a whole-VAE bf16 cast changes audio amplitude. Tiny models do
not establish real-model memory feasibility. Before launching real weights,
measure available host/cgroup memory as well: independent full-policy FP32
materialization plus the conditioner and VAEs has a conservative upper bound
of about210 GB/rank (1.68 TB for eight local ranks), before scorers and
temporary buffers. Shared mmap pages may reduce physical usage, but this has
not been measured. The current implementation does not implement rank0/meta
initialization or automatically claim that eight80 GB GPUs are sufficient.

## Model and algorithm semantics

| Algorithm | Successful steps per complete rollout | Behavior |
| --- | --- | --- |
| ART | 2 | Complete-K routed covariance before disjoint half-K windows per prompt; separate modality RMS/mask/feature means, equal loss weights; exclude final transition |
| NFT | 1 | Endpoint re-noising; separate normalized-x0 mirror losses, beta1, equal weights; all positions/samples accumulate before one step |
| Fast | 2 | H3 CPS observed-next-state replay in a sparse continuous window; combine0.5/0.5 modality feature-mean log probabilities before one PPO ratio |

All three omit pretrained-reference KL/MSE objects, forwards and checkpoint
payloads. NFT retains EMA-old, frozen throughout U1; after success refresh with
`decay=min(j*0.00125,0.6)`, where `j` is the zero-based completed-update index.
The first refresh copies current. Routed advantages use group centering and
the corresponding global raw reward population std plus1e-4. Fresh NFT noising
uses a per-row random permutation of the video grid excluding terminal zero,
with source-matched `floor(1000*sigma)/1000` quantization and the same fresh
sigma passed to both modalities; rollout still has separate modality schedules.
These choices differ from upstream H3 NFT U2/beta 0.1/reference/EMA defaults.

Native states are video `[B,24,T,H,W]` and stereo audio `[B,2,32,Ta]`. Upstream
patchification and rotary layout are used. One joint AV forward per request
returns both differentiable outputs. Public velocity is `dz/dsigma`; native H3
uses time `1-sigma` and the opposite velocity sign. The native schedule counts
terminal zero in requested length; transitions are `len(sigmas)-1`.

Decode carries native 24 fps and 32 kHz stereo, with no hidden trimming or
resampling. Geometry derives from VAE metadata and native frame-count helpers,
not an assumed token ratio. Routed targets do not isolate shared parameter
gradients: the model still attends jointly across modalities.

## Checkpoint continuation and acceptance

Resume from an explicit completed-rollout checkpoint receipt. Do not silently
change dataset/scorers/schedules, algorithm or LoRA/full mode:

```bash
.venv-h3/bin/python -m visual_rl.train \
  --config configs/examples/minimax_h3_nft_lora.yaml \
  --nproc-per-node "$H3_NPROC" --resume /absolute/persistent/path/checkpoint-receipt.json
```

Acceptance independently covers ART/NFT/Fast × LoRA/full: finite real-weight
updates and correct U1/U2 counts, native media/rewards, checkpoint save/load/
continuation and distributed rank progress. Keep healthy experiments running.
CPU oracles, ready assets and launch commands do not replace these runtime
records. The existing `track.md` holds actual evidence and remaining limits.

## 480p, ten-step ART post-training

The current training-only recipe has three public execution profiles.
**Full64 (CP2) is the full-parameter recipe used for the training curves.**

| Profile | Physical GPUs | Logical data replicas | Context parallelism | Rollout / update microbatch |
|---|---:|---:|---:|---:|
| [`minimax_h3_art_posttrain_full_32gpu.yaml`](../configs/examples/minimax_h3_art_posttrain_full_32gpu.yaml) | 32 | 32 | 1 | 8 / 4 |
| [`minimax_h3_art_posttrain_full_64gpu_cp2.yaml`](../configs/examples/minimax_h3_art_posttrain_full_64gpu_cp2.yaml) | 64 | 32 | 2 | 8 / 8 |
| [`minimax_h3_art_posttrain_lora_8gpu.yaml`](../configs/examples/minimax_h3_art_posttrain_lora_8gpu.yaml) | 8 | 8 | 1 | 1 / 1 |

Full32 reduces the update microbatch from 8 to 4 after the original CP1/B8
backward OOM, retaining rollout microbatch 8. Gradient accumulation preserves
the effective batch, optimizer windows and loss denominators. Full64 adds
sequence splitting with CP2 and explicitly retains update microbatch 8.

All use **864×480, 124 frames at 24 fps**
(about 5.17 seconds), guidance scale 1, native Euler integration and the original
video/audio shifts 12/3. The native scheduler takes **11 sigma boundaries**,
including zero, for **10 actual transformer forwards**; `steps: 10` would instead
execute nine transitions. No acceleration LoRA or extra CFG branch is added.

Full32 uses rollout/update microbatches **8/4** and Full64 uses **8/8**; the 8-GPU LoRA
profile uses **1** with rank32/alpha64 attention and FFN adapters. FSDP8 and
activation checkpointing remain enabled. Smaller microbatches accumulate toward
the same effective batch and optimizer boundary. P32/K8 means 256 trajectories per
rollout. Compute each reward's statistics over its complete K8 group, then split
every prompt's samples into disjoint K4+K4 windows: two global optimizer steps,
128 trajectories per step. This is not two epochs over the same 256 trajectories.
The budget is 400 optimizer updates, normally 200 completed rollouts. AdamW
uses learning rate 1e-5, betas 0.9/0.999, epsilon 1e-8, weight decay 1e-4 and
maximum gradient norm 1. BF16 compute, FP32 FSDP masters and activation
checkpointing remain unchanged.

For each rollout, independently sample two video positions from the first four
and two audio positions from the first three, without replacement. Their sigma
candidates are video **[1, 0.990826, 0.979592, 0.965517]** and audio
**[1, 0.964286, 0.923077]**. Replay the union of original paired states and sigmas;
each modality contributes only at its own two positions. Average each modality
over those two positions and combine the modality losses with equal weight.
Keep the six-reward routes, complete-K8 sample-std normalization, corrective
masks and frozen pretrained velocity MSE coefficient **1e-3** described below.
`duration_frames: [124]` fixes the duration; there is no long-video rotation.

All evaluation cadence fields are disabled: no initial, periodic, resumed or
final evaluation is generated. Rollout decoding and six-reward scoring are still
required for training. Save a complete model/Adam/RNG/prompt-cursor/reference
checkpoint every **50 optimizer updates** and at completion; retain the latest
**two successfully published** snapshots. Allow room for a third snapshot while
it is being written; do not delete the last recoverable checkpoint to admit a
new write. These are full training checkpoints, not standalone HF exports.

The disabled evaluation fields bind the future offline protocol in advance:
1344×768, 124 frames, 50 boundaries / 49 forwards, and the fixed held-out manifest.
They generate no evaluation during training. Prepare the pinned OmniNFT splits
[below](#omninft-train-and-eval-prompts), but leave every cadence at zero/empty. For the **CP1 profiles**,
later, on the same FSDP
topology and using the original output directory and data/asset bindings, create
an inherited evaluation config that changes **only** `evaluation.at_updates`.
Use the exact training recipe as its parent (LoRA8 or Full32):

```yaml
extends: /absolute/path/configs/examples/minimax_h3_art_posttrain_full_32gpu.yaml
evaluation:
  at_updates: [2]
```

This enables the eval loader; `--eval-only` evaluates the selected checkpoint
regardless of that cadence value and does not start training.

```bash
.venv-h3-cu128/bin/python -m visual_rl.train \
  --config /absolute/path/offline-eval.yaml --resume latest --eval-only \
  --nproc-per-node 8 --nnodes 4 --node-rank "$H3_NODE_RANK" \
  --master-addr "$H3_MASTER_ADDR" --master-port "$H3_MASTER_PORT" \
  --preparation-run-id "$H3_PREPARATION_RUN_ID"
```

Use the existing fenced site launcher for the real multi-node command. Evaluate
a quiescent checkpoint under exclusive run ownership; do not race an active
trainer or change training cadence when returning to training. The backend
checks the training resume signature and retains the checkpoint's identity;
changing evaluation cadence does not bypass model/data/optimizer validation.
Offline evaluation does not call `optimizer.step` or rewrite checkpoint state.

### OmniNFT train and eval prompts

Use the **same VGGSound prompt splits as verl-omni OmniNFT**: the complete
`train_metadata_20k.jsonl` (19,487 rows) and `test_metadata.jsonl` (16 rows) from
OmniNFT revision `fb9237f6e74edf0d0f2a683f4d975b79fde588fe`. Keep their order,
duplicate rows and text unchanged. Map `prompt_av` to the generation prompt,
`prompt_v` to video reward text and `prompt_a` to audio reward text. This trainer
uses a lossless JSONL schema conversion of the prompts that verl-omni writes
to parquet; it does not require video/audio files. The current six-reward LoRA8,
Full32 and Full64 recipes all bind these exact train/test splits.
Online evaluation remains disabled, and CP2 evaluation is still unsupported.

After selecting a writable persistent output directory, prepare and bind both
splits on the coordinator:

```bash
source /m2v_intern2/zhenglin/my_bashrc.sh
"$H3_PYTHON" -m visual_rl.models.prepare_h3_omninft_data \
  --output-dir "$H3_DATA_ROOT"
source "$H3_DATA_ROOT/data.env"
```

For MPI launches, source this same `data.env` from the attempt's `launch.env`
on every node. Use a fresh output directory when changing dataset bindings;
existing checkpoints retain their original config and data identity.

The helper pins and verifies both source SHA256 values, preserves all three
prompt fields exactly, rejects train/test prompt overlap, and writes the two
training-format manifests, original files, `dataset-info.json` and `data.env`.
It never overwrites different existing artifacts. `data.env` binds the existing
`H3_TRAIN_MANIFEST`/`H3_TRAIN_SHA256` and `H3_EVAL_MANIFEST`/`H3_EVAL_SHA256`
variables for launcher compatibility. The shared six-reward Full32 YAML explicitly
binds `${H3_OMNINFT_DATA_ROOT}/train.jsonl` and `test.jsonl` with fixed converted
SHA256 values, so an old eval32 binding cannot silently select a different split.
The test split is held for later evaluation rather than fed to training.

### 8-GPU LoRA profile

Use the prepared H3 environment and asset bindings:

```bash
.venv-h3-cu128/bin/python -m visual_rl.train \
  --config configs/examples/minimax_h3_art_posttrain_lora_8gpu.yaml \
  --nproc-per-node 8
```

### 64-GPU context-parallel profile

The 32-GPU profile uses full-world FSDP32, CP1/TP1. More FSDP ranks alone do not
split a sample's activation memory. The 64-GPU profile uses physical-world
**FSDP64 with logical DP32 × CP2**: adjacent rank pairs `(0,1)`, `(2,3)`, …
cooperate on the same prompt group. Each pair executes a physical transformer
batch of up to eight trajectories, splitting its packed sequence across two GPUs.
Microbatch 8 is per cooperating pair, not eight independent trajectories per
physical GPU. Rank pairs should be placed within the same node.

Each CP worker keeps its query-token shard and gathers complete keys/values
for global attention. RoPE coordinates and packed modality/time indices retain
their original positions. CP peers share prompt ownership, trajectory noise
and timestep selection; only their leader decodes and scores the unique
trajectories. Reward population gathering uses the 32 logical replicas, so CP
does not double-count samples or reward statistics. The complete modality
losses use SUM-backward sequence gathers followed by the physical-world FSDP
average, preserving the original gradient scale and loss denominators.

Full32 now uses update microbatch 4; Full64 uses CP2 with update microbatch 8.
Both keep
P32/K8, the two disjoint K4 updates, six reward routes, reference MSE, optimizer,
checkpoint cadence and disabled evaluation unchanged. **P32 cannot be launched
as 64 independent data replicas**; changing only the node count on the CP1 YAML
is rejected.

After the asset/environment bindings above, execute the following once on each
of eight nodes with `H3_NODE_RANK=0,1,…,7`, through the same fenced site launcher:

```bash
.venv-h3-cu128/bin/python -m visual_rl.train \
  --config configs/examples/minimax_h3_art_posttrain_full_64gpu_cp2.yaml \
  --nproc-per-node 8 --nnodes 8 --node-rank "$H3_NODE_RANK" \
  --master-addr "$H3_MASTER_ADDR" --master-port "$H3_MASTER_PORT" \
  --preparation-run-id "$H3_PREPARATION_RUN_ID"
```

The supplied MPI helpers use the same configuration via
`H3_ACCEPTANCE_CONFIG` and `H3_NNODES=8`; no second launcher or GPU-lease wrapper
is needed. The CP2 implementation currently supports **full-parameter ART
training only**. Resume uses the same physical64/DP32/CP2 topology and restores
model, Adam, reference, RNG and prompt cursor. CP2 evaluation and cross-topology
checkpoint conversion are not implemented: enabling evaluation is rejected
before loading assets. The CP1 offline-evaluation instructions above apply to
CP1 checkpoints, not to a CP2 checkpoint by changing its YAML.

## Archived 768p multi-duration ART recipe

`configs/examples/minimax_h3_art_posttrain_multiduration_32gpu.yaml` is the archived full-parameter
T2AV recipe with P32/K8, two disjoint half-K updates, BF16 computation and FP32
FSDP master parameters. It uses the same fixed model, optimizer and native
video-shift12/audio-shift3 schedule as the full32 example. Rollout and eval both
use the shared 1344×768, 50-boundary/49-transition profile, with CFG1.

The archived post-training recipe rotates **124, 243 and 345 frames** across
completed rollouts (24 FPS: approximately 5.17, 10.13 and 14.38 seconds).
All ranks and every sample of a prompt group use the same duration in a
rollout. The rotation resumes from the completed-rollout counter without
consuming generation RNG. These are equal rollout counts, not equal GPU time
or video-token budgets. The pinned native pipeline aligns frames to `17n+5`
and rejects aligned durations above 15 seconds; its longest accepted length
is therefore 345 frames. Declare the ordered buckets through
`algorithm.options.post_training.duration_frames`. Configurations without
this option retain their fixed length.

Each rollout independently draws two video positions from the first 15
(`[0,14]`) and two audio positions from the first 10 (`[0,9]`), without replacement.
The pairs are chronological and shared across every rank, sample and both
updates. At 50 boundaries the video candidate pool covers sigma 0.967742–1;
the audio pool covers sigma 0.930233–1. Candidate sizes come from
`algorithm.options.post_training.video_pool` and `audio_pool`; they must contain
at least two positions and exclude the final rollout transition. Frozen legacy
configs declaring video4/audio3 retain their original selection sequence.
A private seed/attempt/modality RNG leaves generation RNG untouched. Joint
forwards replay the original paired video/audio states at the union of positions; each output contributes
loss only when that modality selected the position. Each modality averages over
its own two positions before the final equal-weight video/audio mean. The union
can contain two to four positions; it is not a Cartesian product of independently
re-noised states.

The sixth scorer is ImageBind **audio-video cosine**, not its temperature-scaled
logits or its `all` mode. Retain the original five scorer transformations. Center
each of the six components within the complete prompt K8, divide by that
component's sample std (`correction=1`) plus `1e-4`, then route:

- video: VideoAlign +1.5 HPSv3 +DeSync +0.5 ImageBind;
- audio: 0.5 AudioBox +CLAP +DeSync +0.5 ImageBind.

Constant components give zero signal. Statistics are frozen before U2 splitting;
evaluation uses single-sample original scorer values, never K8 normalization.
Video/audio covariance, RMS, corrective mask and feature means remain separate.
The default recipe also adds `algorithm.options.loss.reference_mse_beta: 0.001`.
This is a pretrained-reference **velocity MSE regularizer**, rather than an
exact probabilistic KL: compare current and frozen initial predictions on the
same cached states, paired sigmas and text conditioning. Average valid features
and each modality's two selected positions, then combine video/audio equally.
The term is not sigma weighted or suppressed by the ART corrective mask.
Frozen initial parameters are kept as rank-local CPU shards, briefly swapped
in to cache no-grad reference predictions before any current-policy backward
graph is built. Checkpoints include this immutable reference independently of
current parameters; resumed training must not redefine it as the resumed model.
This adds reference forwards, CPU storage and checkpoint size. Legacy recipes
without the explicit coefficient load or forward no reference.

`full_duration_rewards: true` accompanies the duration buckets. DeSync uses
overlapping native 8-second windows spanning the whole clip; CLAP obtains its
window limit from the loaded processor and uses the same coverage policy.
Average native transformed window scores and raw diagnostics instead of summing
them. Short clips retain their original scorer path. AudioBox's existing
duration-weighted windows and the other scorers' full-span sampling are retained.

Evaluation runs at step 0, every 50 completed rollouts, and the final update,
using the same 32 held-out prompts and seed identities at **each** duration
(96 AV clips per evaluation). Curves and audible media tables are separate:
`eval/frames124/*`, `eval/frames243/*`, and `eval/frames345/*`. Each duration has
its own manifest, summary and HTML gallery. Resume audits all three receipts
and repairs missing buckets independently. Compare post-training improvements
against the step-0 baseline within each duration. Save full training state every
50 optimizer updates, retaining two checkpoints; evaluation media is independent
of checkpoint retention.

The five-scorer ART/NFT/Fast configurations share the new inference defaults;
their reward and optimizer semantics are unchanged.

### Matched routed Flow-GRPO Fast

For the current 480p train-only recipe, use
[`minimax_h3_fast_routed_480p_full_32gpu.yaml`](../configs/examples/minimax_h3_fast_routed_480p_full_32gpu.yaml).
It matches the ART profile's six rewards, OmniNFT data, 864×480/124-frame
ten-transition rollout, video first4 random2/audio first3 random2, P32/K8/U2,
rollout8/update4 and reference MSE0.001. Online evaluation and checkpoint saving
(including at completion) are disabled for this comparison run; metrics remain
on disk and W&B. Stochastic Flow-SDE rollout
and independently clipped modality ratios replace ART's ODE/covariance update.
This profile explicitly selects `sampling.options.transition_kind: sde`, with
Gaussian log probabilities averaged over valid features. At sigma=1, diffusion
uses the corresponding modality's actual second schedule boundary in the
denominator, identically during rollout and replay. Other positions use ODE.
The high-noise positions use the same selector as Flash;
the trajectories differ because Fast injects noise at those positions.
The archived H3 Fast profiles continue to use CPS and its negative-MSE surrogate.

`configs/examples/minimax_h3_fast_routed_posttrain_full_32gpu.yaml` inherits
the archived 768p multi-duration ART recipe. Model, data, canvas, duration rotation,
P32/K8, disjoint half-K U2, physical microbatches, optimizer, reference MSE,
checkpoint retention and three-duration evaluation are identical. The explicit
algorithm variant is `flow_grpo_fast` with `ratio_mode: per_modality`.

It draws the **same** two video positions from the first15 and two audio
positions from the first10 using the shared selection function and seed/attempt
protocol. Each branch receives CPS noise only at its own selected positions;
joint forwards still use the original paired states and sigmas. Extra CPS
innovations use a private seed/attempt/rank RNG, leaving the matched initial
noise generator unchanged. Unselected transitions retain native zero-noise CPS
arithmetic. Evaluation remains the complete deterministic ODE grid.

For each modality, replay the cached next state to obtain H3's native
negative-valid-feature-MSE CPS surrogate, compute its ratio against that
modality's cached rollout surrogate, and apply PPO clipping independently
(`ratio_clip=1e-4`, `advantage_clip=5`). Use the already normalized and weighted
six-reward route directly, without a second global or group std. Average each
branch over its own two selected positions, then combine video/audio equally.
Inactive positions contribute no branch loss. Add the same raw pretrained
velocity MSE with coefficient0.001 independently of PPO clipping. This is a
**routed high-noise Fast variant**; it retains CPS stochastic transitions and
the clipped surrogate objective, rather than replacing them with ART's ODE
rollout or covariance loss. The original shared-ratio, contiguous-window Fast
recipes remain available unchanged.

`av/video_ratio`, `av/audio_ratio` and their clip fractions distinguish the
branches. Existing per-duration reward, reference-MSE, gradient, time and memory
metrics also apply. PPO's initial scalar loss can be near zero for centered
advantages; assess finite updates and component rewards rather than comparing
its absolute loss value with ART's MSE loss. Real full32 memory/throughput
acceptance is a separate runtime check.

### Bind assets and evaluation

The new reward manifest extends the seven original asset entries with the
`imagebind` checkpoint file and `imagebind_source` directory. Source is pinned to
`facebookresearch/ImageBind@53680b02d7e37b19b124fa37bae4b6c98c38f5be`;
the bridge verifies its four model implementation files, uses an isolated import
namespace and never downloads weights. Include the upstream license in a minimal
source export. `environments/minimax-h3-scorers.txt` documents its dependencies.
The ImageBind implementation/weights and derived bridge carry CC-BY-NC-SA-4.0;
see `licenses/ImageBind-CC-BY-NC-SA-4.0.txt` and `THIRD_PARTY_NOTICES.md`.
The existing preparation CLI accepts the full nine-entry paths/provenance JSON:

```bash
.venv-h3-cu128/bin/python -m visual_rl.models.prepare_minimax_h3 \
  --reward-paths /absolute/path/nine-reward-assets.json \
  --reward-provenance /absolute/path/nine-reward-provenance.json \
  --manifest /absolute/path/h3-six-reward-manifest.json
```

For the archived multi-duration configuration only, a manually authored
32-prompt evaluation suite is provided at
`configs/examples/data/minimax_h3_eval32.jsonl`. Its visible sound-producing
scenes cover dialogue, animals, instruments, actions, transport and environments.
It is a fixed qualitative monitoring suite, not a benchmark result. The loader
requires exactly32 evaluation rows and rejects shared sample IDs or exact prompt
texts with training. Freeze the file/order/hash for a run; each row uses seed
`evaluation.seed + row_index`, with one generated AV sample.

```bash
export H3_EVAL_MANIFEST="$PWD/configs/examples/data/minimax_h3_eval32.jsonl"
export H3_EVAL_SHA256="$(.venv-h3-cu128/bin/python -c \
  'import hashlib,os; print(hashlib.sha256(open(os.environ["H3_EVAL_MANIFEST"],"rb").read()).hexdigest())')"
export H3_OUTPUT_DIR=/absolute/persistent/path/fresh-h3-posttraining-run
# Also bind H3_MODEL, H3_TRAIN_MANIFEST, H3_TRAIN_SHA256 and H3_REWARD_MANIFEST.
.venv-h3-cu128/bin/python -m visual_rl.train \
  --config configs/examples/minimax_h3_art_posttrain_multiduration_32gpu.yaml --validate
```

### Evaluation, checkpoints and AV playback

The recipe trains for400 optimizer updates (normally200 completed rollouts).
In the archived multi-duration recipe, evaluation runs before training, every
**50 completed rollouts** and at the final update. The current 480p training-only
recipe disables all of these events. `evaluation.every_rollouts=50` is a rollout counter, not W&B's internal
row number or an update count; failed/zero-signal skipped attempts do not advance
it. Evaluation always runs the complete sampling schedule, independent of sparse
training positions. It uses current policy weights, fixed prompt seeds and raw
six-component scores.

Every50 optimizer updates, and at the final update, all ranks publish a complete
model/Adam/RNG/data-cursor checkpoint; retain the latest two published snapshots.
At coincident boundaries the checkpoint is saved first. Continuation repairs a
missing/incomplete due evaluation before advancing or returning from a final
checkpoint, without repeating an already verified complete evaluation. Resume
and `--eval-only` use the same FSDP topology and published checkpoint manifest;
these are resume checkpoints rather than standalone Hugging Face exports.

Each evaluation writes an MP4 **with its AAC stereo audio track**, an original
float PCM WAV master, sample metadata, scored manifest and summary. The local
`evaluation/step-*/<id>/gallery.html` has32 players with sound controls, prompt,
seed and scores. The W&B `eval/audio_video` table uses the already muxed MP4 file,
preserving audio instead of re-encoding frame arrays into silent videos. Six
raw component curves remain available under `eval/reward/raw/*`.

W&B media publishing stays in the independent exporter. Upload errors are
reported in its status, leave the local gallery intact and do not fail optimizer
work. Upload may be delayed; local manifests/media are authoritative. Keep the
whole evaluation directory tree together when copying galleries, and enable
sound in the browser player. Media are retained separately from checkpoint
cleanup.

Launch four nodes with the same distributed command/environment described above,
substituting the new config. Site GPU leases and MPI owners remain unchanged.
