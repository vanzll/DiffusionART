# MiniMax-H3 T2AV

This opt-in backend adds text-conditioned audio/video training with ART, NFT
and FlowGRPO Fast. The six recipes are integration examples, not accepted paper
results. CPU tests and upstream runs do not establish real-weight GPU acceptance.
The first version excludes FL2VA, Ref2VA and sequence parallelism. It uses the
guidance-distilled transformer without an unconditional CFG forward.

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
equivalence is not claimed. Full four-rollout/checkpoint stability still has to
be demonstrated for the chosen profile. Deployment evidence is tracked in
[the acceptance PR](https://github.com/vanzll/DiffusionART/pull/4).

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
updates of 128 trajectories, and eight training positions per trajectory.
Only the name, output directory, physical microbatches and deployment notes
differ. A microbatch is an execution chunk, not an additional optimizer update.

| Profile | Launch topology | Rollout microbatch | Update microbatch |
| --- | --- | --- | --- |
| `minimax_h3_art_full_8gpu.yaml` | 1 node x 8 FSDP ranks | 4 | 2 |
| `minimax_h3_art_full_32gpu.yaml` | 4 nodes x 8 FSDP ranks | 8 | 8 |

These are starting values to tune on the actual hardware. The 32-GPU values
match the physical microbatches of the prior Hunyuan FIRST1/FIRST2 experiments;
they do not assert matching model, resolution or total training cost. The
8-GPU profile deliberately permits smaller microbatches. YAML does not reserve
or count GPUs: the launcher determines the world size. Both profiles use
full-world FSDP (`shard_degree=world_size`, `replicate_degree=1`), with no TP/SP;
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
