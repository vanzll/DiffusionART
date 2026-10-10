"""Native distilled Wan I2V, sharing the public video training lifecycle.

Online FLUX references are generated inside the measured training phase.
Fixed training references require an explicit fixed_images opt-in.
Evaluation uses fixed, hashed reference images, never regenerated references.
"""

import json
import sys
from dataclasses import dataclass
from functools import partial

import torch
from PIL import Image

from ..config import ConfigError
from ..data import PromptDataset
from ..models.hunyuan_assets import load_reward
from ..models.wan import WanI2VConditioning
from ..models.wan_assets import load_fixed_reference, reference_mode
from ..training import wan_video
from . import hunyuan

ASSET_KEYS = ("base_checkpoint", "base_revision", "reference_checkpoint", "reference_revision")
_shape = hunyuan._shape


def validate_config(c):
    try:
        fixed = reference_mode(c.model.options) == "fixed_images"
    except ValueError as error:
        raise ConfigError(str(error)) from error
    required = ASSET_KEYS[:2] if fixed else ASSET_KEYS
    extra = list(required)
    if fixed:
        extra.extend(key for key in ASSET_KEYS[2:] if key in c.model.options)
    if "reference_mode" in c.model.options:
        extra.append("reference_mode")
    hunyuan.validate_config(c, model_name="wan_i2v_distilled", task="i2v", extra_model_keys=extra)
    if c.algorithm.name not in wan_video.ALGORITHMS:
        raise ConfigError("choose ART, NFT, Fast-SDE, Dance or final-step ReFL")
    if any(not isinstance(c.model.options[key], str) or not c.model.options[key].strip() for key in required):
        raise ConfigError("explicit Wan component and FLUX reference model identities required")
    if fixed:
        for key in ASSET_KEYS[2:]:
            value = c.model.options.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError("fixed Wan reference identities must be nonempty strings or null")
    if (c.sampling.steps != 4 or c.sampling.options["shift"] != 5
            or c.evaluation.sampling_steps not in (None, 4) or c.evaluation.shift not in (None, 5)):
        raise ConfigError("distilled Wan requires four steps, shift five, CFG off in train and eval")
    if ((c.evaluation.height or c.sampling.height) != c.sampling.height
            or (c.evaluation.width or c.sampling.width) != c.sampling.width):
        raise ConfigError("Wan reference conditioning currently requires equal train/eval geometry")
    pool = [3] if c.algorithm.name == "refl" else list(range(4 if c.algorithm.name == "diffusionnft" else 3))
    count = 1 if c.algorithm.name == "flow_grpo_fast" else len(pool)
    if (c.sampling.options["train_timestep_pool"] != pool
            or c.sampling.options["train_timestep_count"] != count):
        raise ConfigError("Wan uses all three ART/Dance positions, all four NFT positions, or one shared Fast position")
    if c.algorithm.name in ("flow_grpo_fast", "dancegrpo"):
        expected = .8 if c.algorithm.name == "flow_grpo_fast" else .25
        if c.sampling.options["noise_level"] != expected:
            raise ConfigError("distilled Wan noise level differs from the accepted sampler")


@dataclass
class Components:
    adapter: object
    scorer: object
    pipe: object
    flux: object
    refl_decode: object = None


def load_components(c, device):
    from ..models.wan_assets import load_models

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = c.model.options["allow_tf32"]
    adapter, pipe, flux = load_models(c, device)
    parts = Components(adapter, load_reward(c), pipe, flux)
    if c.algorithm.name == "refl":
        from ..models.wan_refl import configure_wan_vae, decode_wan_checkpointed
        from ..training.refl_vjp import configure_video_reward

        configure_wan_vae(adapter.vae)
        configure_video_reward(parts.scorer)
        parts.refl_decode = partial(decode_wan_checkpointed, adapter.vae)
    return parts


def load_datasets(c):
    fixed = reference_mode(c.model.options) == "fixed_images"
    data = PromptDataset.load(c.data.manifest, expected_sha256=c.data.sha256, task="i2v" if fixed else "t2v")
    if fixed:
        # The generic I2V loader also accepts unpinned images; fixed training does not.
        with open(c.data.manifest, "rb") as stream:
            for row in data.rows:
                record = json.loads(stream.readline(1024 * 1024 + 1))
                if record.get("image_sha256") != row.image_sha256:
                    raise ValueError("Wan fixed training requires explicit image_sha256 for every row")
                load_fixed_reference(row.image, row.image_sha256)
    validation = PromptDataset.load(c.evaluation.manifest, expected_sha256=c.evaluation.manifest_sha256,
                                    task="i2v") if c.evaluation.enabled else None
    return data, validation


def _predict(c, parts, rows, manifest, repeats, device, *, attempt=0, rank=0, evaluation=False):
    from ..models.wan_assets import condition_image, generate_reference

    fixed = reference_mode(c.model.options) == "fixed_images"
    conditions = []
    for index, row in enumerate(rows):
        stream = "flux" if index == 0 else f"flux:{index}"
        if evaluation:
            if row.image is None:
                raise ValueError("Wan evaluation requires a fixed reference image")
            with Image.open(row.image) as source:
                image = source.convert("RGB")
        elif fixed:
            image = load_fixed_reference(row.image, row.image_sha256)
        else:
            image = generate_reference(parts.flux, row.text, height=c.sampling.height, width=c.sampling.width,
                device=device, seed=wan_video.scoped_seed(c.seed, rank, attempt, stream))
        condition = condition_image(parts.pipe, row.text, image, height=c.sampling.height,
            width=c.sampling.width, frames=c.sampling.frames, device=device,
            seed=wan_video.scoped_seed(c.seed, rank, attempt, "image_vae" if index == 0 else f"image_vae:{index}"),
            cfg=False)
        conditions.append(condition)
    names = ("prompt_embeds", "image_embeds", "image_condition")
    tensors = {name: torch.cat([getattr(value, name) for value in conditions]).repeat_interleave(repeats, 0)
               for name in names}
    def predict(state, times, indices):
        selected = WanI2VConditioning(**{key: value[indices.cpu()].to(device) for key, value in tensors.items()})
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=c.training.precision == "bf16"):
            return parts.adapter.predict_velocity(state, times, selected).float()
    return predict


def training_inputs(c, attempt, rank=0):
    return wan_video.inputs(algorithm=c.algorithm.name, seed=c.seed, rank=rank, attempt=attempt,
        prompts_per_rank=c.sampling.prompts_per_rollout, k=c.sampling.samples_per_prompt,
        latent_shape=_shape(c, 1)[1:])


def collect_training(c, parts, predict, plan, sigmas, device):
    parts.adapter.transformer.eval()
    return wan_video.collect(c.algorithm.name, predict, plan, sigmas, microbatch=c.sampling.microbatch, device=device)


def prepare_objective(c, parts, cache, predict, scores, global_scores, device):
    if c.algorithm.name in ("flow_grpo_fast", "dancegrpo"):
        result = wan_video.prepare_grpo(c, cache, predict, scores, global_scores, device)
    else:
        result = hunyuan.prepare_objective(c, parts, cache, predict, scores, global_scores, device)
    parts.adapter.transformer.eval()
    return result


@torch.no_grad()
def evaluation_endpoint(predict, noise, sigmas):
    state = noise.float()
    rows = torch.arange(len(noise), device=noise.device)
    for step in range(len(sigmas) - 1):
        velocity = predict(state, sigmas[step].to(noise.device).expand(len(noise)), rows).float()
        state = state + (sigmas[step + 1] - sigmas[step]).to(noise.device) * velocity
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite Wan evaluation endpoint")
    return state.detach().cpu()


def score_videos(c, parts, *args, **kwargs):
    return hunyuan.score_videos(c, parts, *args, **kwargs, decode_bf16=False)


def run(c, *, resume=None, resume_rewind=False, evaluate_only=False):
    return hunyuan.run(c, resume=resume, backend=sys.modules[__name__],
                       resume_rewind=resume_rewind, evaluate_only=evaluate_only)
