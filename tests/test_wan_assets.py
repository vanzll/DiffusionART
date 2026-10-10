"""No downloaded weights: native CPU VAE oracle plus explicit loader/ABI spies."""

import hashlib
import random
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visual_rl.models import wan_assets as assets
from visual_rl.models.wan import WanI2VAdapter, WanI2VConditioning, first_frame_condition

diffusers = pytest.importorskip("diffusers")
Image = pytest.importorskip("PIL.Image")


class Weight(torch.nn.Module):
    def __init__(self, name, events, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        self.name, self.events, self.active = name, events, False
        self.fail_move = False

    def to(self, device, *args, **kwargs):
        self.active = isinstance(device, torch.device)
        self.events.append((self.name, "stage" if self.active else "offload"))
        if self.active and self.fail_move:
            raise RuntimeError("transfer failure")
        # CPU spies record the requested placement without pretending to run CUDA.
        return super().to("cpu", *args, **kwargs)


def consume_rng():
    return random.random(), np.random.random(), torch.rand(3)


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_unchanged(before):
    after = rng_snapshot()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


class PipeSpy:
    def __init__(self, fail=None):
        self.events, self.calls, self.fail = [], {}, fail
        self.text_encoder = Weight("text", self.events, torch.bfloat16)
        self.image_encoder = Weight("clip", self.events)
        self.vae = Weight("vae", self.events)
        self.vae.config = SimpleNamespace(
            latents_mean=[0.1, -0.2], latents_std=[2.0, 3.0], z_dim=2,
            temperal_downsample=[True], scale_factor_temporal=2, patch_size=None,
        )
        self.vae_scale_factor_temporal = self.vae_scale_factor_spatial = 2
        self.config = SimpleNamespace(expand_timesteps=False, boundary_ratio=None)
        self.transformer = self.transformer_2 = None
        self.video_processor = SimpleNamespace(preprocess=self.preprocess)

    def stage(self, name):
        active = [m.name for m in (self.text_encoder, self.image_encoder, self.vae) if m.active]
        assert active == [name]
        for m in (self.text_encoder, self.image_encoder, self.vae):
            assert not m.training and not m.weight.requires_grad and m.weight.grad is None
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")
        consume_rng()
        if self.fail == name:
            raise RuntimeError(f"{name} failure")

    def encode_prompt(self, **kwargs):
        self.stage("text")
        self.calls["text"] = kwargs
        positive = torch.ones(1, 512, 4)
        return positive, -positive if kwargs["do_classifier_free_guidance"] else None

    def encode_image(self, image, *, device):
        self.stage("clip")
        self.calls["clip"] = (image, device)
        return torch.ones(1, 5, 3)

    def preprocess(self, image, *, height, width):
        assert not any(m.active for m in (self.text_encoder, self.image_encoder, self.vae))
        self.calls["preprocess"] = (image, height, width)
        return torch.full((1, 3, height, width), 0.25)

    def prepare_latents(self, **kwargs):
        self.stage("vae")
        self.calls["vae"] = kwargs
        assert torch.count_nonzero(kwargs["latents"]) == 0
        assert "generator" not in kwargs
        shape = kwargs["latents"].shape
        encoded = torch.full(shape, 0.7)
        return kwargs["latents"], first_frame_condition(encoded, 2)


def call_condition(pipe, **overrides):
    kwargs = dict(height=8, width=8, frames=5, device="cpu", seed=123)
    kwargs.update(overrides)
    return assets.condition_image(pipe, "A moving scene", Image.new("RGB", (8, 8)), **kwargs)


def test_scoped_seed_canonical_little_endian_and_separate_streams():
    digest = hashlib.sha256(b"42:3:17:flux").digest()
    assert assets.scoped_seed(42, 3, 17, "flux") == int.from_bytes(digest[:8], "little") % (2**63 - 1)
    assert len({assets.scoped_seed(s, r, a, stream) for s, r, a, stream in (
        (42, 3, 17, "flux"), (43, 3, 17, "flux"), (42, 4, 17, "flux"),
        (42, 3, 18, "flux"), (42, 3, 17, "image_vae"),
    )}) == 5
    for args in ((True, 0, 0, "x"), (0, -1, 0, "x"), (0, 0, True, "x"), (0, 0, 0, "")):
        with pytest.raises(ValueError):
            assets.scoped_seed(*args)


@pytest.mark.parametrize("fail", [False, True])
def test_rng_restores_all_cpu_streams_even_on_exception(fail):
    before = rng_snapshot()
    try:
        with assets.isolated_rng(987, "cpu"):
            first = consume_rng()
            if fail:
                raise RuntimeError("expected")
    except RuntimeError:
        pass
    assert_rng_unchanged(before)
    with assets.isolated_rng(987, "cpu"):
        again = consume_rng()
    assert first[:2] == again[:2]
    assert torch.equal(first[2], again[2])
    assert_rng_unchanged(before)


@pytest.mark.parametrize("cfg", [False, True])
def test_condition_native_abi_cpu_outputs_precision_and_sequential_staging(cfg):
    pipe = PipeSpy()
    before = rng_snapshot()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = call_condition(pipe, cfg=cfg)
    assert_rng_unchanged(before)
    assert isinstance(result, WanI2VConditioning)
    text = pipe.calls["text"]
    assert text == dict(prompt="A moving scene", negative_prompt="",
                        do_classifier_free_guidance=cfg, num_videos_per_prompt=1,
                        max_sequence_length=512, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert result.prompt_embeds.dtype == result.image_embeds.dtype == torch.bfloat16
    assert (result.negative_prompt_embeds is not None) == cfg
    assert result.image_condition.dtype == torch.float32
    assert result.image_condition.shape == (1, 4, 3, 4, 4)
    for value in (result.prompt_embeds, result.image_embeds, result.image_condition):
        assert value.device.type == "cpu" and not value.requires_grad
    assert pipe.events == [(name, "offload") for name in ("text", "clip", "vae")] + [
        (name, event) for name in ("text", "clip", "vae") for event in ("stage", "offload")]
    vae = pipe.calls["vae"]
    assert vae["batch_size"] == 1 and vae["num_channels_latents"] == 2
    assert vae["num_frames"] == 5 and vae["dtype"] == torch.float32
    assert torch.all(vae["image"] == 0.25)


@pytest.mark.parametrize("stage", ["text", "clip", "vae"])
def test_condition_failures_offload_frozen_weights_and_restore_rng(stage):
    pipe, before = PipeSpy(fail=stage), rng_snapshot()
    with pytest.raises(RuntimeError, match=f"{stage} failure"):
        call_condition(pipe)
    assert_rng_unchanged(before)
    assert all(not m.active and not m.weight.requires_grad
               for m in (pipe.text_encoder, pipe.image_encoder, pipe.vae))


def test_staging_transfer_failure_cleans_all_pipeline_modules():
    events = []
    first, second, third = [Weight(name, events) for name in ("first", "second", "third")]
    second.fail_move = True
    with pytest.raises(RuntimeError, match="transfer failure"):
        pipe = SimpleNamespace(components={"one": first, "two": second, "three": third})
        with assets.staged(pipe, "cpu"):
            pytest.fail("must not enter")
    assert events[-3:] == [(name, "offload") for name in ("first", "second", "third")]
    assert all(not m.active and not m.training and not m.weight.requires_grad
               for m in (first, second, third))


def test_staging_continues_cleanup_and_reports_uncertain_residency(monkeypatch):
    events = []
    first, second = Weight("first", events), Weight("second", events)
    original = first.to

    def fail_offload(device):
        if device == "cpu":
            raise RuntimeError("cannot copy")
        return original(device)

    monkeypatch.setattr(first, "to", fail_offload)
    with pytest.raises(RuntimeError, match="residency is not guaranteed"):
        with assets.staged(SimpleNamespace(components={"one": first, "two": second}), "cpu"):
            pass
    assert not second.active


@pytest.mark.parametrize("field,value", [
    ("expand_timesteps", True), ("boundary_ratio", 0.9),
])
def test_condition_rejects_wan22_without_model_calls(field, value):
    pipe = PipeSpy()
    setattr(pipe.config, field, value)
    with pytest.raises(ValueError, match="Wan2.1"):
        call_condition(pipe)
    assert not pipe.calls and not pipe.events


@pytest.mark.parametrize("kwargs", [{"frames": 4}, {"height": 7}, {"width": 6}, {"cfg": 1}])
def test_condition_rejects_rounding_or_implicit_cfg(kwargs):
    with pytest.raises(ValueError):
        call_condition(PipeSpy(), **kwargs)


def test_condition_rejects_wrong_dtype_hooks_and_bad_native_mask():
    pipe = PipeSpy()
    pipe.image_encoder.bfloat16()
    with pytest.raises(ValueError, match="weights must"):
        call_condition(pipe)
    pipe = PipeSpy()
    pipe.vae._hf_hook = object()
    with pytest.raises(ValueError, match="offload hooks"):
        call_condition(pipe)
    pipe = PipeSpy()
    original = pipe.prepare_latents

    def corrupt(**kwargs):
        noise, condition = original(**kwargs)
        condition[:, :2, -1] = 1
        return noise, condition

    pipe.prepare_latents = corrupt
    with pytest.raises(ValueError, match="first-frame-only"):
        call_condition(pipe)
    assert not pipe.vae.active


def test_condition_rejects_group_offload_hooks_and_unknown_native_abi(monkeypatch):
    pipe = PipeSpy()
    pipe.vae._diffusers_hook = object()
    with pytest.raises(ValueError, match="offload hooks"):
        call_condition(pipe)
    monkeypatch.setattr(diffusers, "__version__", "0.41.0")
    with pytest.raises(RuntimeError, match="unsupported Wan conditioning ABI"):
        call_condition(PipeSpy())


class FluxSpy:
    def __init__(self, fail=False):
        self.events, self.calls, self.fail = [], [], fail
        self.components = {name: Weight(name, self.events) for name in ("flux", "text", "vae")}
        array = np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3)
        self.image = Image.fromarray(array)

    def __call__(self, prompt, **kwargs):
        assert all(m.active and not m.weight.requires_grad for m in self.components.values())
        self.calls.append((prompt, kwargs))
        consume_rng()
        if self.fail:
            raise RuntimeError("online generation failed")
        return SimpleNamespace(images=[self.image])


@pytest.mark.parametrize("fail", [False, True])
def test_reference_online_flux_abi_jpeg_roundtrip_and_rng_isolation(fail):
    flux, before = FluxSpy(fail), rng_snapshot()
    kwargs = dict(height=16, width=32, device="cpu", seed=456)
    if fail:
        with pytest.raises(RuntimeError, match="online generation failed"):
            assets.generate_reference(flux, "Scene", **kwargs)
    else:
        result = assets.generate_reference(flux, "Scene", **kwargs)
        with BytesIO() as buffer:
            flux.image.save(buffer, format="JPEG")
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                assert result.tobytes() == decoded.convert("RGB").tobytes()
        assert result.mode == "RGB" and result.size == (32, 16)
        assert result.tobytes() != flux.image.tobytes()
        assets.generate_reference(flux, "Scene", **kwargs)
        assert len(flux.calls) == 2  # no cached reference substitution
    assert_rng_unchanged(before)
    assert all(not m.active for m in flux.components.values())
    prompt, call = flux.calls[0]
    generator = call.pop("generator")
    assert prompt == "Scene" and generator.device.type == "cpu" and generator.initial_seed() == 456
    assert call == dict(height=16, width=32, num_inference_steps=30,
                        guidance_scale=3.5, max_sequence_length=512)


def test_genuine_native_wan_vae_full_zero_pixel_video_oracle(monkeypatch):
    if diffusers.__version__ != "0.40.0":
        pytest.skip("this tiny VAE constructor oracle targets locked diffusers 0.40.0")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with assets.isolated_rng(41, "cpu"):
            vae = diffusers.AutoencoderKLWan(
                base_dim=4, z_dim=2, dim_mult=[1, 2], num_res_blocks=1,
                temperal_downsample=[True], latents_mean=[0.1, -0.2], latents_std=[2.0, 3.0],
                scale_factor_temporal=2, scale_factor_spatial=2,
            ).float().eval()
        pipe = diffusers.WanImageToVideoPipeline(
            tokenizer=None, text_encoder=torch.nn.Linear(2, 2).bfloat16(), vae=vae,
            scheduler=diffusers.FlowMatchEulerDiscreteScheduler(),
            image_encoder=torch.nn.Linear(2, 2), image_processor=None, transformer=None,
        )
        # Actual native preprocessing, pipeline prepare_latents and VAE encoding;
        # text/CLIP need no tokenizer downloads and are separate ABI spies above.
        monkeypatch.setattr(pipe, "encode_prompt", lambda **kw: (torch.ones(1, 512, 4), None))
        monkeypatch.setattr(pipe, "encode_image", lambda image, device: torch.ones(1, 5, 4))
        videos = []
        encode = vae.encode

        def observe(video, *args, **kwargs):
            videos.append(video.detach().clone())
            return encode(video, *args, **kwargs)

        monkeypatch.setattr(vae, "encode", observe)
        import diffusers.pipelines.wan.pipeline_wan_i2v as native

        def no_random_noise(*args, **kwargs):
            pytest.fail("supplied zero noise must bypass native randn_tensor")

        monkeypatch.setattr(native, "randn_tensor", no_random_noise)
        before = rng_snapshot()
        result = call_condition(pipe)
        assert_rng_unchanged(before)
        assert len(videos) == 1 and videos[0].shape == (1, 3, 5, 8, 8)
        assert torch.count_nonzero(videos[0][:, :, 1:]) == 0
        with torch.no_grad():
            raw = encode(videos[0]).latent_dist.mode()
            mean = torch.tensor([0.1, -0.2]).view(1, 2, 1, 1, 1)
            std = torch.tensor([2.0, 3.0]).view(1, 2, 1, 1, 1)
            expected = first_frame_condition((raw - mean) / std, 2)
            single = encode(videos[0][:, :, :1]).latent_dist.mode()
        torch.testing.assert_close(result.image_condition, expected)
        assert torch.count_nonzero(result.image_condition[:, 2:, 1:]) > 0
        wrong = torch.cat([(single - mean) / std, torch.zeros_like(raw[:, :, 1:])], dim=2)
        assert not torch.allclose(result.image_condition[:, 2:], wrong)
        assert all(p.device.type == "cpu" and not p.requires_grad for p in vae.parameters())
    finally:
        torch.set_num_threads(threads)


def config(distributed="none"):
    return SimpleNamespace(
        model=SimpleNamespace(checkpoint="/converted", revision="policy-pin", options={
            "base_checkpoint": "/base", "base_revision": "base-pin",
            "reference_checkpoint": "/flux", "reference_revision": "flux-pin",
            "finetune": {"mode": "full"}, "gradient_checkpointing": True,
        }), runtime=SimpleNamespace(distributed=distributed),
        sampling=SimpleNamespace(guidance_scale=1.0),
    )


def test_loader_rejects_current_runtime_before_any_weight_load(monkeypatch):
    monkeypatch.setattr(diffusers, "__version__", "0.40.0")
    with pytest.raises(RuntimeError, match="require diffusers==0.33.1"):
        assets.load_models(config(), "cpu")
    with pytest.raises(RuntimeError, match="silent ABI upgrade"):
        assets.require_native_version("0.40.0")


@pytest.mark.parametrize("distributed", ["none", "fsdp"])
@pytest.mark.parametrize("mode", [None, "online_flux", "fixed_images"])
def test_pinned_loader_paths_dtypes_policy_preparation_and_shared_vae(monkeypatch, distributed, mode):
    import transformers

    # Loader contract spy, NOT a claim that installed 0.40 executes native 0.33.
    monkeypatch.setattr(diffusers, "__version__", "0.33.1")
    calls, events, patches = [], [], []
    pipe = PipeSpy()
    pipe.set_progress_bar_config = lambda **kwargs: None
    flux = FluxSpy()
    flux.set_progress_bar_config = lambda **kwargs: None
    policy = Weight("policy", events)
    policy.config = SimpleNamespace(in_channels=6, out_channels=2, image_dim=4)
    policy.enable_gradient_checkpointing = lambda: calls.append(("checkpointing",))
    monkeypatch.setattr(assets, "stabilize_wan_time_embedder_dtype_lookup", patches.append)

    def loader(name, value):
        def load(path, **kwargs):
            calls.append((name, path, kwargs))
            if name == "pipe":
                assert kwargs["vae"] is pipe.vae
                assert kwargs["image_encoder"] is pipe.image_encoder
            return value
        return load

    for cls, name, value in (
        (diffusers.WanTransformer3DModel, "policy", policy),
        (diffusers.AutoencoderKLWan, "vae", pipe.vae),
        (transformers.CLIPVisionModel, "clip", pipe.image_encoder),
        (diffusers.WanImageToVideoPipeline, "pipe", pipe),
        (diffusers.FluxPipeline, "flux", flux),
    ):
        monkeypatch.setattr(cls, "from_pretrained", loader(name, value))
    c = config(distributed)
    if mode is not None:
        c.model.options["reference_mode"] = mode
    if mode == "fixed_images":
        del c.model.options["reference_checkpoint"], c.model.options["reference_revision"]
    adapter, got_pipe, got_flux = assets.load_models(c, "cpu")
    assert isinstance(adapter, WanI2VAdapter) and adapter.transformer is policy
    assert got_pipe is pipe and adapter.vae is pipe.vae
    assert got_flux is (None if mode == "fixed_images" else flux)
    assert adapter.config.cfg_scale is None and adapter.config.temporal_compression == 2
    assert policy.weight.requires_grad and not pipe.vae.weight.requires_grad
    assert [c[0] for c in calls].count("vae") == 1
    records = {c[0]: c[1:] for c in calls if len(c) == 3}
    assert records["policy"] == ("/converted", dict(revision="policy-pin", local_files_only=True,
                                                  torch_dtype=torch.float32))
    assert records["vae"] == ("/base", dict(subfolder="vae", revision="base-pin",
                                           local_files_only=True, torch_dtype=torch.float32))
    assert records["clip"][1]["torch_dtype"] == torch.float32
    assert records["pipe"][1]["torch_dtype"] == torch.bfloat16
    assert records["pipe"][1]["transformer"] is None
    assert "transformer_2" not in records["pipe"][1]  # not a 0.33.1 constructor option
    if mode == "fixed_images":
        assert "flux" not in records and not flux.events and not flux.calls
    else:
        assert records["flux"] == ("/flux", dict(revision="flux-pin", local_files_only=True,
                                                torch_dtype=torch.bfloat16))
    assert patches == ([policy] if distributed == "fsdp" else [])
    assert (("checkpointing",) in calls) == (distributed != "fsdp")
    assert bool(events) == (distributed != "fsdp")


@pytest.mark.parametrize("references", ["omitted", "null"])
def test_fixed_loader_never_imports_or_acquires_flux(monkeypatch, references):
    import builtins

    monkeypatch.setattr(diffusers, "__version__", "0.33.1")
    c = config()
    c.model.options["reference_mode"] = "fixed_images"
    for key in ("reference_checkpoint", "reference_revision"):
        if references == "omitted":
            del c.model.options[key]
        else:
            c.model.options[key] = None
    adapter = SimpleNamespace(vae=object())
    calls = []
    pipe = SimpleNamespace(set_progress_bar_config=lambda **kw: calls.append(kw))
    monkeypatch.setattr(assets, "load_adapter", lambda c, device: adapter)

    def components(c, *, vae):
        assert vae is adapter.vae
        return pipe

    monkeypatch.setattr(assets, "load_components", components)
    original = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        assert not (name == "diffusers" and "FluxPipeline" in (fromlist or ()))
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert assets.load_models(c, "cpu") == (adapter, pipe, None)
    assert calls == [{"disable": True}]


def test_loader_rejects_invalid_reference_mode_before_loading(monkeypatch):
    monkeypatch.setattr(diffusers, "__version__", "0.33.1")
    c = config()
    c.model.options["reference_mode"] = "fixed_image"
    monkeypatch.setattr(assets, "load_adapter", lambda *args: pytest.fail("unexpected weight load"))
    with pytest.raises(ValueError, match="reference_mode"):
        assets.load_models(c, "cpu")


def test_fsdp_dtype_fix_bound_instance_and_empty_parameter_iterator(monkeypatch):
    monkeypatch.setattr(diffusers, "__version__", "0.33.1")

    def legacy_forward(self, timestep, text, image=None):
        timestep = self.timesteps_proj(timestep)
        timestep = timestep.to(next(self.time_embedder.parameters()).dtype)
        temb = self.time_embedder(timestep).type_as(text)
        return temb, self.time_proj(self.act_fn(temb)), self.text_embedder(text), (
            self.image_embedder(image) if image is not None else None)

    cls = type("WanTimeTextImageEmbedding", (torch.nn.Module,), {
        "__module__": "diffusers.models.transformers.transformer_wan", "forward": legacy_forward,
    })

    def embedding():
        result = cls()
        result.timesteps_proj = torch.nn.Identity()
        result.time_embedder = torch.nn.Linear(2, 2)
        result.time_proj = torch.nn.Linear(2, 2)
        result.text_embedder = torch.nn.Linear(2, 2)
        result.image_embedder = torch.nn.Linear(2, 2)
        result.act_fn = torch.nn.SiLU()
        return result

    first, second = embedding(), embedding()
    args = (torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2))
    before = first(*args)
    model = SimpleNamespace(condition_embedder=first)
    assert assets.stabilize_wan_time_embedder_dtype_lookup(model)
    assert not assets.stabilize_wan_time_embedder_dtype_lookup(model)
    assert cls.forward is legacy_forward and second.forward.__func__ is legacy_forward
    assert first.forward.__self__ is first
    monkeypatch.setattr(first.time_embedder, "parameters", lambda: iter(()))
    for actual, expected in zip(first(*args), before):
        torch.testing.assert_close(actual, expected)
    assert first(*args[:2])[-1] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this test host")
def test_selected_cuda_rng_is_restored_on_failure():
    before = torch.cuda.get_rng_state(0).clone()
    with pytest.raises(RuntimeError):
        with assets.isolated_rng(88, "cuda:0"):
            torch.rand(2, device="cuda:0")
            raise RuntimeError("failure")
    assert torch.equal(before, torch.cuda.get_rng_state(0))
