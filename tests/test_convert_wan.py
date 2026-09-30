"""CPU preparation tests; only the version-gated test claims pinned integration."""

import hashlib
import json
import os
import shutil
import weakref

import pytest
import torch

from visual_rl.models import convert_wan as convert

diffusers = pytest.importorskip("diffusers")
safetensors = pytest.importorskip("safetensors")
safetensors_torch = pytest.importorskip("safetensors.torch")
load_file, save_file = safetensors_torch.load_file, safetensors_torch.save_file

TINY_CONFIG = dict(
    patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=4,
    in_channels=6, out_channels=2, text_dim=8, freq_dim=8, ffn_dim=16,
    num_layers=1, image_dim=8, added_kv_proj_dim=8,
)


def original_name(name):
    # Independent fixture construction, NOT a production replacement converter.
    replacements = {
        "condition_embedder.time_embedder.linear_1": "time_embedding.0",
        "condition_embedder.time_embedder.linear_2": "time_embedding.2",
        "condition_embedder.text_embedder.linear_1": "text_embedding.0",
        "condition_embedder.text_embedder.linear_2": "text_embedding.2",
        "condition_embedder.time_proj": "time_projection.1",
        "condition_embedder.image_embedder.norm1": "img_emb.proj.0",
        "condition_embedder.image_embedder.ff.net.0.proj": "img_emb.proj.1",
        "condition_embedder.image_embedder.ff.net.2": "img_emb.proj.3",
        "condition_embedder.image_embedder.norm2": "img_emb.proj.4",
        "attn1": "self_attn", "attn2": "cross_attn",
        ".to_out.0.": ".o.", ".to_q.": ".q.", ".to_k.": ".k.", ".to_v.": ".v.",
        ".add_k_proj.": ".k_img.", ".add_v_proj.": ".v_img.",
        ".norm_added_k.": ".norm_k_img.",
        "ffn.net.0.proj": "ffn.0", "ffn.net.2": "ffn.2",
        "scale_shift_table": "modulation", "proj_out": "head.head",
    }
    for new, old in replacements.items():
        name = name.replace(new, old)
    if name == "modulation":
        name = "head.modulation"
    name = name.replace("norm2", "norm__tmp").replace("norm3", "norm2")
    return name.replace("norm__tmp", "norm3")


@pytest.fixture
def native_shapes():
    return convert._expected_shapes(diffusers.WanTransformer3DModel, TINY_CONFIG)


@pytest.fixture
def runtime(monkeypatch, native_shapes):
    if diffusers.__version__ == convert.DIFFUSERS_VERSION:
        return convert._native_converter()[1]
    # The pinned converter is unavailable in this environment. Exercise storage
    # with a clearly identified name-map double and the REAL installed Wan model.
    names = {original_name(name): name for name in native_shapes}
    names.update({"blocks.0.norm2.weight": "blocks.0.norm3.weight"})

    def fallback_name_map(checkpoint):
        return {names.get(key.removeprefix("model.diffusion_model."),
                          key.removeprefix("model.diffusion_model.")): value
                for key, value in checkpoint.items()}

    monkeypatch.setattr(convert, "_native_converter",
                        lambda: (diffusers.WanTransformer3DModel, fallback_name_map))
    return fallback_name_map


@pytest.fixture
def inputs(tmp_path, native_shapes):
    source, config = tmp_path / "source.safetensors", tmp_path / "original.json"
    tensors = {}
    dtypes = (torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.int64)
    for i, (name, shape) in enumerate(sorted(native_shapes.items())):
        value = (torch.arange(torch.Size(shape).numel()) % 19).reshape(shape).to(dtypes[i % 5])
        if value.is_floating_point() and value.numel() >= 3:
            value.flatten()[:3] = torch.tensor([-0.0, float("inf"), float("nan")])
        tensors["model.diffusion_model." + original_name(name)] = value
    save_file(tensors, str(source))
    config.write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    return source, config, tensors


def run_conversion(inputs, output, **kwargs):
    source, config, _ = inputs
    return convert.convert_checkpoint(source, config, output,
                                      expected_source_sha256=convert.file_sha256(source), **kwargs)


def test_tiny_native_meta_architecture(native_shapes):
    assert native_shapes["blocks.0.norm2.weight"] == (8,)
    assert native_shapes["patch_embedding.weight"] == (8, 6, 1, 2, 2)
    assert len(native_shapes) > 40


@pytest.mark.parametrize("shard_bytes", [1, 512, 10**9])
def test_bitwise_preservation_shards_receipt_and_relocation(tmp_path, inputs, runtime,
                                                           shard_bytes):
    output = tmp_path / "transformer"
    receipt = run_conversion(inputs, output, shard_bytes=shard_bytes)
    assert output.is_dir() and output.is_symlink()
    assert not os.path.isabs(os.readlink(output))
    assert not (output / "INCOMPLETE").exists()
    assert (output / "config.json").read_bytes() == inputs[1].read_bytes()
    assert json.loads((output / convert.RECEIPT_NAME).read_text()) == receipt
    assert receipt["source"] == {"sha256": convert.file_sha256(inputs[0]),
                                 "size": inputs[0].stat().st_size}
    assert receipt["config"]["sha256"] == convert.file_sha256(inputs[1])
    assert receipt["value_transform"] == receipt["dtype_transform"] == "none"
    assert str(tmp_path) not in json.dumps(receipt)
    assert receipt["converter"]["diffusers"] == "0.33.1"
    assert receipt["converter"]["implementation_sha256"] == convert.file_sha256(convert.__file__)
    index = json.loads((output / convert.INDEX_NAME).read_text())
    mapping = convert.rename_keys(inputs[2], runtime)
    total, seen, oversized = 0, set(), False
    for filename in set(index["weight_map"].values()):
        tensors = load_file(str(output / filename))
        size = sum(v.numel() * v.element_size() for v in tensors.values())
        assert size <= shard_bytes or len(tensors) == 1
        oversized |= size > shard_bytes
        for name, value in tensors.items():
            assert name not in seen
            seen.add(name)
            assert index["weight_map"][name] == filename
            original = inputs[2][mapping[name]]
            assert value.dtype == original.dtype and value.shape == original.shape
            assert torch.equal(value.flatten().view(torch.uint8),
                               original.flatten().view(torch.uint8))
        total += size
    assert seen == set(mapping) == set(index["weight_map"])
    assert total == receipt["tensor_bytes"] == index["metadata"]["total_size"]
    assert len(seen) == receipt["tensor_count"]
    if shard_bytes == 1:
        assert oversized
    assert {row["path"] for row in receipt["files"]} == {
        "config.json", convert.INDEX_NAME, *index["weight_map"].values()}
    for row in receipt["files"]:
        path = output / row["path"]
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    backing = output.resolve()
    shutil.move(str(backing), relocated / backing.name)
    shutil.move(str(output), relocated / output.name)
    assert (relocated / output.name / convert.RECEIPT_NAME).is_file()


def test_hash_checked_before_converter_or_staging(tmp_path, inputs, monkeypatch):
    def forbidden():
        pytest.fail("converter must not load on SHA256 mismatch")

    monkeypatch.setattr(convert, "_native_converter", forbidden)
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        convert.convert_checkpoint(inputs[0], inputs[1], tmp_path / "out",
                                   expected_source_sha256="0" * 64)
    assert not list(tmp_path.glob(".out.staging-*"))


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_shard_budget(tmp_path, inputs, size):
    with pytest.raises(ValueError, match="positive integer"):
        run_conversion(inputs, tmp_path / "out", shard_bytes=size)


@pytest.mark.parametrize("digest", [None, "", "f" * 63, "z" * 64])
def test_invalid_digest(tmp_path, inputs, digest):
    with pytest.raises(ValueError, match="64-digit"):
        convert.convert_checkpoint(inputs[0], inputs[1], tmp_path / "out",
                                   expected_source_sha256=digest)


@pytest.mark.parametrize("kind", ["directory", "file", "dangling-link"])
def test_existing_destination_is_never_changed(tmp_path, inputs, kind):
    output = tmp_path / "out"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_bytes(b"keep")
    else:
        output.symlink_to("absent")
    before = output.lstat()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_conversion(inputs, output)
    assert output.lstat() == before
    assert not list(tmp_path.glob(".out.staging-*"))


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape"])
def test_complete_architecture_validation_before_tensor_load(tmp_path, inputs, runtime,
                                                           monkeypatch, corruption):
    source, _, tensors = inputs
    if corruption == "missing":
        tensors.pop(next(iter(tensors)))
    elif corruption == "extra":
        tensors["unexpected.weight"] = torch.zeros(1)
    else:
        tensors[sorted(tensors)[-1]] = torch.zeros(1)
    save_file(tensors, str(source))
    real_open = safetensors.safe_open

    class MetadataOnly:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"
            self.reader = real_open(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            pytest.fail("all keys/shapes must be validated before tensor loading")

    monkeypatch.setattr(safetensors, "safe_open", MetadataOnly)
    with pytest.raises(ValueError, match="mismatch"):
        run_conversion(inputs, tmp_path / "out")
    stage, = tmp_path.glob(".out.staging-*")
    assert (stage / "INCOMPLETE").exists()
    assert not list(stage.glob("*.safetensors"))
    assert not os.path.lexists(tmp_path / "out")


def test_writer_failure_preserves_partial_staging(tmp_path, inputs, runtime, monkeypatch, capsys):
    writes = []

    def failing_write(tensors, filename, **kwargs):
        if writes:
            raise OSError("disk full")
        save_file(tensors, filename, **kwargs)
        writes.append(filename)

    monkeypatch.setattr("safetensors.torch.save_file", failing_write)
    with pytest.raises(OSError, match="disk full"):
        run_conversion(inputs, tmp_path / "out", shard_bytes=1)
    stage, = tmp_path.glob(".out.staging-*")
    assert (stage / "INCOMPLETE").exists() and len(writes) == 1
    assert not (stage / convert.RECEIPT_NAME).exists()
    assert not os.path.lexists(tmp_path / "out")
    assert str(stage) in capsys.readouterr().err


def test_live_tensor_references_bounded_to_shard_plus_incoming(tmp_path, inputs, runtime,
                                                            monkeypatch):
    real_open, refs, budget = safetensors.safe_open, [], 512

    class TrackingReader:
        def __init__(self, *args, **kwargs):
            assert kwargs["device"] == "cpu"
            self.reader = real_open(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            live_bytes = sum(size for ref, size in refs if ref() is not None)
            assert live_bytes <= budget  # Oversized tensors must flush immediately.
            value = self.reader.get_tensor(name)
            refs.append((weakref.ref(value), value.numel() * value.element_size()))
            return value

    monkeypatch.setattr(safetensors, "safe_open", TrackingReader)
    run_conversion(inputs, tmp_path / "out", shard_bytes=budget)
    assert len(refs) == len(inputs[2])
    assert all(ref() is None for ref, _ in refs)


@pytest.mark.parametrize("kind", ["directory", "file", "dangling-link"])
def test_publication_race_never_overwrites(tmp_path, inputs, runtime, monkeypatch, kind):
    symlink = os.symlink

    def racing_publish(source, target, **kwargs):
        if kind == "directory":
            target.mkdir()
        elif kind == "file":
            target.write_bytes(b"other writer")
        else:
            symlink("absent", target)
        return symlink(source, target, **kwargs)

    monkeypatch.setattr(convert.os, "symlink", racing_publish)
    output = tmp_path / "out"
    with pytest.raises(FileExistsError):
        run_conversion(inputs, output)
    if kind == "directory":
        assert output.is_dir() and not list(output.iterdir())
    elif kind == "file":
        assert output.read_bytes() == b"other writer"
    else:
        assert os.readlink(output) == "absent"
    stage, = tmp_path.glob(".out.staging-*")
    assert (stage / "INCOMPLETE").exists()
    assert (stage / convert.RECEIPT_NAME).exists()


def test_source_mutation_prevents_publication(tmp_path, inputs, runtime, monkeypatch):
    def mutate_source(tensors, filename, **kwargs):
        save_file(tensors, filename, **kwargs)
        with inputs[0].open("ab") as stream:
            stream.write(b"changed")

    monkeypatch.setattr("safetensors.torch.save_file", mutate_source)
    with pytest.raises(ValueError, match="source changed"):
        run_conversion(inputs, tmp_path / "out")
    assert not os.path.lexists(tmp_path / "out")


def test_cli_required_hash_and_receipt(tmp_path, inputs, runtime, capsys):
    with pytest.raises(SystemExit) as error:
        convert.main(["--source", str(inputs[0]), "--original-config", str(inputs[1]),
                      "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2
    capsys.readouterr()
    convert.main(["--source", str(inputs[0]), "--original-config", str(inputs[1]),
                  "--output-dir", str(tmp_path / "out"), "--shard-bytes", "512",
                  "--expected-source-sha256", convert.file_sha256(inputs[0]).upper()])
    assert json.loads(capsys.readouterr().out)["shard_bytes"] == 512


def test_installed_version_gate():
    if diffusers.__version__ == convert.DIFFUSERS_VERSION:
        assert callable(convert._native_converter()[1])
    else:
        with pytest.raises(RuntimeError, match="requires diffusers==0.33.1"):
            convert._native_converter()


@pytest.mark.skipif(diffusers.__version__ != "0.33.1",
                    reason="genuine pinned Diffusers 0.33.1 converter unavailable")
def test_genuine_pinned_norm_swaps_collisions_and_load(tmp_path, inputs):
    _, official = convert._native_converter()
    mapping = convert.rename_keys(["blocks.0.norm2.weight", "blocks.0.norm3.weight",
                                  "img_emb.proj.4.weight"], official)
    assert mapping == {"blocks.0.norm3.weight": "blocks.0.norm2.weight",
                       "blocks.0.norm2.weight": "blocks.0.norm3.weight",
                       "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight"}
    with pytest.raises(ValueError, match="lossy/colliding"):
        convert.rename_keys(["head.head.weight", "proj_out.weight"], official)
    source, config, tensors = inputs
    tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
    save_file(tensors, str(source))
    run_conversion((source, config, tensors), tmp_path / "out", shard_bytes=512)
    model = diffusers.WanTransformer3DModel.from_pretrained(
        tmp_path / "out", local_files_only=True, torch_dtype=torch.float32)
    for name, value in model.state_dict().items():
        original = tensors["model.diffusion_model." + original_name(name)]
        # Native loading has its own dtype policy; stored bytes are tested above.
        torch.testing.assert_close(value, original.float(), rtol=0, atol=0, equal_nan=True)
