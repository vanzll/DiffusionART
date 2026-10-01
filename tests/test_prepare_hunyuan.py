import json
from pathlib import Path

import pytest

from visual_rl.models.prepare_hunyuan import prepare_text_encoder_view


def assets(tmp_path):
    source = tmp_path / "download"
    files = {
        "text_encoder": ("config.json", "model-00001-of-00004.safetensors"),
        "text_encoder_2": ("config.json", "model.safetensors"),
        "tokenizer": ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"),
        "tokenizer_2": ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"),
    }
    for directory, names in files.items():
        (source / directory).mkdir(parents=True)
        for name in names:
            (source / directory / name).write_text(json.dumps({"identity": f"{directory}/{name}"}))
    return source


def test_native_text_encoder_view_preserves_assets(tmp_path):
    source = assets(tmp_path)
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "native-text"
    receipt = prepare_text_encoder_view(source, output)
    assert output.is_symlink()
    assert receipt["weights_copied"] is False
    for encoder, files in receipt["files"].items():
        for name, target in files.items():
            assert (output / encoder / name).is_symlink()
            assert (output / encoder / name).resolve() == Path(target)
            assert (output / encoder / name).read_bytes() == Path(target).read_bytes()
    assert json.loads((output / "asset-view.json").read_text()) == receipt
    assert before == {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert not (source / "text_encoder" / "tokenizer.json").exists()
    assert (output / "text_encoder" / "tokenizer.json").is_file()
    assert (output / "text_encoder_2" / "vocab.json").is_file()


@pytest.mark.parametrize("existing", ["directory", "file", "dangling-link"])
def test_existing_view_is_never_replaced(tmp_path, existing):
    source = assets(tmp_path)
    output = tmp_path / "native-text"
    if existing == "directory":
        output.mkdir()
    elif existing == "file":
        output.write_text("existing")
    else:
        output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        prepare_text_encoder_view(source, output)
    assert not list(tmp_path.glob(".native-text.staging-*"))


@pytest.mark.parametrize("problem", ["missing-tokenizer", "missing-encoder", "conflict", "broken-link"])
def test_invalid_layout_fails_before_publication(tmp_path, problem):
    source = assets(tmp_path)
    if problem == "missing-tokenizer":
        (source / "tokenizer_2" / "vocab.json").unlink()
    elif problem == "missing-encoder":
        (source / "text_encoder" / "config.json").unlink()
    elif problem == "conflict":
        (source / "tokenizer" / "config.json").write_text("wrong")
    else:
        (source / "text_encoder" / "broken").symlink_to(source / "missing")
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_text_encoder_view(source, tmp_path / "native-text")
    assert not list(tmp_path.glob(".native-text.staging-*"))


def test_publication_failure_does_not_replace_output(tmp_path, monkeypatch, capsys):
    source = assets(tmp_path)
    output = tmp_path / "native-text"
    original = Path.symlink_to

    def collision(path, target, **kwargs):
        if path == output:
            output.mkdir()
        return original(path, target, **kwargs)

    monkeypatch.setattr(Path, "symlink_to", collision)
    with pytest.raises(FileExistsError):
        prepare_text_encoder_view(source, output)
    assert not output.is_symlink() and list(output.iterdir()) == []
    assert "Unpublished asset view preserved" in capsys.readouterr().err
    assert len(list(tmp_path.glob(".native-text.staging-*"))) == 1


@pytest.mark.parametrize("use_alias", [False, True])
def test_nested_output_cannot_modify_source_tree(tmp_path, use_alias):
    source = assets(tmp_path)
    before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))
    parent = source
    if use_alias:
        parent = tmp_path / "alias"
        parent.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the source"):
        prepare_text_encoder_view(source, parent / "native-text")
    assert before == sorted(str(path.relative_to(source)) for path in source.rglob("*"))
