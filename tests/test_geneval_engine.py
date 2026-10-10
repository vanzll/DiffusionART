"""CPU-only fixtures; opt in to the pinned-source oracle with GENEVAL_COMPARE_UPSTREAM=1."""

import ast
import copy
import importlib.util
import inspect
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "geneval_engine_under_test", ROOT / "src/visual_rl/rewards/geneval_engine.py"
)
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)
build_evaluator = ENGINE.build_evaluator

UPSTREAM = (
    "https://raw.githubusercontent.com/yifan123/reward-server/"
    "8245af22b75fe640950a880ad31fe702a5d9a014/reward_server/gen_eval.py"
)
TAGS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
CLASSNAMES = ("cat", "dog")
COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
)


def transform(image):
    return torch.tensor(np.asarray(image).mean(axis=(0, 1)), dtype=torch.float32)


class ColorBackend:
    def __init__(self, *, consume=True):
        self.consume = consume
        self.classifiers = []
        self.loaders = []
        self.crops = []

    def zero_shot_classifier(self, model, tokenizer, colors, templates, device):
        assert colors == list(COLORS)
        assert device == "cpu"
        self.classifiers.append((model, tokenizer, templates))
        return object()

    def run_classification(self, model, classifier, loader, device):
        assert not torch.is_grad_enabled()
        assert device == "cpu"
        self.loaders.append((loader.batch_size, loader.num_workers))
        if self.consume:
            self.crops.extend(loader.dataset[i][0] for i in range(len(loader.dataset)))
            batches = [pixels for pixels, _ in loader]
            pixels = torch.cat(batches)
            # Only red/blue are needed to exercise the injected color contract.
            indices = torch.where(pixels[:, 0] >= pixels[:, 2], 0, 4)
        else:
            indices = torch.zeros(len(loader.dataset), dtype=torch.long)
        return torch.nn.functional.one_hot(indices, len(COLORS)).float(), None


def detections(cat=(), dog=()):
    return [np.asarray(rows, dtype=float).reshape(-1, 5) for rows in (cat, dog)]


def box(confidence=0.99, x=0, y=0):
    return [x, y, x + 4, y + 4, confidence]


def metadata(include=(), *, tag="single_object", exclude=()):
    return {
        "tag": tag,
        "prompt": "fixture",
        "include": list(include),
        "exclude": list(exclude),
    }


def requirement(classname="cat", count=1, **kwargs):
    return {"class": classname, "count": count, **kwargs}


def make_evaluator(results, *, factory=build_evaluator, backend=None, **kwargs):
    backend = backend if backend is not None else ColorBackend()
    detector = object()
    calls = []

    def inference_detector(actual_detector, images):
        assert actual_detector is detector
        assert not torch.is_grad_enabled()
        assert all(isinstance(image, np.ndarray) for image in images)
        calls.append(images)
        assert len(images) == len(results)
        return results

    dependencies = {
        "detector": detector,
        "clip_model": object(),
        "transform": transform,
        "tokenizer": object(),
        "classnames": CLASSNAMES,
        "device": "cpu",
        "inference_detector": inference_detector,
        "zsc": backend,
    }
    dependencies.update(kwargs)
    dependencies.setdefault("color_workers", 0)
    return factory(**dependencies), backend, calls


def score(result, meta, **kwargs):
    evaluator, _, _ = make_evaluator([result])
    return evaluator([Image.new("RGB", (64, 64), "red")], [meta], **kwargs)


@pytest.mark.parametrize("workers", [-1, 1.5, "4", None, True, False])
def test_color_workers_validation(workers):
    with pytest.raises(ValueError, match="integer >= 0"):
        make_evaluator([], color_workers=workers)


@pytest.mark.parametrize(
    "n_images,n_metadata", [(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)]
)
def test_input_validation_precedes_inference(n_images, n_metadata):
    evaluator, _, calls = make_evaluator([])
    with pytest.raises(ValueError, match="matching lengths|nonempty"):
        evaluator([None] * n_images, [{}] * n_metadata)
    assert calls == []


def test_ordinary_minimum_count_strict_exact_count_and_exclusion():
    too_many = score(detections([box(), box()]), metadata([requirement()]))
    assert too_many[:3] == ([0.0], [1.0], [0.0])
    excluded = score(
        detections([box()], [box()]),
        metadata([requirement()], exclude=[requirement("dog")]),
    )
    assert excluded[:3] == ([1.0], [0.0], [1.0])
    allowed = score(
        detections([box()], [box()]),
        metadata([requirement()], exclude=[requirement("dog", 2)]),
    )
    assert allowed[:3] == ([1.0], [1.0], [1.0])


def test_dense_reward_is_not_clipped_and_max_objects_is_16():
    result = score(detections([box()] * 20), metadata([requirement()]))
    assert result[:3] == ([-14.0], [1.0], [0.0])
    exact = score(detections([box()] * 20), metadata([requirement(count=16)]))
    assert exact[:3] == ([1.0], [1.0], [1.0])


@pytest.mark.parametrize("tag,threshold", [("single_object", 0.3), ("counting", 0.9)])
def test_confidence_threshold_is_strictly_greater(tag, threshold):
    result = detections([box(threshold - 1e-6), box(threshold), box(threshold + 1e-6)])
    assert score(result, metadata([requirement()], tag=tag))[:3] == (
        [1.0],
        [1.0],
        [1.0],
    )


@pytest.mark.parametrize(
    "relation,cat_boxes,dog_boxes,expected",
    [
        ("left of", [box(x=30)], [box()], ([1.0], [1.0], [1.0])),
        ("right of", [box(x=30)], [box()], ([0.75], [0.0], [0.0])),
        ("left of", [], [box()], ([1 / 3], [0.0], [0.0])),
        ("left of", [box()], [], ([1 / 3], [0.0], [0.0])),
        ("above", [box(y=30)], [box()], ([1.0], [1.0], [1.0])),
        ("below", [box()], [box(y=30)], ([1.0], [1.0], [1.0])),
        ("left of", [box()], [box()], ([0.75], [0.0], [0.0])),
    ],
)
def test_position_success_missing_target_and_failure_quirk(
    relation, cat_boxes, dog_boxes, expected
):
    meta = metadata(
        [requirement(), requirement("dog", position=[relation, 0])], tag="position"
    )
    assert score(detections(cat_boxes, dog_boxes), meta)[:3] == expected


def test_position_requires_every_pair_to_match():
    meta = metadata(
        [requirement(count=2), requirement("dog", position=["left of", 0])],
        tag="position",
    )
    assert score(detections([box(x=30), box()], [box(x=15)]), meta)[:3] == (
        [0.75],
        [0.0],
        [0.0],
    )


def test_color_sorted_top_count_crops_and_classifier_cache():
    image = Image.new("RGB", (64, 64), "blue")
    image.paste("red", (20, 0, 24, 4))
    evaluator, backend, calls = make_evaluator(
        [detections([box(0.7), box(0.99, x=20)])]
    )
    meta = metadata([requirement(color="red")], tag="colors")
    assert evaluator([image], [meta])[:3] == ([0.0], [1.0], [0.0])
    assert evaluator([image], [meta])[:3] == ([0.0], [1.0], [0.0])
    assert len(calls) == 2
    assert len(backend.classifiers) == 1
    assert backend.classifiers[0][2] == [
        "a photo of a {c} cat",
        "a photo of a {c}-colored cat",
        "a photo of a {c} object",
    ]
    assert backend.loaders == [(16, 0), (16, 0)]
    assert all(
        torch.equal(crop, torch.tensor([255.0, 0.0, 0.0])) for crop in backend.crops
    )


@pytest.mark.parametrize(
    "color,expected", [("red", ([1.0], [1.0], [1.0])), ("blue", ([0.5], [0.0], [0.0]))]
)
def test_color_dense_and_binary_rewards(color, expected):
    assert (
        score(detections([box()]), metadata([requirement(color=color)], tag="colors"))[
            :3
        ]
        == expected
    )


def test_color_partial_credit_and_mask_background():
    image = Image.new("RGB", (64, 64), "blue")
    image.paste("red", (20, 0, 24, 4))
    evaluator, _, _ = make_evaluator([detections([box(), box(x=20)])])
    meta = metadata([requirement(count=2, color="red")], tag="colors")
    assert evaluator([image], [meta])[:3] == ([0.75], [0.0], [0.0])

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0:4, 0:2] = 255
    evaluator, backend, _ = make_evaluator([(detections([box()]), [[mask], []])])
    evaluator(
        [Image.new("RGB", (64, 64), "red")], [metadata([requirement(color="red")])]
    )
    assert torch.equal(backend.crops[0], torch.tensor([204.0, 76.5, 76.5]))


def test_default_and_configurable_workers_without_starting_processes():
    assert inspect.signature(build_evaluator).parameters["color_workers"].default == 4
    for workers in (0, 2, 4):
        evaluator, backend, _ = make_evaluator(
            [detections([box()])],
            color_workers=workers,
            backend=ColorBackend(consume=False),
        )
        evaluator([Image.new("RGB", (64, 64))], [metadata([requirement(color="red")])])
        assert backend.loaders == [(16, workers), (16, workers)]


@pytest.mark.parametrize("only_strict", [False, True])
def test_grouped_sentinels_order_and_only_strict(only_strict):
    tags = (*TAGS, "unknown")
    evaluator, backend, calls = make_evaluator([detections([box()])] * len(tags))
    images = [Image.new("RGB", (64, 64), "red")] * len(tags)
    metas = [metadata([requirement(color="red")], tag=tag) for tag in tags]
    scores, ordinary, strict, grouped, grouped_strict = evaluator(
        images, metas, only_strict
    )
    assert scores == strict == [1.0] * len(tags)
    assert ordinary == [float(not only_strict)] * len(tags)
    assert list(grouped) == list(grouped_strict) == list(TAGS)
    for key in TAGS:
        assert grouped[key] == [
            float(not only_strict) if tag == key else -10.0 for tag in tags
        ]
        assert grouped_strict[key] == [1.0 if tag == key else -10.0 for tag in tags]
    assert len(calls) == 1
    assert len(backend.loaders) == len(tags) * (1 if only_strict else 2)


def test_empty_requirements_keep_upstream_score_and_correctness():
    assert score(detections(), metadata())[:3] == ([0], [1.0], [1.0])


def upstream_factory(source):
    """Keep the pinned scoring AST intact; replace only asset loading and workers."""
    tree = ast.parse(source)
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_geneval"
    )
    loader.body = [
        node
        for node in loader.body
        if not (
            isinstance(node, ast.FunctionDef) and node.name in {"timed", "load_models"}
        )
    ]
    for node in ast.walk(loader):
        if isinstance(node, ast.keyword) and node.arg == "num_workers":
            node.value = ast.Constant(value=0)
    module = ast.fix_missing_locations(ast.Module(body=[loader], type_ignores=[]))

    def factory(**deps):
        namespace = {
            "np": np,
            "torch": torch,
            "Image": Image,
            "ImageOps": ImageOps,
            "json": json,
            "defaultdict": defaultdict,
            "DEVICE": deps["device"],
            "zsc": deps["zsc"],
            "inference_detector": deps["inference_detector"],
            "load_models": lambda: (
                deps["detector"],
                (deps["clip_model"], deps["transform"], deps["tokenizer"]),
                deps["classnames"],
            ),
        }
        # Execute only the explicitly requested pinned-source scoring oracle.
        exec(compile(module, "<pinned-upstream-geneval>", "exec"), namespace)  # noqa: S102
        return namespace["load_geneval"]()

    return factory


@pytest.mark.skipif(
    os.environ.get("GENEVAL_COMPARE_UPSTREAM") != "1",
    reason="opt-in pinned upstream download",
)
def test_random_fixtures_match_pinned_upstream():
    with urlopen(UPSTREAM, timeout=30) as response:
        factory = upstream_factory(response.read().decode())
    rng = np.random.default_rng(20260929)
    images, metas, results = [], [], []
    for index in range(240):
        image = Image.new("RGB", (64, 64), "red" if index % 2 else "blue")
        include = []
        for group, classname in enumerate(CLASSNAMES):
            req = requirement(classname, int(rng.integers(1, 5)))
            if rng.random() < 0.5:
                req["color"] = str(rng.choice(["red", "blue"]))
            if group and rng.random() < 0.7:
                req["position"] = [
                    str(rng.choice(["left of", "right of", "above", "below"])),
                    0,
                ]
            include.append(req)
        exclude = (
            [requirement("dog", int(rng.integers(1, 4)))] if rng.random() < 0.5 else []
        )
        metas.append(metadata(include, tag=TAGS[index % len(TAGS)], exclude=exclude))
        rows = [
            [
                box(
                    float(rng.choice([0.3, 0.9, 0.99, 0.5])),
                    int(rng.integers(0, 60)),
                    int(rng.integers(0, 60)),
                )
                for _ in range(int(rng.integers(0, 22)))
            ]
            for _ in CLASSNAMES
        ]
        result = detections(*rows)
        if index % 3 == 0:
            masks = [[rng.random((64, 64)) > 0.5 for _ in boxes] for boxes in result]
            result = (result, masks)
        images.append(image)
        results.append(result)
    for only_strict in (False, True):
        actual, _, _ = make_evaluator(copy.deepcopy(results))
        reference, _, _ = make_evaluator(copy.deepcopy(results), factory=factory)
        assert actual(images, metas, only_strict) == reference(
            images, metas, only_strict
        )
