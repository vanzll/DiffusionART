"""Independent historical OCR cases; no native engine or model assets required."""

import traceback
import warnings

import numpy as np
import pytest
import torch

from visual_rl.rewards.ocr import OCRScore, extract_target

pytest.importorskip("rapidfuzz")
Image = pytest.importorskip("PIL.Image")


class Engine:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def ocr(self, image, *, cls):
        assert not torch.is_grad_enabled()
        assert isinstance(image, np.ndarray) and image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[-1] == 3
        assert cls is False
        self.calls.append(image.copy())
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


def recognized(*lines):
    return [[[None, (text, confidence)] for text, confidence in lines]]


@pytest.mark.parametrize(
    "prompt,target",
    [
        ('A sign "He LLo" and "ignored"', "hello"),
        ('" A\tB\nC "', "a\tb\nc"),
        ('"\t"', "\t"),
        ('"A\u00a0B"', "a\u00a0b"),
    ],
)
def test_extract_target(prompt, target):
    assert extract_target(prompt) == target


@pytest.mark.parametrize(
    "prompt",
    [None, 4, "", "unquoted", "'single quotes'", 'missing "end', '""', '"   " then "valid"'],
)
def test_extract_invalid_target(prompt):
    with pytest.raises(ValueError):
        extract_target(prompt)


@pytest.mark.parametrize(
    "result,prompt,expected",
    [
        (recognized(("He L", 0.01), ("LO", 1)), 'sign " hello "', 1),
        (recognized(("BAD", 0), ("BAD", -1), ("OK", 1e-20)), '"ok"', 1),
        (recognized(("BAD", float("nan")), ("OK", 1)), '"ok"', 1),
        (recognized(("prefixCATsuffix", 1)), '"cat"', 1),
        (recognized(("cat", 1)), '"cats"', 0.75),
        (recognized(("cut", 1)), '"cat"', 2 / 3),
        (recognized(("ac", 1)), '"ca"', 0),
        (recognized(("unrelatedlongtext", 1)), '"cat"', 0),
        (recognized(("abc", 1), ("def", 1)), '"abcdef"', 1),
        (recognized(("def", 1), ("abc", 1)), '"abcdef"', 0),
        (recognized(("a\tb", 1)), '"ab"', 0.5),
        (recognized(("ab", 1)), '"a\tb"', 2 / 3),
        (recognized(("a\nb", 1)), '"ab"', 0.5),
        (recognized(("FIRST", 1)), '"first" and "second"', 1),
        (recognized(("second", 1)), '"first" and "second"', 0),
        ([None], '"text"', 0),
        ([[]], '"text"', 0),
        (recognized(("ignored", 0)), '"text"', 0),
    ],
)
def test_historical_scores(result, prompt, expected):
    engine = Engine([result])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        output = OCRScore(engine, error_policy="raise")(
            [Image.new("RGB", (2, 3))],
            [prompt],
            microbatch=1,
        )
    assert output.raw["ocr"].tolist() == pytest.approx([expected])
    assert output.space == "raw" and output.weights == {"ocr": 1}
    assert output.normalization is None and output.fixed_normalized is None
    torch.testing.assert_close(output.weighted, output.raw["ocr"])


@pytest.mark.parametrize("microbatch", [1, 2, 4, 20])
@pytest.mark.parametrize("pil_input", [False, True])
def test_conversion_order_and_microbatch_invariance(monkeypatch, microbatch, pil_input):
    import visual_rl.rewards.ocr as module

    pixels = (
        torch.tensor(
            [
                [[0, 0.5, 1.5], [127.5, 255, 254.5]],
                [[21, 22, 23], [24, 25, 26]],
                [[101, 102, 103], [104, 105, 106]],
            ]
        )
        .unsqueeze(0)
        .repeat(5, 1, 1, 1)
        / 255
    )
    pixels[:, 1, 0, 0] = torch.arange(5) / 255
    pixels.requires_grad_()
    expected = np.array(
        [
            [[0, 21, 101], [0, 22, 102], [2, 23, 103]],
            [[128, 24, 104], [255, 25, 105], [254, 26, 106]],
        ],
        dtype=np.uint8,
    )
    arrays = []
    for index in range(5):
        array = expected.copy()
        array[0, 0, 1] = index
        arrays.append(array)
    batches = []
    original = module.quantized_rgb

    def convert(batch):
        batches.append(len(batch))
        return original(batch)

    monkeypatch.setattr(module, "quantized_rgb", convert)
    engine = Engine([recognized((text, 1)) for text in ["cat", "z", "yx", "wrong", "abc"]])
    inputs = [Image.fromarray(array) for array in arrays] if pil_input else pixels
    result = OCRScore(engine, error_policy="raise")(
        inputs,
        ['"cats"', '"z"', '"yx"', '"q"', '"ab"'],
        microbatch=microbatch,
    )
    assert result.raw["ocr"].tolist() == [0.75, 1, 1, 0, 1]
    for actual, expected_array in zip(engine.calls, arrays):
        np.testing.assert_array_equal(actual, expected_array)
    assert len(engine.calls) == 5
    assert batches == (
        [] if pil_input else [min(microbatch, 5 - i) for i in range(0, 5, microbatch)]
    )
    assert not result.weighted.requires_grad and pixels.grad is None


@pytest.mark.parametrize(
    "bad_result",
    [
        None,
        [],
        [1],
        [[None]],
        [[[None, ("text",)]]],
        recognized((None, 1)),
        recognized(("text", "invalid")),
        RuntimeError("private-engine-secret"),
    ],
)
@pytest.mark.parametrize("policy", ["raise", "zero"])
def test_errors_are_indexed_and_sanitized(bad_result, policy, capsys):
    engine = Engine([recognized(("ok", 1)), bad_result, recognized(("ok", 1))])
    scorer = OCRScore(engine, error_policy=policy)
    images = torch.zeros(3, 3, 2, 2)
    prompts = ['private-prompt-secret "ok"'] * 3
    if policy == "zero":
        with pytest.warns(RuntimeWarning, match="sample 1") as caught:
            result = scorer(images, prompts, microbatch=1)
        assert result.raw["ocr"].tolist() == [1, 0, 1]
        assert len(engine.calls) == 3 and len(caught) == 1
        diagnostic = str(caught[0].message)
    else:
        with pytest.raises(RuntimeError, match="sample 1") as caught:
            scorer(images, prompts, microbatch=1)
        assert len(engine.calls) == 2
        diagnostic = "".join(traceback.format_exception_only(caught.type, caught.value))
        assert caught.value.__suppress_context__
    assert "private" not in diagnostic
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("policy", ["zero", "raise"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(policy, error):
    scorer = OCRScore(Engine([error()]), error_policy=policy)
    with pytest.raises(error):
        scorer(torch.zeros(1, 3, 1, 1), ['"ok"'], microbatch=1)


@pytest.mark.parametrize(
    "prompts", [[], '"ok"', b'"ok"', ['"ok"'], ['"ok"', "bad"], ['"ok"', '"  "'], ['"ok"', None]]
)
@pytest.mark.parametrize("policy", ["raise", "zero"])
def test_all_prompts_are_preflighted_before_engine(prompts, policy):
    engine = Engine([])
    with pytest.raises(ValueError):
        OCRScore(engine, error_policy=policy)(torch.zeros(2, 3, 2, 2), prompts, microbatch=1)
    assert not engine.calls


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros(2, 2, 2, 2),
        torch.zeros(2, 3, 0, 2),
        torch.zeros(2, 3, 2),
        torch.zeros(0, 3, 2, 2),
        torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
        torch.stack([torch.zeros(3, 2, 2), torch.full((3, 2, 2), float("nan"))]),
        torch.stack([torch.zeros(3, 2, 2), torch.full((3, 2, 2), float("inf"))]),
        torch.stack([torch.zeros(3, 2, 2), torch.full((3, 2, 2), -0.01)]),
        torch.stack([torch.zeros(3, 2, 2), torch.full((3, 2, 2), 1.01)]),
        [Image.new("RGB", (2, 2)), Image.new("RGBA", (2, 2))],
        [Image.new("RGB", (2, 2)), Image.new("L", (2, 2))],
        [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
        None,
    ],
)
def test_invalid_inputs_fail_before_engine(images):
    engine = Engine([])
    with pytest.raises(ValueError):
        OCRScore(engine, error_policy="zero")(images, ['"ok"'] * 2, microbatch=1)
    assert not engine.calls


@pytest.mark.parametrize("microbatch", [True, False, 0, -1, 1.5, None])
def test_invalid_microbatch(microbatch):
    engine = Engine([])
    with pytest.raises(ValueError, match="microbatch"):
        OCRScore(engine, error_policy="raise")(
            torch.zeros(1, 3, 1, 1),
            ['"ok"'],
            microbatch=microbatch,
        )
    assert not engine.calls


def test_constructor_requires_explicit_policy_and_engine():
    with pytest.raises(TypeError):
        OCRScore(Engine([]))
    with pytest.raises(ValueError, match="error_policy"):
        OCRScore(Engine([]), error_policy="ignore")
    with pytest.raises(ValueError, match="engine"):
        OCRScore(object(), error_policy="raise")


def test_frozen_module_engine_and_fp32_output():
    class ModuleEngine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def ocr(self, image, *, cls):
            assert not self.training and not torch.is_grad_enabled()
            return recognized(("ok", 1))

    engine = ModuleEngine()
    scorer = OCRScore(engine, error_policy="raise").train().to(dtype=torch.float16)
    assert not scorer.training and not engine.training and not engine.weight.requires_grad
    engine.train()
    result = scorer(torch.zeros(1, 3, 1, 1, requires_grad=True), ['"ok"'], microbatch=1)
    assert not engine.training and engine.weight.grad is None
    assert result.raw["ocr"].dtype == torch.float32
    assert result.raw["ocr"].device == torch.device("cpu")
    assert not result.weighted.requires_grad


def test_output_device_uses_movable_empty_buffer_without_gpu(monkeypatch):
    import visual_rl.rewards.ocr as module

    scorer = OCRScore(Engine([recognized(("ok", 1))]), error_policy="raise", device="meta")
    buffers = list(scorer.buffers())
    assert len(buffers) == 1 and buffers[0].numel() == 0
    assert buffers[0].device == torch.device("meta")
    scorer.to_empty(device="cpu").to(dtype=torch.float64)
    seen = []
    original = module.torch.tensor

    def tensor(*args, **kwargs):
        seen.append(kwargs.get("device"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module.torch, "tensor", tensor)
    result = scorer([Image.new("RGB", (2, 2))], ['"ok"'], microbatch=1)
    assert seen == [next(scorer.buffers()).device]
    assert result.raw["ocr"].device == torch.device("cpu")
    assert result.raw["ocr"].dtype == torch.float32
