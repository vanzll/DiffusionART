import unittest

from visual_rl.rewards import ContinuousRGBInput, FixedNormalization, weighted_score

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch is required for tensor contract tests")
class TensorTests(unittest.TestCase):
    def make(self, **changes):
        args = dict(
            frames=torch.full((2, 4, 3, 8, 8), 0.5, requires_grad=True),
            prompts=("first", "second"),
            fps=8,
            value_range="zero_one",
        )
        return ContinuousRGBInput(**dict(args, **changes))

    def test_graph_preserved_and_exact_weighted_gradient(self):
        inputs = self.make().validate()
        frames = inputs.frames
        self.assertIs(inputs.validate().frames, frames)
        raw = {"quality": frames.mean((1, 2, 3, 4)), "motion": frames.square().mean((1, 2, 3, 4))}
        result = weighted_score(
            raw,
            {"quality": 2, "motion": 1},
            space="fixed_normalized",
            normalization={k: FixedNormalization(0, 2, "toy-v1") for k in raw},
        )
        self.assertEqual(result.weighted.shape, (2,))
        result.weighted.sum().backward()
        torch.testing.assert_close(frames.grad, torch.full_like(frames, 1.5 / frames[0].numel()))
        with self.assertRaises(ValueError):
            result.to_dict()

    def test_bad_input_contract(self):
        for change in [
            dict(source="mp4_ste"),
            dict(fps=0),
            dict(value_range="guess"),
            dict(prompts="ab"),
            dict(prompts=("", "second")),
            dict(prompts=("one",)),
            dict(frames=torch.zeros(2, 4, 3, 8, 8, dtype=torch.uint8)),
            dict(frames=torch.zeros(2, 3, 4, 8, 8, requires_grad=True)),
            dict(frames=torch.zeros(2, 4, 3, 8, 8)),
            dict(frames=torch.full((2, 4, 3, 8, 8), float("nan"), requires_grad=True)),
            dict(frames=torch.full((2, 4, 3, 8, 8), 1.1, requires_grad=True)),
        ]:
            with self.subTest(change=tuple(change)), self.assertRaises(ValueError):
                self.make(**change).validate()
        self.make(frames=torch.zeros(2, 4, 3, 8, 8)).validate(require_grad=False)
        self.make(
            frames=torch.full((2, 4, 3, 8, 8), -1.0, requires_grad=True),
            value_range="minus_one_one",
        ).validate()

    def test_no_broadcasting_or_detaching(self):
        for right in [
            torch.ones(1),
            torch.ones(2, 1),
            1.0,
            torch.ones(2, dtype=torch.float64),
            torch.tensor([1.0, float("nan")]),
        ]:
            with self.assertRaises(ValueError):
                weighted_score({"a": torch.ones(2), "b": right}, {"a": 1, "b": 0}, space="raw")
        with self.assertRaises(ValueError):
            weighted_score({"a": torch.ones(0)}, {"a": 1}, space="raw")


if __name__ == "__main__":
    unittest.main()
