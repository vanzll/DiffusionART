import json
import math
import unittest

from visual_rl.rewards import FixedNormalization, ScoreResult, weighted_score


class ScalarTests(unittest.TestCase):
    def test_explicit_spaces_and_no_coefficient_renormalization(self):
        norm = {
            "VQ": FixedNormalization(2, 2, "checkpoint-A"),
            "MQ": FixedNormalization(1, 4, "checkpoint-A"),
        }
        raw = {"VQ": 6, "MQ": 9}
        weights = {"VQ": 2, "MQ": 1}
        result = weighted_score(raw, weights, space="fixed_normalized", normalization=norm)
        self.assertEqual(result.weighted, 6)
        self.assertEqual(dict(result.raw), raw)
        self.assertEqual(dict(result.fixed_normalized), {"VQ": 2, "MQ": 2})
        self.assertEqual(weighted_score(raw, weights, space="raw", normalization=norm).weighted, 21)
        self.assertEqual(
            ScoreResult.from_dict(
                json.loads(json.dumps(result.to_dict(), sort_keys=True))
            ).to_dict(),
            result.to_dict(),
        )
        raw["VQ"], weights["VQ"] = 100, 100
        self.assertEqual(result.raw["VQ"], 6)
        self.assertEqual(result.weights["VQ"], 2)
        with self.assertRaises(TypeError):
            result.raw["VQ"] = 0

    def test_signed_and_zero_coefficients_are_literal(self):
        self.assertEqual(weighted_score({"x": 2}, {"x": -3}, space="raw").weighted, -6)
        self.assertEqual(weighted_score({"x": 2}, {"x": 0}, space="raw").weighted, 0)

    def test_invalid_or_missing_components(self):
        cases = [
            ({}, {}, "raw", None),
            ({"x": 1}, {}, "raw", None),
            ({"x": 1}, {"x": 1}, "normalized", None),
            ({"x": 1}, {"x": 1}, "fixed_normalized", None),
            ({"x": 1}, {"x": 1}, "raw", {}),
            ({"x": 1}, {"x": 1}, "raw", {"x": (0, 1)}),
        ]
        for raw, weights, space, norm in cases:
            with self.subTest(raw=raw, space=space), self.assertRaises(ValueError):
                weighted_score(raw, weights, space=space, normalization=norm)
        with self.assertRaises(TypeError):
            weighted_score({"x": 1}, {"x": 1})

    def test_nonfinite_zero_weight_components_and_overflow(self):
        for value in [math.nan, math.inf, -math.inf, True, "1"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                weighted_score({"x": value}, {"x": 0}, space="raw")
            with self.assertRaises(ValueError):
                weighted_score({"x": 1}, {"x": value}, space="raw")
        with self.assertRaises(ValueError):
            weighted_score({"x": 1e308}, {"x": 1e308}, space="raw")
        for mean, std, source in [
            (math.nan, 1, "a"),
            (0, 0, "a"),
            (0, -1, "a"),
            (0, math.inf, "a"),
            (0, 1, ""),
        ]:
            with self.assertRaises(ValueError):
                FixedNormalization(mean, std, source)
        with self.assertRaises(ValueError):
            weighted_score(
                {"x": 1, "y": 2},
                {"x": 1, "y": 0},
                space="raw",
                normalization={"x": FixedNormalization(0, 1, "a")},
            )

    def test_order_independence_and_tampered_score(self):
        raw = {"z": -1e16, "b": 1, "a": 1e16}
        result = weighted_score(raw, dict.fromkeys(raw, 1), space="raw")
        reordered = weighted_score(
            dict(reversed(list(raw.items()))), dict.fromkeys(raw, 1), space="raw"
        )
        self.assertEqual(result.weighted, reordered.weighted)
        altered = result.to_dict()
        altered["weighted"] = 42
        with self.assertRaises(ValueError):
            ScoreResult.from_dict(altered)


if __name__ == "__main__":
    unittest.main()
