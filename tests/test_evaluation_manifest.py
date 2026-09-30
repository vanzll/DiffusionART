import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from visual_rl.evaluation import (
    EvaluationManifest,
    SampleRecord,
    audit_manifest,
    file_sha256,
    resolve_media,
)
from visual_rl.rewards import weighted_score


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "media").mkdir()
        (self.root / "media/z.mp4").write_bytes(b"video-z")
        (self.root / "media/a.mp4").write_bytes(b"video-a")
        self.z = SampleRecord.from_file(
            media_root=self.root,
            sample_id="first",
            prompt="First prompt",
            seed=7,
            optimizer_step=10,
            framecount=16,
            fps=8,
            path="media/z.mp4",
            scores=weighted_score({"VQ": 2}, {"VQ": 3}, space="raw"),
        )
        self.a = replace(
            self.z,
            sample_id="second",
            prompt="Second prompt",
            seed=8,
            path="media/a.mp4",
            content_sha256=hashlib.sha256(b"video-a").hexdigest(),
        )
        self.manifest = EvaluationManifest(10, ("first", "second"), (self.z, self.a))

    def test_roundtrip_identity_not_order(self):
        manifest = EvaluationManifest.from_json(self.manifest.to_json())
        self.assertEqual(manifest.sha256, self.manifest.sha256)
        shuffled = replace(
            manifest,
            samples=tuple(reversed(manifest.samples)),
            expected_sample_ids=("second", "first"),
        )
        self.assertEqual(shuffled.sha256, manifest.sha256)
        self.assertEqual(
            {s.sample_id: s.path for s in shuffled.samples},
            {"first": "media/z.mp4", "second": "media/a.mp4"},
        )
        audit = audit_manifest(manifest, media_root=self.root)
        self.assertTrue(audit.complete)
        self.assertEqual(audit.completed_count, 2)

    def test_omitted_record_and_missing_file_never_complete(self):
        (self.root / "media/z.mp4").unlink()
        manifest = replace(self.manifest, samples=(self.z,))
        audit = audit_manifest(manifest, media_root=self.root)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.completed_count, 0)
        self.assertEqual(
            [s.media_status for s in audit.samples], ["missing_file", "missing_record"]
        )

    def test_missing_score_and_hash(self):
        manifest = replace(
            self.manifest,
            samples=(replace(self.z, scores=None), replace(self.a, content_sha256=None)),
        )
        audit = audit_manifest(manifest, media_root=self.root)
        self.assertFalse(audit.complete)
        self.assertEqual(audit.samples[0].media_status, "verified")
        self.assertFalse(audit.samples[0].scores_present)
        self.assertEqual(audit.samples[1].media_status, "missing_hash")

    def test_swapped_file_and_bounded_read(self):
        (self.root / "media/z.mp4").write_bytes(b"video-a")
        audit = audit_manifest(self.manifest, media_root=self.root)
        self.assertEqual(audit.samples[0].media_status, "hash_mismatch")
        self.assertFalse(audit.complete)
        with self.assertRaises(ValueError):
            file_sha256(self.root / "media/a.mp4", max_bytes=2)
        (self.root / "empty.mp4").touch()
        for path in [self.root / "empty.mp4", self.root / "media"]:
            with self.assertRaises(ValueError):
                file_sha256(path)

    def test_path_validation_and_symlinks(self):
        for path in [
            "/tmp/a",
            "../a",
            "a/../b",
            "./a",
            "a//b",
            "a/",
            "https://host/a",
            "C:/a",
            "a\\b",
            "a\nb",
            "a\x7fb",
            "",
        ]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(self.z, path=path)
        (self.root / "link").symlink_to(self.root / "media", target_is_directory=True)
        (self.root / "dangling.mp4").symlink_to(self.root / "absent")
        for path in ["link/z.mp4", "dangling.mp4"]:
            with self.assertRaises(ValueError):
                resolve_media(self.root, path)
        special = "media/a #?%25.mp4"
        self.assertEqual(resolve_media(self.root, special), self.root.resolve() / special)

    def test_schema_and_identity_validation(self):
        for changes in [
            dict(seed=True),
            dict(seed=2**53),
            dict(framecount=0),
            dict(fps=float("nan")),
            dict(content_sha256="a"),
            dict(sample_id=""),
            dict(prompt=""),
        ]:
            with self.assertRaises(ValueError):
                replace(self.z, **changes)
        for changes in [
            dict(samples=(self.z, self.z)),
            dict(samples=(replace(self.z, optimizer_step=11),)),
            dict(expected_sample_ids=()),
            dict(expected_sample_ids=("first", "first")),
            dict(samples=(replace(self.z, sample_id="unknown"),)),
        ]:
            with self.assertRaises(ValueError):
                replace(self.manifest, **changes)
        for key in ["prompt", "path", "seed", "framecount", "fps", "optimizer_step"]:
            data = json.loads(self.manifest.to_json())
            data["samples"][0][key] = "changed" if key in ("prompt", "path") else 99
            with self.assertRaises(ValueError):
                EvaluationManifest.from_json(json.dumps(data))
        with self.assertRaises(ValueError):
            EvaluationManifest.from_json(
                self.manifest.to_json().replace(
                    '"schema_version":1', '"schema_version":1,"schema_version":1'
                )
            )


if __name__ == "__main__":
    unittest.main()
