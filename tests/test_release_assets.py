import re
import struct
from pathlib import Path
from xml.etree import ElementTree


def test_readme_images_are_valid_local_assets():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    images = re.findall(r'<img\s[^>]*src="(assets/readme/[^"]+)"', readme)
    assert images, "README must contain local figure assets"
    assert len(images) == len(set(images)), "README figure paths must be unique"
    for filename in images:
        path = root / filename
        data = path.read_bytes()
        if path.suffix == ".png":
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", data[16:24])
            assert width > 0 and height > 0
        elif path.suffix == ".svg":
            assert ElementTree.fromstring(data).tag.endswith("svg")
        else:
            raise AssertionError(f"Unexpected README image format: {path}")


def test_readme_relative_links_exist():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    links = re.findall(r'(?:href|src)="([^"]+)"|\]\(([^)]+)\)', readme)
    for html, markdown in links:
        target = html or markdown
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (root / target.split("#")[0]).exists(), target
