"""Offline asset/identity checks before publishing the static project page."""
import hashlib
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.references = [], []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for attr in ('src', 'href', 'poster'):
            if attr in attrs:
                self.references.append(attrs[attr])


def main():
    page = Page()
    page.feed((DOCS / 'index.html').read_text())
    assert len(page.ids) == len(set(page.ids)), 'Duplicate element IDs'
    for link in page.references:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (DOCS / url.path).is_file(), link
        if url.fragment:
            assert url.fragment in page.ids, link
    source = (DOCS / 'media.js').read_text()
    data = json.loads(source.removeprefix('window.PROJECT_MEDIA = ').removesuffix(';\n'))
    provenance = json.loads((DOCS / 'assets/provenance.json').read_text())
    assert data == provenance['media']
    for asset in provenance['files']:
        assert hashlib.sha256((DOCS / asset['asset']).read_bytes()).hexdigest() == asset['sha256']
    for pair in data['videos']:
        streams = []
        for side in pair['sides']:
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                     '-show_entries', 'stream=codec_name,width,height,nb_frames,avg_frame_rate,duration',
                                     '-of', 'json', str(DOCS / side['video'])],
                                    check=True, capture_output=True, text=True, timeout=15)
            stream, = json.loads(result.stdout)['streams']
            assert stream['codec_name'] == 'h264'
            assert stream['nb_frames'] == '53'
            assert stream['avg_frame_rate'] == '8/1'
            streams.append(stream)
        assert streams[0] == streams[1], pair['index']
    for file in [ROOT / 'README.md', *DOCS.glob('*.js'), *DOCS.glob('*.html'), *DOCS.glob('*.css'), DOCS / 'assets/provenance.json']:
        assert not re.search(r'/m2v_intern|/Users/|gho_[A-Za-z0-9]{20}|WANDB_API_KEY|corp\.kuaishou', file.read_text()), file
    print(f'PASS: {len(data["videos"])} matched video pairs, {len(data["images"])} image pairs; links, hashes, H264/8FPS/53-frame streams and public metadata checked.')


if __name__ == '__main__':
    main()
