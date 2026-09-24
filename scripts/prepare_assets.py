"""Export only explicitly selected public media; never publish research paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualitative-root', type=Path, required=True)
    parser.add_argument('--paper-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.qualitative_root
    docs = Path(__file__).resolve().parents[1] / 'docs'
    assets = docs / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    provenance = []

    def copy(source, name):
        target = assets / name
        shutil.copyfile(source, target)
        provenance.append({'asset': f'assets/{name}',
                           'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        return f'assets/{name}'

    def poster(source, name):
        target = assets / name
        with Image.open(source) as image:
            image.convert('RGB').save(target, 'JPEG', quality=92)
        provenance.append({'asset': f'assets/{name}',
                           'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                           'transform': 'JPEG poster, full frame, quality 92'})
        return f'assets/{name}'

    selections = json.loads((root / 'figures/manifest_selected26.json').read_text())
    videos = []
    choices = [(94, 'Ocean'), (113, 'Coast'), (136, 'Viaduct'),
               (139, 'Architecture'), (106, 'Autumn'), (144, 'Fantasy')]
    for index, label in choices:
        item, = [r for r in selections['records'] if r['index'] == index and r['run'] == '7l6xk2na']
        entry = {k: item[k] for k in ('index', 'text', 'seed', 'run')}
        entry.update(label=label, sides=[])
        for side, data in zip(('before', 'after'), item['pair']):
            source = Path(data['video'])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == data['sha256']
            entry['sides'].append({'step': data['step'],
                                  'video': copy(source, f'video-{index}-{side}.mp4'),
                                  'poster': poster(Path(data['frames'][0]), f'video-{index}-{side}.jpg')})
        videos.append(entry)
    poster(root / 'frames-showcase/00106-step20/03.png', 'hero.jpg')

    images = []
    selected = json.loads((root / 't2i/selection.json').read_text())['selected']
    for slug, label in [('chicago', 'Christmas'), ('teddy-bears', 'Jungle'), ('greenhouse', 'Greenhouse')]:
        pair = sorted([r for r in selected if r['slug'] == slug], key=lambda r: r['step'])
        entry = {'label': label, 'text': pair[0]['prompt'], 'run': 'mofumvbw',
                 'benchmark': 'PickScore', 'index': pair[0]['index'], 'sides': []}
        for side, data in zip(('before', 'after'), pair):
            source = root / 't2i' / data['local_file']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == data['sha256']
            entry['sides'].append({'step': data['step'], 'image': copy(source, f'image-{slug}-{side}.jpg')})
        images.append(entry)
    selected_multi = json.loads((root / 't2i-appendix/manifest.json').read_text())['rendered_pairs']['multi']
    for index, label in [(13, 'Glass'), (0, 'Suitcases'), (1, 'Horse')]:
        pair, = [r for r in selected_multi if r['index'] == index]
        entry = {'label': label, 'text': pair['prompt'], 'run': '3t35bwps',
                 'benchmark': 'Multi-reward', 'index': index, 'sides': []}
        for side, data in zip(('before', 'after'), pair['sides']):
            source = root / 't2i-appendix' / data['local']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == data['sha256']
            entry['sides'].append({'step': data['step'], 'image': copy(source, f'image-{index}-{side}.jpg')})
        images.append(entry)
    copy(root / 'cover/diffusionart_cover_left.png', 'efficiency.png')
    copy(root / 'cover/cover_curves.csv', 'efficiency.csv')
    copy(args.paper_root / 'Figures/training-curves/t2v-paper-v1/t2v_pairwise_gpu_hours.png', 'pairwise.png')
    copy(args.paper_root / 'Figures/training-curves/i2v-paper-v1/i2v_main_training_curves.png', 'i2v.png')
    payload = {'videos': videos, 'images': images}
    (docs / 'media.js').write_text('window.PROJECT_MEDIA = ' + json.dumps(payload, ensure_ascii=True) + ';\n')
    (assets / 'provenance.json').write_text(json.dumps({
        'media': payload, 'files': provenance,
        'note': 'Curated qualitative examples. Videos are byte-identical to the retained evaluations; images are original JPEGs. Posters are full-frame JPEG previews.',
        'video_identity': 'Same run, prompt, seed and sampling protocol within each pair.',
        'image_identity': 'Same prompt and logged media index. Per-image seeds are not asserted.'}, indent=2) + '\n')
    print(f'Exported {len(videos)} video pairs and {len(images)} image pairs to {docs}')


if __name__ == '__main__':
    main()
