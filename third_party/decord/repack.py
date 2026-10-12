"""Repair only Decord0.6 public wheel tags; preserve every runtime byte."""
import argparse,base64,csv,hashlib,io,json,urllib.request,zipfile
from pathlib import Path

SOURCE_URL = 'https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl'
SOURCE_SHA256 = '51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976'
NAME = 'decord-0.6.0-1-py3-none-manylinux2010_x86_64.whl'

def repair(source, destination):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Public Decord source wheel digest mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        original = {name: wheel.read(name) for name in wheel.namelist()}
    files = dict(original)
    metadata = 'decord-0.6.0.dist-info/WHEEL'
    old = b'Tag: cp36-cp36m-manylinux2010_x86_64'
    new = b'Tag: py3-none-manylinux2010_x86_64'
    if files[metadata].count(old) != 1:
        raise ValueError('Unexpected upstream wheel tag, refuse generic rewrite')
    files[metadata] = files[metadata].replace(old, new)
    record = 'decord-0.6.0.dist-info/RECORD'
    output = io.StringIO(newline=''); writer = csv.writer(output, lineterminator='\n')
    for name, data in sorted(files.items()):
        if name == record:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(data)])
    writer.writerow([record, '', '']); files[record] = output.getvalue().encode()
    changed = [name for name in files if files[name] != original[name]]
    if set(changed) != {metadata, record}:
        raise ValueError('Repair changed runtime payload or license')
    destination = Path(destination)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for name, data in sorted(files.items()):
            out.writestr(name, data)
    return {'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256,
            'output': destination.name, 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'changed_members': changed, 'runtime_members_identical': True,
            'rationale': 'Public filename advertises py3-none, but internal WHEEL incorrectly declares cp36-cp36m. Runtime uses ctypes/CFFI, no CPython extension ABI change.'}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True); args = parser.parse_args()
    source = Path(args.source)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            source.write_bytes(response.read())
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    record = repair(source, root / NAME)
    (root / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
