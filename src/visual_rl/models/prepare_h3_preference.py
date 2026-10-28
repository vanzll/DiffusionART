"""Build the three-reward manifest from verified existing AV assets and local PickScore."""

import argparse
import json
import os
from pathlib import Path

from ..rewards.av_profiles import (
    PICKSCORE_PROCESSOR_REPOSITORY,
    PICKSCORE_PROCESSOR_REVISION,
    PICKSCORE_REPOSITORY,
    PICKSCORE_REVISION,
)
from .prepare_minimax_h3 import build_reward_manifest, write_manifest


def prepare_manifest(legacy_manifest, pickscore, processor):
    legacy_manifest = Path(legacy_manifest)
    if legacy_manifest.stat().st_size > 1024 * 1024:
        raise ValueError("existing reward manifest exceeds 1 MiB")
    legacy = json.loads(legacy_manifest.read_text())
    if legacy.get('schema_version') != 1:
        raise ValueError("unsupported existing reward manifest")
    retained = {'clap', 'imagebind', 'imagebind_source'}
    paths = {key: os.path.abspath(legacy_manifest.parent / legacy['paths'][key]) for key in retained}
    provenance = {key: legacy['provenance'][key] for key in retained}
    paths.update(pickscore=str(Path(pickscore).absolute()), pickscore_processor=str(Path(processor).absolute()))
    provenance.update(
        pickscore={'repo': PICKSCORE_REPOSITORY, 'revision': PICKSCORE_REVISION,
                   'license': 'See upstream model card and PickScore repository LICENSE (MIT code)'},
        pickscore_processor={'repo': PICKSCORE_PROCESSOR_REPOSITORY, 'revision': PICKSCORE_PROCESSOR_REVISION,
                             'license': 'See upstream OpenCLIP model card and repository license'},
    )
    result = build_reward_manifest(paths, provenance)
    for key in retained:
        if result['sha256'][key] != legacy['sha256'][key]:
            raise ValueError(f"retained scorer differs from the successful run inventory: {key}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-manifest', required=True)
    parser.add_argument('--pickscore', required=True, help='materialized official model directory, no symlinks')
    parser.add_argument('--pickscore-processor', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args(argv)
    if Path(args.manifest).exists():
        raise ValueError("use a fresh three-reward manifest path; existing assets/receipts are never overwritten")
    result = prepare_manifest(args.legacy_manifest, args.pickscore, args.pickscore_processor)
    write_manifest(args.manifest, result)
    print(json.dumps({'manifest': str(Path(args.manifest).absolute()), 'assets': sorted(result['paths'])}))


if __name__ == '__main__':
    main()
