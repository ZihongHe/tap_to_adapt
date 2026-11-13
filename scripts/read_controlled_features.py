"""Authenticate and load the controlled-access acoustic/correction dataset.

Use a separately authorized 32-byte key file stored outside the repository.
The command reports array shapes only and never writes decrypted observations.
Import load_pairs() for an in-memory research workflow.
"""
from pathlib import Path
import argparse
import io
import json
import zipfile

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b'TTACR01\x00'


def load_pairs(archive_path, key_path):
    """Return verified arrays, keeping participant data in process memory.

    Wrong keys or modified ciphertext raise InvalidTag. Group values are
    release-local pseudonyms; use group-disjoint evaluation to avoid leakage.
    """
    key = Path(key_path).read_bytes()
    if len(key) != 32:
        raise ValueError('Expected a separately supplied 32-byte binary key.')
    encrypted = Path(archive_path).read_bytes()
    if not encrypted.startswith(MAGIC) or len(encrypted) < 36:
        raise ValueError('Unrecognized controlled-data container.')
    offset = len(MAGIC)
    plain = AESGCM(key).decrypt(encrypted[offset:offset+12], encrypted[offset+12:], MAGIC)
    with zipfile.ZipFile(io.BytesIO(plain)) as archive:
        if set(archive.namelist()) != {'feature_label_pairs.npz', 'SCHEMA.json'}:
            raise ValueError('Unexpected archive members.')
        with np.load(io.BytesIO(archive.read('feature_label_pairs.npz')), allow_pickle=False) as data:
            if set(data.files) != {'features', 'labels', 'group'}:
                raise ValueError('Unexpected array schema.')
            result = {name: data[name].copy() for name in data.files}
    x, y, g = result['features'], result['labels'], result['group']
    if x.ndim != 2 or x.shape[1] != 10752 or y.shape != (len(x),) or g.shape != y.shape:
        raise ValueError('Inconsistent array shapes.')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid values in the authenticated data.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--key-file', type=Path, required=True)
    args = parser.parse_args()
    arrays = load_pairs(args.archive, args.key_file)
    print(json.dumps({name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                      for name, value in arrays.items()}, indent=2))
