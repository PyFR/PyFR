from io import BytesIO
import struct

import numpy as np
import pytest

from pyfr.readers.stl import read_stl


def _binary_stl(ntri, header=b'PyFR test'):
    data = header.ljust(80, b'\0')[:80] + struct.pack('<I', ntri)

    for _ in range(ntri):
        data += struct.pack('<3f', 0, 0, 1)
        data += struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0)
        data += struct.pack('<H', 0)

    return data


def _ascii_stl(lines):
    return ('\n'.join(lines) + '\n').encode('ascii')


_FACET = [
    'facet normal 0 0 1',
    'outer loop',
    'vertex 0 0 0',
    'vertex 1 0 0',
    'vertex 0 1 0',
    'endloop',
    'endfacet',
]

_TRI = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)


def test_binary_stl_standard_header(tmp_path):
    path = tmp_path / 'std.stl'
    path.write_bytes(_binary_stl(3))

    tris = read_stl(str(path))

    assert tris.shape == (3, 4, 3)
    assert np.array_equal(tris[0], _TRI)


def test_binary_stl_solid_header(tmp_path):
    # The STL spec allows the 80-byte binary header to contain anything,
    # including the word 'solid', so such files must not be parsed as ASCII
    path = tmp_path / 'solid-header.stl'
    path.write_bytes(_binary_stl(1, header=b'solid from cad tool'))

    tris = read_stl(str(path))

    assert tris.shape == (1, 4, 3)
    assert np.array_equal(tris[0], _TRI)


def test_ascii_stl_endsolid_with_name():
    data = _ascii_stl(['solid cube'] + _FACET + ['endsolid cube'])

    tris = read_stl(BytesIO(data))

    assert tris.shape == (1, 4, 3)
    assert np.array_equal(tris[0], _TRI)


def test_ascii_stl_endsolid_bare():
    data = _ascii_stl(['solid cube'] + _FACET + ['endsolid'])

    tris = read_stl(BytesIO(data))

    assert tris.shape == (1, 4, 3)
    assert np.array_equal(tris[0], _TRI)


def test_ascii_stl_crlf():
    data = _ascii_stl(['solid cube'] + _FACET + ['endsolid cube'])
    data = data.replace(b'\n', b'\r\n')

    tris = read_stl(BytesIO(data))

    assert tris.shape == (1, 4, 3)
    assert np.array_equal(tris[0], _TRI)


def test_ascii_stl_malformed_line():
    data = _ascii_stl(['solid cube', 'facet junk'] + _FACET[1:] + ['endsolid'])

    with pytest.raises(ValueError):
        read_stl(BytesIO(data))


def test_ascii_stl_incomplete():
    data = _ascii_stl(['solid cube'] + _FACET)

    with pytest.raises(ValueError, match='Incomplete file'):
        read_stl(BytesIO(data))
