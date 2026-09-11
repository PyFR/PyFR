import struct

import numpy as np


def read_stl(f):
    if isinstance(f, str):
        with open(f, 'rb') as fh:
            return read_stl(fh)

    # Read the 80-byte header plus the 4-byte triangle count. The STL spec
    # permits the binary header to contain arbitrary bytes, including the
    # word 'solid', so the leading bytes alone cannot be used to
    # distinguish binary from ASCII files. Instead, valid binary files are
    # exactly 84 + 50*ntri bytes long; use this invariant to disambiguate.
    head = f.read(84)

    binary = head[:5] != b'solid'
    if not binary and len(head) == 84:
        f.seek(0, 2)
        binary = 84 + 50*struct.unpack('<I', head[80:84])[0] == f.tell()

    f.seek(0)

    # Binary
    if binary:
        f.seek(80)
        ntri = np.fromfile(f, dtype='<i4', count=1)[0]
        tris = np.fromfile(f, dtype='(4,3)<f4, <i2', count=ntri)

        return np.ascontiguousarray(tris['f0'])

    # ASCII
    stlf = f.read().replace(b'\r\n', b'\n')
    stlit = (l.split() for l in stlf.split(b'\n')[1:])
    tris = []

    while (l := next(stlit, None)):
        match l:
            case [b'endsolid', *_]:
                break
            case [b'facet', b'normal', ni, nj, nk]:
                tris.append([float(ni), float(nj), float(nk)])

                if next(stlit, None) != [b'outer', b'loop']:
                    raise ValueError('Expected "outer loop"')

                for i in range(3):
                    v = next(stlit, None)
                    if v is None or v[0] != b'vertex':
                        raise ValueError('Expected "vertex"')

                    tris.append([float(vi) for vi in v[1:]])

                if next(stlit, None) != [b'endloop']:
                    raise ValueError('Expected "endloop"')
                if next(stlit, None) != [b'endfacet']:
                    raise ValueError('Expected "endfacet"')
            case _:
                raise ValueError(f'Unexpected line in ASCII STL file: {l}')
    else:
        raise ValueError('Incomplete file')

    return np.array(tris, dtype=np.float32).reshape(-1, 4, 3)
