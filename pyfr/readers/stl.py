import numpy as np


def read_stl(f):
    if isinstance(f, str):
        with open(f, 'rb') as fh:
            data = fh.read()
    else:
        data = f.read()

    # Binary files are exactly 84 + 50*ntri bytes; the 80-byte header may
    # contain the word 'solid', so size (not the header text) is the sniff.
    is_bin = (len(data) >= 84
              and 84 + 50*int.from_bytes(data[80:84], 'little') == len(data))

    return _read_stl_bin(data) if is_bin else _read_stl_ascii(data)


def _read_stl_bin(data):
    ntri = np.frombuffer(data, dtype='<i4', count=1, offset=80)[0]
    tris = np.frombuffer(data, dtype='(4,3)<f4, <i2', count=ntri, offset=84)

    return np.ascontiguousarray(tris['f0'])


def _read_stl_ascii(data):
    stlit = (l.split() for l in data.replace(b'\r\n', b'\n').split(b'\n')[1:])
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
