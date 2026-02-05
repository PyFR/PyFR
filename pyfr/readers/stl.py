import numpy as np


def read_stl(f):
    # ASCII
    if f.read(len(b'solid')) == b'solid':
        tris = []

        # Read and split the file
        stlf = f.read().replace(b'\r\n', b'\n')
        stlit = (l.split() for l in stlf.split(b'\n')[1:])

        # Parse
        while (l := next(stlit)):
            match l:
                case [b'endsolid']:
                    break
                case [b'facet', b'normal', ni, nj, nk]:
                    tris.append([float(ni), float(nj), float(nk)])

                    if next(stlit) != [b'outer', b'loop']:
                        raise ValueError('Expected "outer loop"')

                    for i in range(3):
                        v = next(stlit)
                        if v[0] != b'vertex':
                            raise ValueError('Expected "vertex"')

                        tris.append([float(vi) for vi in v[1:]])

                    if next(stlit) != [b'endloop']:
                        raise ValueError('Expected "endloop"')
                    if next(stlit) != [b'endfacet']:
                        raise ValueError('Expected "endfacet"')
        else:
            raise ValueError('Incomplete file')

        return np.array(tris, dtype=np.float32).reshape(-1, 4, 3)
    # Binary
    else:
        f.seek(80)
        ntri = np.fromfile(f, dtype='<i4', count=1)[0]
        tris = np.fromfile(f, dtype='(4,3)<f4, <i2', count=ntri)

        return np.ascontiguousarray(tris['f0'])
