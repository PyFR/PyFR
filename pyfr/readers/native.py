# -*- coding: utf-8 -*-

from collections.abc import Mapping
from functools import cached_property
import os
import re

import h5py
import numpy as np

from pyfr.util import memoize


class NativeReader(Mapping):
    def __init__(self, fname):
        self.fname = os.path.abspath(fname)
        self._file = h5py.File(fname, 'r')

    def __contains__(self, aname):
        if isinstance(aname, str):
            return aname in self._keys
        else:
            p, q = aname
            return p in self._keys and q in self._file[p].attrs

    def __getitem__(self, aname):
        if isinstance(aname, str):
            ret = self._file[aname]

            if ret.shape == ():
                ret = ret[()]
            else:
                ret = np.array(ret)

            return ret.decode() if isinstance(ret, bytes) else ret
        else:
            return self._file[aname[0]].attrs[aname[1]]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    @cached_property
    def _keys(self):
        keys = set()

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.add(name)

        self._file.visititems(visitor)

        return keys

    @memoize
    def array_info(self, prefix):
        # Entries in the file which start with the prefix
        names = [n for n in self if n.startswith(prefix)]

        # Distinct element types in the file
        ftypes = sorted({n.split('_')[1] for n in names})

        # Highest partition number in the file
        fmaxpn = max(int(re.search(r'\d+$', n)[0]) for n in names)

        # Extract array information
        info = {}
        for i in range(fmaxpn + 1):
            for et in ftypes:
                try:
                    n = f'{prefix}_{et}_p{i}'

                    info[n] = (et, self._file.get(n).shape)
                except AttributeError:
                    pass

        return info

    @memoize
    def partition_info(self, prefix):
        ai = self.array_info(prefix)

        # Number of partitions in the mesh
        npr = max(int(re.search(r'\d+$', k)[0]) for k in ai) + 1

        # Element types in the mesh
        etypes = {v[0] for v in ai.values()}

        # Compute the number of elements of each type in each partition
        nep = {et: [0]*npr for et in etypes}

        for k, v in ai.items():
            nep[v[0]][int(re.search(r'\d+$', k)[0])] = v[1][1]

        return nep
