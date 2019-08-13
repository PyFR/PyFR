# -*- coding: utf-8 -*-

from collections import Mapping, OrderedDict
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
        return aname in self._file

    def __getitem__(self, aname):
        ret = self._file[aname]

        if ret.shape == ():
            ret = ret[()]
        else:
            ret = np.array(ret)

        return ret.decode() if isinstance(ret, bytes) else ret

    def __iter__(self):
        return iter(self._file)

    def __len__(self):
        return len(self._file)

    @memoize
    def array_info(self, prefix):
        # Entries in the file which start with the prefix
        names = [n for n in self if n.startswith(prefix)]

        # Distinct element types in the file
        ftypes = sorted({n.split('_')[1] for n in names})

        # Highest partition number in the file
        fmaxpn = max(int(re.search(r'\d+$', n).group(0)) for n in names)

        # Extract array information
        info = OrderedDict()
        for i in range(fmaxpn + 1):
            for et in ftypes:
                try:
                    n = '{0}_{1}_p{2}'.format(prefix, et, i)

                    info[n] = (et, self._file.get(n).shape)
                except AttributeError:
                    pass

        return info

    @memoize
    def partition_info(self, prefix):
        ai = self.array_info(prefix)

        # Number of partitions in the mesh
        npr = max(int(re.search(r'\d+$', k).group(0)) for k in ai) + 1

        # Element types in the mesh
        etypes = {v[0] for v in ai.values()}

        # Compute the number of elements of each type in each partition
        nep = {et: [0]*npr for et in etypes}

        for k, v in ai.items():
            nep[v[0]][int(re.search(r'\d+$', k).group(0))] = v[1][1]

        return nep
