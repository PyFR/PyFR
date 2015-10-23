# -*- coding: utf-8 -*-

from collections import Mapping, OrderedDict
import re

import h5py
import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import lazyprop, subclasses


class NativeReader(Mapping):
    def __init__(self, fname):
        self._file = h5py.File(fname, 'r')

    def __getitem__(self, aname):
        ret = self._file[aname]

        if ret.shape == ():
            ret = ret.value
        else:
            ret = np.array(ret)

        return ret.decode() if isinstance(ret, bytes) else ret

    def __iter__(self):
        return iter(self._file)

    def __len__(self):
        return len(self._file)

    def list_archives_startswith(self, prefix):
        return [name for name in self if name.startswith(prefix)]

    @property
    def spt_files(self):
        return self.list_archives_startswith('spt')

    @property
    def soln_files(self):
        return self.list_archives_startswith('soln')

    @lazyprop
    def array_info(self):
        # Retrieve list of {mesh, solution} array names, and set name prefix.
        if self.spt_files:
            ls_files = self.spt_files
            prfx = 'spt'
        elif self.soln_files:
            ls_files = self.soln_files
            prfx = 'soln'
        else:
            raise RuntimeError('"%s" does not contain solution or shape point '
                               'files' % (self.fname))

        # Element types known to PyFR
        eletypes = [b.name for b in subclasses(BaseShape)
                    if hasattr(b, 'name')]

        # Extract array information; FIXME
        info = OrderedDict()
        prt = 0

        while len(info) < len(ls_files):
            for et in eletypes:
                name = '%s_%s_p%d' % (prfx, et, prt)

                if name in ls_files:
                    info[name] = (et, self._file.get(name).shape)

            prt += 1

        return info

    @lazyprop
    def partition_info(self):
        ai = self.array_info

        # Number of partitions in the mesh
        npr = max(int(re.search(r'\d+$', k).group(0)) for k in ai) + 1

        # Element types in the mesh
        etypes = set(v[0] for v in ai.values())

        # Compute the number of elements of each type in each partition
        nep = {et: [0]*npr for et in etypes}

        for k, v in ai.items():
            nep[v[0]][int(re.search(r'\d+$', k).group(0))] = v[1][1]

        return nep
