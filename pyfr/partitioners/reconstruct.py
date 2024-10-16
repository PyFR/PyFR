from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.partitioners.base import BasePartitioner
from pyfr.progress import NullProgressSequence


def reconstruct_partitioning(mesh, soln, progress=NullProgressSequence):
    if mesh['mesh-uuid'][()] != soln['mesh-uuid'][()]:
        raise ValueError('Invalid solution for mesh')

    prefix = Inifile(soln['stats'][()].decode()).get('data', 'prefix')
    sparts = defaultdict(list)

    # Read the partition data from the solution
    for k in soln[prefix]:
        if (m := re.match(r'(p(?:\d+)-(\w+))-parts$', k)):
            parts = soln[f'{prefix}/{k}'][:]

            idxs = soln.get(f'{prefix}/{m[1]}-idxs')
            if idxs is None:
                idxs = np.arange(len(parts))

            sparts[m[2]].append((idxs, parts))

    # Group the data together by element type
    for etype, sp in sparts.items():
        idxs, parts = map(np.concatenate, zip(*sp))

        sparts[etype] = parts[np.argsort(idxs)]

    vparts = np.concatenate([p for _, p in sorted(sparts.items())])

    # Construct the global connectivity array
    with progress.start('Construct global connectivity array'):
        con, ecurved, edisps, _ = BasePartitioner.construct_global_con(mesh)

    # Ensure that the solution has not been subset
    if len(vparts) != len(ecurved):
        raise ValueError('Can not reconstruct partitioning from subsetted '
                         'solution')

    # Construct the partitioning data
    with progress.start('Construct partitioning'):
        pinfo = BasePartitioner.construct_partitioning(mesh, ecurved, edisps,
                                                       con, vparts)

    return pinfo
