from collections import namedtuple

import numpy as np

from pyfr.nputil import iter_struct, range_offsets
from pyfr.progress import NullProgressSequence
from pyfr.util import DisjointSet


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


def write_partitioning(mesh, pname, pinfo):
    ppath = f'partitionings/{pname}'
    (partitioning, pregions), (neighbours, nregions) = pinfo

    if ppath in mesh:
        mesh[f'{ppath}/eles'][:] = partitioning
        del mesh[f'{ppath}/neighbours']
    else:
        mesh[f'{ppath}/eles'] = partitioning

    mesh[f'{ppath}/neighbours'] = neighbours
    mesh[f'{ppath}/eles'].attrs['regions'] = pregions
    mesh[f'{ppath}/neighbours'].attrs['regions'] = nregions


class BasePartitioner:
    def __init__(self, partwts, elewts='balanced', tagwts={}, opts={}):
        self.partwts = partwts
        self.elewts = elewts
        self.tagwts = tagwts
        self.nparts = len(partwts)

        if not self.has_part_weights and len(set(partwts)) != 1:
            raise ValueError(f'Partitioner {self.name} does not support '
                             'per-partition weights')

        if elewts == 'balanced' and not self.has_multiple_constraints:
            raise ValueError(f'Partitioner {self.name} does not support '
                             'balanced partitioning')

        if tagwts == 'balanced' and not self.has_multiple_constraints:
            raise ValueError(f'Partitioner {self.name} does not support '
                             'balanced tag partitioning')

        # Parse the options list
        self.opts = {}
        for k, v in dict(self.dflt_opts, **opts).items():
            if k in self.int_opts:
                self.opts[k] = int(v)
            elif k in self.enum_opts:
                self.opts[k] = self.enum_opts[k][v]
            else:
                raise ValueError('Invalid partitioner option')

    @staticmethod
    def construct_global_con(mesh):
        codec = [c.decode() for c in mesh['codec']]
        efaces, ecurved, etags = {}, [], []

        # Read the data for each element type
        for etype, einfo in sorted(mesh['eles'].items()):
            einfo = einfo['curved', 'faces', 'tags'][()]
            efaces[etype] = einfo['faces']
            ecurved.append(einfo['curved'])
            etags.append(einfo['tags'])

        # Build per-etype displacements
        edisps, _ = range_offsets(efaces.items())

        # Create a map from cidx element types to their displacements
        cdisps = np.empty(len(codec), dtype=int)
        for etype, disp in edisps.items():
            for i in range(efaces[etype].shape[-1]):
                cdisps[codec.index(f'eles/{etype}/face/{i}')] = disp

        # Construct the global element-element connectivity array
        conn = []
        for etype, einfo in efaces.items():
            # Allocate the element-element connectivity array
            econ = np.empty((*einfo.shape, 2), dtype=int)

            # Compute our element numbers
            disp = edisps[etype]
            econ[:, :, 0] = np.arange(disp, disp + len(einfo))[:, None]

            # Next, prune element-boundary connectivity
            eidx = einfo['off'] >= 0
            einfo, econ = einfo[eidx], econ[eidx]

            # Compute the numbers of the elements we are connected to
            econ[:, 1] = cdisps[einfo['cidx']] + einfo['off']

            conn.append(econ)

        # Stack all of the global connectivity arrays together
        conn = np.vstack(conn)

        # Sort the connectivity pairs to help identify duplicates
        conn.sort()

        # Eliminate duplicates
        conn = conn[np.lexsort(conn.T[::-1])[::2]]

        # Stack all of the curved element and tag arrays together
        ecurved = np.concatenate(ecurved)
        etags = np.concatenate(etags)

        return conn, ecurved, etags, edisps, cdisps

    def _get_elewts_fn(self, edisps, etags, tagwts):
        # Use multiple constraints for a balanced partitioning
        if self.elewts == 'balanced':
            elewts = dict(zip(edisps, np.eye(len(edisps), dtype=int)))
        # Else, use the provided element weighting table
        else:
            elewts = self.elewts

        # Unpack the weights and displacement dictionaries
        elewts = np.array([elewts[etype] for etype in edisps])
        edisps = np.array(list(edisps.values()))

        # Determine the number of elements and tags in the mesh
        neles, ntags = len(etags), int(etags.max()).bit_length()

        # Balanced tag partitioning
        if tagwts == 'balanced' and ntags > 1:
            # Obtain the base set of weights
            etidx = np.searchsorted(edisps, np.arange(neles), side='right') - 1
            base = elewts[etidx].reshape(neles, -1)

            # Add in one constraint column per tag
            bits = np.arange(ntags, dtype=np.uint64)
            tmat = ((etags[:, None] >> bits) & 1).astype(int)

            # Concatenate
            fwts = np.hstack([base, tmat])

            def wts(e):
                return fwts[e]
        # Multiplicative tag partitioning; scale by product of tag weights
        else:
            if tagwts and ntags > 1:
                tmult = np.ones(neles, dtype=int)
                for bit, weight in tagwts.items():
                    idxs = ((etags >> np.uint64(bit)) & 1).astype(bool)
                    tmult[idxs] *= weight
            else:
                tmult = None

            def wts(e):
                w = elewts[np.searchsorted(edisps, e, side='right') - 1]
                if tmult is not None:
                    m = tmult[e]
                    w *= m[..., None] if np.ndim(w) > 1 else m

                return w

        return wts

    @staticmethod
    def _construct_graph(con, emap, elewts):
        # Construct the dual graph of the merged elements
        con = emap[np.vstack([con, con[:, ::-1]])]

        # Remove connections internal to a merged element
        con = con[con[:, 0] != con[:, 1]]

        # Sort by the left hand side, removing any parallel edges
        con = np.sort(con[:, 0]*len(emap) + con[:, 1])
        con = con[np.r_[True, con[1:] != con[:-1]]]

        # Left and right hand side global element numbers
        lhs, rhs = np.divmod(con, len(emap))

        # Compute vertex offsets
        vtab = (lhs[1:] != lhs[:-1]).nonzero()[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))

        # Compute the vertex number to global element number map
        vemap = lhs[vtab[:-1]]

        # Compute the global element number to vertex number map
        evmap = np.searchsorted(vemap, emap)

        # Weight each vertex by the total weight of its merged elements
        elewts = elewts.reshape(len(emap), -1)
        vwts = np.zeros((len(vemap), elewts.shape[1]), dtype=elewts.dtype)
        np.add.at(vwts, evmap, elewts)

        # Prepare the edges and their weights
        etab = np.searchsorted(vemap, rhs)
        ewts = np.ones_like(etab)

        return Graph(vtab, etab, vwts, ewts), evmap

    def _partition_graph(self, graph, partwts):
        pass

    @staticmethod
    def _group_periodic_eles(mesh, cdisps, neles):
        ds = DisjointSet()

        # Obtain the periodic connectivity info
        pfaces = mesh['periodic'] if 'periodic' in mesh else {}

        # Loop over periodic faces
        for k, pcon in pfaces.items():
            # Flatten the periodic connectivity array
            pcon = pcon[()].reshape(-1)

            # Convert from local to global element numbers
            pcon = cdisps[pcon['cidx']] + pcon['off']

            # Determine which elements require merging
            for l, r in iter_struct(pcon.reshape(-1, 2)):
                ds.union(l, r)

        # Map each element to the representative of its merged group
        emap = np.arange(neles)
        for i, j in ds.merges().items():
            emap[i] = j

        return emap

    @staticmethod
    def _analyse_parts(nparts, con, vparts):
        neighbours = [[] for i in range(nparts)]

        # Map element numbers to partitions numbers in the connectivity array
        vpcon = vparts[con]

        # Identify inter-partition connectivity
        ipartcon = vpcon[:, 0] != vpcon[:, 1]

        # Use this to mark elements which are on partition boundaries
        internal = np.ones(len(vparts), dtype=bool)
        internal[np.unique(con[ipartcon, 0])] = False
        internal[np.unique(con[ipartcon, 1])] = False

        # Next, sort the partition connectivity array along both axes
        vpcon.sort()
        vpcon = vpcon[np.lexsort(vpcon.T)]

        # With this, identify unique pairings
        pidx = np.searchsorted(vpcon[:, 1], np.arange(1, nparts))
        for i, vpicon in enumerate(np.split(vpcon[:, 0], pidx)):
            for j in iter_struct(np.unique(vpicon)):
                if i != j:
                    neighbours[i].append(j)
                    neighbours[j].append(i)

        # Construct and sort the neighbours array
        neighbours = [np.sort(np.array(n)) for n in neighbours]

        return neighbours, internal

    @classmethod
    def construct_partitioning(cls, mesh, ecurved, edisps, con, vparts):
        nparts = vparts.max() + 1
        etypes = np.arange(len(edisps))
        edisps = list(edisps.values())[1:]

        # Analyse the partitioning
        neighbours, internal = cls._analyse_parts(nparts, con, vparts)

        # Put the neighbours data into canonical form
        nregions = np.cumsum([len(n) for n in neighbours])
        nregions = np.concatenate(([0], nregions))
        neighbours = np.concatenate(neighbours)

        # Allocate the main partitioning array
        peidx = np.empty(len(vparts), dtype=np.int64)
        for p in np.split(peidx, edisps):
            p[:] = np.arange(len(p))

        # Also note the type of each element in the partitioning array
        petype = np.empty(len(vparts), dtype=np.int8)
        for i, p in enumerate(np.split(petype, edisps)):
            p[:] = i

        # Sort by partition number, type, internal, and if linear or not
        pidx = np.lexsort((~ecurved, internal, petype, vparts))

        # Apply this permutation to the various arrays
        peidx, petype, vparts = peidx[pidx], petype[pidx], vparts[pidx]

        # Determine region of peidx associated with each partition
        pregions = np.empty((nparts, len(etypes) + 1), dtype=np.int64)
        pregions[:, 0] = np.unique(vparts, return_index=True)[1]
        pregions[:, -1] = np.concatenate((pregions[1:, 0], [len(vparts)]))

        # Finally, fill in where the element type transitions are
        for i, p in enumerate(pregions):
            s, e = p[0], p[-1]
            p[:-1] = np.searchsorted(petype[s:e], etypes) + s

        return (peidx, pregions), (neighbours, nregions)

    def _resolve_tag_weights(self, mesh):
        # Balance tag weighting
        if self.tagwts == 'balanced':
            return 'balanced'
        # Weights provided; resolve names to bit masks
        elif self.tagwts:
            # Obtain the tags in the mesh
            tnames = [c[4:].decode()
                      for c in mesh['codec'] if c.startswith(b'tag/')]

            # Check no weights are missing
            if (missing := set(self.tagwts) - set(tnames)):
                raise ValueError('Unknown tag name(s) in tag weights: '
                                 f'{", ".join(sorted(missing))}')

            # Resolve
            return {tnames.index(n): w for n, w in self.tagwts.items()}
        # No weights provided
        else:
            return {}

    def partition(self, mesh, progress=NullProgressSequence()):
        # Construct the global connectivity array
        with progress.start('Construct global connectivity array'):
            info = self.construct_global_con(mesh)
            con, ecurved, etags, edisps, cdisps = info

        # Resolve tag weights
        tagwts = self._resolve_tag_weights(mesh)

        # Obtain the global element number weighting function
        elewts_fn = self._get_elewts_fn(edisps, etags, tagwts)

        # Merge periodic elements
        with progress.start('Group periodic elements'):
            emap = self._group_periodic_eles(mesh, cdisps, len(etags))

        # Obtain the dual graph for this mesh
        with progress.start('Construct graph'):
            elewts = elewts_fn(np.arange(len(etags)))
            graph, evmap = self._construct_graph(con, emap, elewts)

        # Partition the graph
        with progress.start('Partition graph'):
            if self.nparts > 1:
                vparts = self._partition_graph(graph, self.partwts)

                if (n := len(np.unique(vparts))) != self.nparts:
                    raise RuntimeError(f'Partitioner error: mesh has {n} '
                                       f'parts versus goal of {self.nparts}')
            else:
                vparts = np.zeros(len(graph.vtab) - 1, dtype=np.int32)

        # Unmerge periodic elements
        with progress.start('Ungroup periodic elements'):
            vparts = vparts[evmap]

        # Construct the partitioning data
        with progress.start('Construct partitioning'):
            pinfo = self.construct_partitioning(mesh, ecurved, edisps, con,
                                                vparts)

        return pinfo
