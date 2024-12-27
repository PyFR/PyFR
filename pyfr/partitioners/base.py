from collections import namedtuple

import numpy as np

from pyfr.nputil import iter_struct
from pyfr.progress import NullProgressSequence


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
    def __init__(self, partwts, elewts=None, nsubeles=64, opts={}):
        self.partwts = partwts
        self.elewts = elewts
        self.nparts = len(partwts)
        self.nsubeles = nsubeles

        if not self.has_part_weights and len(set(partwts)) != 1:
            raise ValueError(f'Partitioner {self.name} does not support '
                             'per-partition weights')

        if elewts is None and not self.has_multiple_constraints:
            raise ValueError(f'Partitioner {self.name} does not support '
                             'balanced partitioning')

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
        edisps, efaces, ecurved = {}, {}, []

        # Read the data for each element type
        disp = 0
        for etype, einfo in sorted(mesh['eles'].items()):
            einfo = einfo['curved', 'faces'][()]
            efaces[etype] = einfo['faces']
            ecurved.append(einfo['curved'])

            # Note the displacement
            edisps[etype] = disp
            disp += len(einfo)

        # Create a map from cidx element types to their displacements
        cdisps = np.empty(len(codec), dtype=int)
        for etype, disp in edisps.items():
            for i in range(efaces[etype].shape[-1]):
                cdisps[codec.index(f'eles/{etype}/{i}')] = disp

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

        # Stack all of the curved element arrays together
        ecurved = np.concatenate(ecurved)

        # Eliminate duplicates and return
        return conn, ecurved, edisps, cdisps

    def _get_elewts_fn(self, edisps):
        # If we have an element weighting table then use it
        if self.elewts is not None:
            elewts = self.elewts
        # Else, use multiple constraints for a balanced partitioning
        else:
            elewts = dict(zip(edisps, np.eye(len(edisps), dtype=int)))

        # Unpack the weights and displacement dictionaries
        elewts = np.array([elewts[etype] for etype in edisps])
        edisps = np.array(list(edisps.values()))

        def wts(e):
            return elewts[np.searchsorted(edisps, e, side='right') - 1]

        return wts

    @staticmethod
    def _construct_graph(con, elewts_fn, exwts={}):
        # Construct the dual graph
        con = np.vstack([con, con[:, ::-1]])

        # Sort by the left hand side
        con = con[np.argsort(con[:, 0])]

        # Left and right hand side global element numbers
        lhs, rhs = con.T

        # Compute vertex offsets
        vtab = (lhs[1:] != lhs[:-1]).nonzero()[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))

        # Compute the vertex number to global element number map
        vemap = lhs[vtab[:-1]]

        # Prepare vertex weights
        vwts = elewts_fn(vemap)
        for i, j in exwts.items():
            vwts[np.searchsorted(vemap, i)] = j

        # Ensure vwts is always two dimensional
        vwts = vwts.reshape(len(vwts), -1)

        # Prepare the edges and their weights
        etab = np.searchsorted(vemap, rhs)
        ewts = np.ones_like(etab)

        return Graph(vtab, etab, vwts, ewts), vemap

    def _partition_graph(self, graph, partwts):
        pass

    @classmethod
    def _merge_con(cls, pmerge, l, r):
        if l == r:
            return

        if l in pmerge and r in pmerge:
            cls._merge_con(pmerge, pmerge[l], pmerge[r])
        elif l in pmerge:
            cls._merge_con(pmerge, pmerge[l], r)
        elif r in pmerge:
            cls._merge_con(pmerge, l, pmerge[r])
        else:
            pmerge[l] = r

    @staticmethod
    def _resolve_merge(pmerge):
        nmerge = {}

        for l, r in pmerge.items():
            while (rr := pmerge.get(r)) is not None:
                r = rr

            nmerge[l] = r

        return nmerge

    @classmethod
    def _group_periodic_eles(cls, mesh, con, cdisps, elewts_fn):
        cdtype = [('l', np.int64), ('r', np.int64)]
        pmerge, pidx = {}, []

        # View it as a structured array
        conv = con.view(cdtype).squeeze()

        # Obtain the periodic connectivity info
        pfaces = mesh['periodic'] if 'periodic' in mesh else {}

        # Loop over periodic faces
        for k, pcon in pfaces.items():
            # Flatten the periodic connectivity array
            pcon = pcon[()].reshape(-1)

            # Convert from local to global element numbers
            pcon = cdisps[pcon['cidx']] + pcon['off']
            pcon = pcon.reshape(-1, 2)

            # Sort so lhs <= rhs
            pcon.sort()

            # Locate these entries in the connectivity array
            pidx.append(np.searchsorted(conv, pcon.view(cdtype).squeeze()))

            # Determine which elements require mering
            for l, r in iter_struct(pcon.reshape(-1, 2)):
                cls._merge_con(pmerge, l, r)

        if pmerge:
            pmerge = cls._resolve_merge(pmerge)

            # Eliminate connectivity entries associated with periodic faces
            con = np.delete(con, np.hstack(pidx), axis=0)

            mfrom = np.array(list(pmerge))
            mto = np.array(list(pmerge.values()))

            # Merge the associated elements on the left
            ls = np.searchsorted(con[:, 0], mfrom, side='left')
            le = np.searchsorted(con[:, 0], mfrom, side='right')
            for s, e, t in zip(ls, le, mto):
                con[s:e, 0] = t

            # Merge the associated elements on the right
            rix = np.argsort(con[:, 1])
            rs = np.searchsorted(con[:, 1], mfrom, side='left', sorter=rix)
            re = np.searchsorted(con[:, 1], mfrom, side='right', sorter=rix)
            for s, e, t in zip(rs, re, mto):
                con[rix[s:e], 1] = t

        # Tally up the weights for the merged elements
        exwts = {j: elewts_fn(j) for j in set(pmerge.values())}
        for i, j in pmerge.items():
            exwts[j] = exwts[j] + elewts_fn(i)

        return con, exwts, pmerge

    @staticmethod
    def _ungroup_periodic_eles(pmerge, vemap, vparts):
        # For each merged element identify its partition number
        pparts = vparts[np.searchsorted(vemap, list(pmerge.values()))]

        # With this we can unmerge the elements and update the arrays
        vemap = np.concatenate((vemap, list(pmerge)))
        vparts = np.concatenate((vparts, pparts))

        # Sort by vemap to give the global element number partition array
        return vparts[np.argsort(vemap)]

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

        # Sort by partition number, type, internal, and if curved or not
        pidx = np.lexsort((ecurved, internal, petype, vparts))

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

    def partition(self, mesh, progress=NullProgressSequence):
        # Construct the global connectivity array
        with progress.start('Construct global connectivity array'):
            con, ecurved, edisps, cdisps = self.construct_global_con(mesh)

        # Obtain the global element number weighting function
        elewts_fn = self._get_elewts_fn(edisps)

        # Merge periodic elements
        with progress.start('Group periodic elements'):
            pmcon, exwts, pmerge = self._group_periodic_eles(mesh, con,
                                                             cdisps, elewts_fn)

        # Obtain the dual graph for this mesh
        with progress.start('Construct graph'):
            graph, vemap = self._construct_graph(pmcon, elewts_fn, exwts=exwts)

        # Partition the graph
        with progress.start('Partition graph'):
            if self.nparts > 1:
                vparts = self._partition_graph(graph, self.partwts)

                if (n := len(np.unique(vparts))) != self.nparts:
                    raise RuntimeError(f'Partitioner error: mesh has {n} '
                                       f'parts versus goal of {self.nparts}')
            else:
                vparts = np.zeros(len(vemap), dtype=np.int32)

        # Unmerge periodic elements
        with progress.start('Ungroup periodic elements'):
            vparts = self._ungroup_periodic_eles(pmerge, vemap, vparts)

        # Construct the partitioning data
        with progress.start('Construct partitioning'):
            pinfo = self.construct_partitioning(mesh, ecurved, edisps, con,
                                                vparts)

        return pinfo
