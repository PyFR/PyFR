# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


class BasePartitioner(object):
    # Approximate element weighting table
    _ele_wts = {'quad': 3, 'tri': 2, 'tet': 2, 'hex': 6, 'pri': 4, 'pyr': 3}

    def __init__(self, partwts, opts={}):
        self.partwts = partwts

        # Parse the options list
        self.opts = {}
        for k, v in dict(self.dflt_opts, **opts).items():
            if k in self.int_opts:
                self.opts[k] = int(v)
            elif k in self.enum_opts:
                self.opts[k] = self.enum_opts[k][v]
            else:
                raise ValueError('Invalid partitioner option')

    def _combine_mesh_parts(self, mesh):
        # Get the per-partition element counts
        pinf = mesh.partition_info

        # Shape points and element number offsets
        spts = defaultdict(list)
        offs = defaultdict(dict)

        for en, pn in pinf.items():
            for i, n in enumerate(pn):
                if n > 0:
                    offs[en][i] = sum(s.shape[1] for s in spts[en])
                    spts[en].append(mesh['spt_{0}_p{1}'.format(en, i)])

        def offset_con(con, pr):
            con = con.copy().astype('U4,i4,i1,i1')

            for en, pn in pinf.items():
                if pn[pr] > 0:
                    con['f1'][np.where(con['f0'] == en)] += offs[en][pr]

            return con

        # Connectivity
        intcon, mpicon, bccon = [], {}, defaultdict(list)

        for f in mesh:
            mi = re.match(r'con_p(\d+)$', f)
            mm = re.match(r'con_p(\d+)p(\d+)$', f)
            bc = re.match(r'bcon_(.+?)_p(\d+)$', f)

            if mi:
                intcon.append(offset_con(mesh[f], int(mi.group(1))))
            elif mm:
                l, r = int(mm.group(1)), int(mm.group(2))
                lcon = offset_con(mesh[f], l)

                if (r, l) in mpicon:
                    rcon = mpicon.pop((r, l))
                    intcon.append(np.vstack([lcon, rcon]))
                else:
                    mpicon[l, r] = lcon
            elif bc:
                name, l = bc.group(1), int(bc.group(2))
                bccon[name].append(offset_con(mesh[f], l))

        # Output data type
        dtype = 'S4,i4,i1,i1'

        # Concatenate these arrays to from the new mesh
        newmesh = {'con_p0': np.hstack(intcon).astype(dtype)}

        for k, v in spts.items():
            newmesh['spt_{0}_p0'.format(k)] = np.hstack(v)

        for k, v in bccon.items():
            newmesh['bcon_{0}_p0'.format(k)] = np.hstack(v).astype(dtype)

        return newmesh

    def _combine_soln_parts(self, soln):
        newsoln = defaultdict(list)

        for f, (en, shape) in soln.array_info.items():
            newsoln['soln_{0}_p0'.format(en)].append(soln[f])

        newsoln = {k: np.dstack(v) for k, v in newsoln.items()}
        newsoln['config'] = soln['config']
        newsoln['stats'] = soln['stats']

        return newsoln

    def _construct_graph(self, mesh):
        # Edges of the dual graph
        con = mesh['con_p0'].astype('U4,i4,i1,i1')
        con = np.hstack([con, con[::-1]])

        # Sort by the left hand side
        idx = np.lexsort([con['f0'][0], con['f1'][0]])
        con = con[:, idx]

        # Left and right hand side element types/indicies
        lhs, rhs = con[['f0', 'f1']]

        # Compute vertex offsets
        vtab = np.where(lhs[1:] != lhs[:-1])[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))

        # Compute the element type/index to vertex number map
        vetimap = [tuple(lhs[i]) for i in vtab[:-1]]
        etivmap = {k: v for v, k in enumerate(vetimap)}

        # Prepare the list of edges for each vertex
        etab = np.array([etivmap[tuple(r)] for r in rhs])

        # Prepare the list of vertex and edge weights
        vwts = np.array([self._ele_wts[t] for t, i in vetimap])
        ewts = np.ones_like(etab)

        return Graph(vtab, etab, vwts, ewts), vetimap

    def _partition_graph(self, graph, partwts):
        pass

    def _partition_spts(self, mesh, vparts, vetimap):
        # Get the shape point arrays from the mesh
        spt_p0 = {}
        for f in mesh:
            if f.startswith('spt'):
                spt_p0[f.split('_')[1]] = mesh[f]

        # Partition the shape points
        spt_px = defaultdict(list)
        for (etype, eidxg), part in zip(vetimap, vparts):
            spt_px[etype, part].append(spt_p0[etype][:, eidxg, :])

        # Stack
        return {'spt_{0}_p{1}'.format(*k): np.array(v).swapaxes(0, 1)
                for k, v in spt_px.items()}

    def _partition_soln(self, soln, vparts, vetimap):
        # Get the solution arrays from the file
        soln_p0 = {}
        for f in soln:
            if f.startswith('soln'):
                soln_p0[f.split('_')[1]] = soln[f]

        # Partition the solutions
        soln_px = defaultdict(list)
        for (etype, eidxg), part in zip(vetimap, vparts):
            soln_px[etype, part].append(soln_p0[etype][..., eidxg])

        # Stack
        return {'soln_{0}_p{1}'.format(*k): np.dstack(v)
                for k, v in soln_px.items()}

    def _partition_con(self, mesh, vparts, vetimap):
        con_px = defaultdict(list)
        con_pxpy = defaultdict(list)
        bcon_px = defaultdict(list)

        # Global-to-local element index map
        eleglmap = defaultdict(list)
        pcounter = Counter()

        for (etype, eidxg), part in zip(vetimap, vparts):
            eleglmap[etype].append((part, pcounter[etype, part]))
            pcounter[etype, part] += 1

        # Generate the face connectivity
        for l, r in zip(*mesh['con_p0'].astype('U4,i4,i1,i1')):
            letype, leidxg, lfidx, lflags = l
            retype, reidxg, rfidx, rflags = r

            lpart, leidxl = eleglmap[letype][leidxg]
            rpart, reidxl = eleglmap[retype][reidxg]

            conl = (letype, leidxl, lfidx, lflags)
            conr = (retype, reidxl, rfidx, rflags)

            if lpart == rpart:
                con_px[lpart].append([conl, conr])
            else:
                con_pxpy[lpart, rpart].append(conl)
                con_pxpy[rpart, lpart].append(conr)

        # Generate boundary conditions
        for f in mesh:
            m = re.match('bcon_(.+?)_p0$', f)
            if m:
                lhs = mesh[f].astype('U4,i4,i1,i1')

                for lpetype, leidxg, lfidx, lflags in lhs:
                    lpart, leidxl = eleglmap[lpetype][leidxg]
                    conl = (lpetype, leidxl, lfidx, lflags)

                    bcon_px[m.group(1), lpart].append(conl)

        # Output data type
        dtype = 'S4,i4,i1,i1'

        # Output
        ret = {}

        for k, v in con_px.items():
            ret['con_p{0}'.format(k)] = np.array(v, dtype=dtype).T

        for k, v in con_pxpy.items():
            ret['con_p{0}p{1}'.format(*k)] = np.array(v, dtype=dtype)

        for k, v in bcon_px.items():
            ret['bcon_{0}_p{1}'.format(*k)] = np.array(v, dtype=dtype)

        return ret

    def partition(self, mesh):
        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']

        # Combine any pre-existing parititons
        mesh = self._combine_mesh_parts(mesh)

        # Perform the partitioning
        if len(self.partwts) > 1:
            # Obtain the dual graph for this mesh
            graph, vetimap = self._construct_graph(mesh)

            # Partition the graph
            vparts = self._partition_graph(graph, self.partwts)

            # Partition the connectivity portion of the mesh
            newmesh = self._partition_con(mesh, vparts, vetimap)

            # Handle the shape points
            newmesh.update(self._partition_spts(mesh, vparts, vetimap))
        # Short circuit
        else:
            newmesh = mesh

        # Generate a new UUID for the mesh
        newmesh['mesh_uuid'] = newuuid = str(uuid.uuid4())

        # Build the solution converter
        def partition_soln(soln):
            # Check the UUID
            if curruuid != soln['mesh_uuid']:
                raise ValueError('Mismatched solution/mesh')

            # Combine any pre-existing parititons
            soln = self._combine_soln_parts(soln)

            # Partition
            if len(self.partwts) > 1:
                newsoln = self._partition_soln(soln, vparts, vetimap)
            else:
                newsoln = soln

            # Handle the metadata
            newsoln['config'] = soln['config']
            newsoln['stats'] = soln['stats']
            newsoln['mesh_uuid'] = newuuid

            return newsoln

        return newmesh, partition_soln
