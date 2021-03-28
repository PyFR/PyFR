# -*- coding: utf-8 -*-

from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np

from pyfr.inifile import Inifile


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


class BasePartitioner(object):
    # Approximate element weightings at each polynomial order
    elewtsmap = {
        1: {'quad': 5, 'tri': 3, 'tet': 3, 'hex': 9, 'pri': 6, 'pyr': 4},
        2: {'quad': 6, 'tri': 3, 'tet': 3, 'hex': 16, 'pri': 8, 'pyr': 5},
        3: {'quad': 6, 'tri': 3, 'tet': 3, 'hex': 24, 'pri': 10, 'pyr': 6},
        4: {'quad': 7, 'tri': 3, 'tet': 3, 'hex': 30, 'pri': 12, 'pyr': 7},
        5: {'quad': 7, 'tri': 3, 'tet': 3, 'hex': 34, 'pri': 13, 'pyr': 7},
        6: {'quad': 8, 'tri': 3, 'tet': 3, 'hex': 38, 'pri': 14, 'pyr': 8}
    }

    def __init__(self, partwts, elewts=None, order=None, opts={}):
        self.partwts = partwts
        self.nparts = len(partwts)

        if elewts is not None:
            self.elewts = elewts
        elif order is not None:
            self.elewts = self.elewtsmap[min(order, max(self.elewtsmap))]
        else:
            raise ValueError('Must provide either elewts or order')

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
        pinf = mesh.partition_info('spt')

        # Shape points, linear flags, element offsets, and remapping table
        spts = defaultdict(list)
        linf = defaultdict(list)
        offs = defaultdict(dict)
        rnum = defaultdict(dict)

        for en, pn in pinf.items():
            for i, n in enumerate(pn):
                if n > 0:
                    offs[en][i] = off = sum(s.shape[1] for s in spts[en])
                    spts[en].append(mesh[f'spt_{en}_p{i}'])
                    rnum[en].update(((i, j), (0, off + j)) for j in range(n))

                    try:
                        linf[en].append(mesh[f'spt_{en}_p{i}', 'linear'])
                    except KeyError:
                        linf[en].append(np.full(spts[en][-1].shape[1], False))

        def offset_con(con, pr):
            con = con.copy().astype('U4,i4,i1,i2')

            for en, pn in pinf.items():
                if pn[pr] > 0:
                    con['f1'][np.where(con['f0'] == en)] += offs[en][pr]

            return con

        # Connectivity
        intcon, mpicon, bccon = [], {}, defaultdict(list)

        for f in mesh:
            if (mi := re.match(r'con_p(\d+)$', f)):
                intcon.append(offset_con(mesh[f], int(mi.group(1))))
            elif (mm := re.match(r'con_p(\d+)p(\d+)$', f)):
                l, r = int(mm.group(1)), int(mm.group(2))
                lcon = offset_con(mesh[f], l)

                if (r, l) in mpicon:
                    rcon = mpicon.pop((r, l))
                    intcon.append(np.vstack([lcon, rcon]))
                else:
                    mpicon[l, r] = lcon
            elif (bc := re.match(r'bcon_(.+?)_p(\d+)$', f)):
                name, l = bc.group(1), int(bc.group(2))
                bccon[name].append(offset_con(mesh[f], l))

        # Output data type
        dtype = 'U4,i4,i1,i2'

        # Concatenate these arrays to from the new mesh
        newmesh = {'con_p0': np.hstack(intcon).astype(dtype)}

        for en in spts:
            newmesh[f'spt_{en}_p0'] = np.hstack(spts[en])
            newmesh[f'spt_{en}_p0', 'linear'] = np.hstack(linf[en])

        for k, v in bccon.items():
            newmesh[f'bcon_{k}_p0'] = np.hstack(v).astype(dtype)

        return newmesh, rnum

    def _combine_soln_parts(self, soln, prefix):
        newsoln = defaultdict(list)

        for f, (en, shape) in soln.array_info(prefix).items():
            newsoln[f'{prefix}_{en}_p0'].append(soln[f])

        newsoln = {k: np.dstack(v) for k, v in newsoln.items()}
        newsoln['config'] = soln['config']
        newsoln['stats'] = soln['stats']

        return newsoln

    def _construct_graph(self, mesh):
        # Edges of the dual graph
        con = mesh['con_p0']
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
        vetimap = lhs[vtab[:-1]].tolist()
        etivmap = {k: v for v, k in enumerate(vetimap)}

        # Prepare the list of edges for each vertex
        etab = np.array([etivmap[r] for r in rhs.tolist()])

        # Prepare the list of vertex and edge weights
        vwts = np.array([self.elewts[t] for t, i in vetimap])
        ewts = np.ones_like(etab)

        return Graph(vtab, etab, vwts, ewts), vetimap

    def _partition_graph(self, graph, partwts):
        pass

    def _renumber_verts(self, mesh, vetimap, vparts):
        vpartmap = dict(zip(vetimap, vparts))
        bndeti = set()

        # Identify vertices whose edges cross partition boundaries
        for l, r in zip(*mesh['con_p0'][['f0', 'f1']].tolist()):
            l, r = tuple(l), tuple(r)

            if vpartmap[l] != vpartmap[r]:
                bndeti |= {l, r}

        # Move these exterior vertices to the start of the list
        nvetimap, nvparts = list(bndeti), [vpartmap[eti] for eti in bndeti]

        # Next, process the internal vertices
        for f in mesh:
            if isinstance(f, str) and f.startswith('spt'):
                etype = f.split('_')[1]

                # Start with curved elements, followed by linear elements
                for i in np.argsort(mesh[f, 'linear']):
                    if (etype, i) not in bndeti:
                        nvetimap.append((etype, i))
                        nvparts.append(vpartmap[etype, i])

        return nvetimap, nvparts

    def _partition_spts(self, mesh, vetimap, vparts):
        spt_px = defaultdict(list)
        lin_px = defaultdict(list)

        for (etype, eidxg), part in zip(vetimap, vparts):
            f = f'spt_{etype}_p0'

            spt_px[etype, part].append(mesh[f][:, eidxg, :])
            lin_px[etype, part].append(mesh[f, 'linear'][eidxg])

        newmesh = {}
        for etype, pn in spt_px:
            f = f'spt_{etype}_p{pn}'

            newmesh[f] = np.array(spt_px[etype, pn]).swapaxes(0, 1)
            newmesh[f, 'linear'] = np.array(lin_px[etype, pn])

        return newmesh

    def _partition_soln(self, soln, prefix, vetimap, vparts):
        soln_px = defaultdict(list)
        for (etype, eidxg), part in zip(vetimap, vparts):
            f = f'{prefix}_{etype}_p0'

            soln_px[etype, part].append(soln[f][..., eidxg])

        return {f'{prefix}_{etype}_p{pn}': np.dstack(v)
                for (etype, pn), v in soln_px.items()}

    def _partition_con(self, mesh, vetimap, vparts):
        con_px = defaultdict(list)
        con_pxpy = defaultdict(list)
        bcon_px = defaultdict(list)

        # Global-to-local element index map
        eleglmap = {}
        pcounter = Counter()

        for (etype, eidxg), part in zip(vetimap, vparts):
            eleglmap[etype, eidxg] = (part, pcounter[etype, part])
            pcounter[etype, part] += 1

        # Generate the face connectivity
        for l, r in zip(*mesh['con_p0'].tolist()):
            letype, leidxg, lfidx, lflags = l
            retype, reidxg, rfidx, rflags = r

            lpart, leidxl = eleglmap[letype, leidxg]
            rpart, reidxl = eleglmap[retype, reidxg]

            conl = (letype, leidxl, lfidx, lflags)
            conr = (retype, reidxl, rfidx, rflags)

            if lpart == rpart:
                con_px[lpart].append([conl, conr])
            else:
                con_pxpy[lpart, rpart].append(conl)
                con_pxpy[rpart, lpart].append(conr)

        # Generate boundary conditions
        for f in filter(lambda f: isinstance(f, str), mesh):
            if (m := re.match('bcon_(.+?)_p0$', f)):
                lhs = mesh[f].tolist()

                for lpetype, leidxg, lfidx, lflags in lhs:
                    lpart, leidxl = eleglmap[lpetype, leidxg]
                    conl = (lpetype, leidxl, lfidx, lflags)

                    bcon_px[m.group(1), lpart].append(conl)

        # Output data type
        dtype = 'S4,i4,i1,i2'

        # Output
        con = {}

        for px, v in con_px.items():
            con[f'con_p{px}'] = np.array(v, dtype=dtype).T

        for (px, py), v in con_pxpy.items():
            con[f'con_p{px}p{py}'] = np.array(v, dtype=dtype)

        for (etype, px), v in bcon_px.items():
            con[f'bcon_{etype}_p{px}'] = np.array(v, dtype=dtype)

        return con, eleglmap

    def partition(self, mesh):
        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']

        # Combine any pre-existing partitions
        mesh, rnum = self._combine_mesh_parts(mesh)

        # Obtain the dual graph for this mesh
        graph, vetimap = self._construct_graph(mesh)

        # Partition the graph
        if self.nparts > 1:
            vparts = self._partition_graph(graph, self.partwts).tolist()
        else:
            vparts = [0]*len(vetimap)

        # Renumber vertices
        vetimap, vparts = self._renumber_verts(mesh, vetimap, vparts)

        # Partition the connectivity portion of the mesh
        newmesh, eleglmap = self._partition_con(mesh, vetimap, vparts)

        # Handle the shape points
        newmesh.update(self._partition_spts(mesh, vetimap, vparts))

        # Update the renumbering table
        for etype, emap in rnum.items():
            for k, (pidx, eidx) in emap.items():
                emap[k] = eleglmap[etype, eidx]

        # Generate a new UUID for the mesh
        newmesh['mesh_uuid'] = newuuid = str(uuid.uuid4())

        # Build the solution converter
        def partition_soln(soln):
            # Check the UUID
            if curruuid != soln['mesh_uuid']:
                raise ValueError('Mismatched solution/mesh')

            # Obtain the prefix
            prefix = Inifile(soln['stats']).get('data', 'prefix')

            # Combine any pre-existing partitions
            soln = self._combine_soln_parts(soln, prefix)

            # Partition
            if self.nparts > 1:
                newsoln = self._partition_soln(soln, prefix, vetimap, vparts)
            else:
                newsoln = soln

            # Handle the metadata
            newsoln['config'] = soln['config']
            newsoln['stats'] = soln['stats']
            newsoln['mesh_uuid'] = newuuid

            return newsoln

        return newmesh, rnum, partition_soln
