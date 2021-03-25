# -*- coding: utf-8 -*-

from pyfr.inifile import Inifile
from collections import Counter, defaultdict, namedtuple
import re
import uuid

import numpy as np

from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


def _find_curved_eles(etype, eles, tol=1e-5):
    nspts, neles, ndims = eles.shape

    shape = subclass_where(BaseShape, name=etype)
    order = shape.order_from_nspts(nspts)
    basis = get_polybasis(etype, order, shape.std_ele(order - 1))

    lmodes = get_polybasis(etype, 2).degrees
    hmodes = [i for i, j in enumerate(basis.degrees) if j not in lmodes]

    ehmodes = basis.invvdm.T[hmodes] @ eles.reshape(nspts, -1)
    ehmodes = ehmodes.reshape(-1, neles, ndims)

    return np.any(np.abs(ehmodes) > tol, axis=(0, 2))


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

    def __init__(self, partwts, elewts=None, order=None, cfg=None, opts={}):
        self.partwts = partwts
        self.nparts = len(partwts)

        if elewts is not None:
            self.elewts = elewts
        elif order is not None:
            self.elewts = self.elewtsmap[min(order, max(self.elewtsmap))]
        else:
            raise ValueError('Must provide either elewts or order')

        if cfg is not None:
            self.cfg = Inifile.load(cfg)
        else:
            self.cfg = None

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

        # Shape points, element number offsets, and remapping table
        spts = defaultdict(list)
        offs = defaultdict(dict)
        rnum = defaultdict(dict)

        for en, pn in pinf.items():
            for i, n in enumerate(pn):
                if n > 0:
                    offs[en][i] = off = sum(s.shape[1] for s in spts[en])
                    spts[en].append(mesh[f'spt_{en}_p{i}'])
                    rnum[en].update(((i, j), (0, off + j)) for j in range(n))

        def offset_con(con, pr):
            con = con.copy().astype('U4,i4,i1,i2')

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
        dtype = 'U4,i4,i1,i2'

        # Concatenate these arrays to from the new mesh
        newmesh = {'con_p0': np.hstack(intcon).astype(dtype)}

        for k, v in spts.items():
            newmesh[f'spt_{k}_p0'] = np.hstack(v)

        for k, v in bccon.items():
            newmesh[f'bcon_{k}_p0'] = np.hstack(v).astype(dtype)

        return newmesh, rnum

    def _combine_soln_parts(self, soln):
        newsoln = defaultdict(list)

        for f, (en, shape) in soln.array_info('soln').items():
            newsoln[f'soln_{en}_p0'].append(soln[f])

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

        # Correct the weights to take into account turb generation.
        cfgsect = 'soln-plugin-turbulencegenerator'
        if self.cfg:
            if self.cfg.hassect('soln-plugin-turbulencegenerator'):
                lturbref = np.array(self.cfg.getliteral(cfgsect, 'lturbref'))
                ctr = np.array(self.cfg.getliteral(cfgsect, 'center'))
                inflow = np.array(self.cfg.getliteral(cfgsect, 'plane-dimensions'))
                Ubulkdir = self.cfg.getint(cfgsect, 'Ubulk-dir')

                t, _ = vetimap[0]
                ndims = mesh['spt_{}_p0'.format(t)].shape[-1]
                dirs = [i for i in range(ndims) if i != Ubulkdir]
                nel_affected = 0

                # determine the bounds of the box of eddies
                delta = np.zeros(ndims)
                delta[Ubulkdir] = lturbref[Ubulkdir]
                delta[dirs] = 0.5*inflow + lturbref[dirs]
                x0 = ctr - delta
                x1 = ctr + delta

                print('Taking into account turb. generation...')
                for t, i in vetimap:
                    pname = 'spt_{}_p0'.format(t)
                    pts = np.moveaxis(mesh[pname], 2, 0)[:, :, i] #ndims, npoints

                    # Determine which points are inside the box
                    inside = np.ones(pts.shape[1:], dtype=np.bool)
                    for l, p, u in zip(x0, pts, x1):
                        inside  &= (l <= p) & (p <= u)

                    if np.any(inside, axis=0):
                        vwts[i] *= self.cfg.getfloat(cfgsect,
                                                     'partitioner-weight', 2.0)
                        nel_affected +=1
                print('nel_affected = {} out of {}'.format(nel_affected,
                                                           mesh[pname].shape[1]))

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
        for k, v in mesh.items():
            if k.startswith('spt'):
                etype = k.split('_')[1]

                # Start with curved elements, followed by linear elements
                for i in np.argsort(~_find_curved_eles(etype, v)):
                    if (etype, i) not in bndeti:
                        nvetimap.append((etype, i))
                        nvparts.append(vpartmap[etype, i])

        return nvetimap, nvparts

    def _prepare_attrs(self, mesh):
        int_offs = defaultdict(lambda: 0)

        # Tag the offset of interior elements
        for k, v in mesh.items():
            m = re.match(r'con_p(\d+)p\d+', k)
            if m:
                p = m.group(1)
                for etype, eidx in v[['f0', 'f1']].tolist():
                    int_offs[etype, p] = max(int_offs[etype, p], eidx + 1)

        # Tag the offset of linear elements
        for k, v in dict(mesh).items():
            m = re.match(r'spt_(\w+)_p(\d+)', k)
            if m:
                midx = np.where(_find_curved_eles(m.group(1), v))[0]
                midx = midx[-1] + 1 if midx.size else 0

                mesh[k, 'lin_off'] = midx
                mesh[k, 'int_off'] = int_offs[m.groups()]

    def _partition_spts(self, mesh, vetimap, vparts):
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
        return {f'spt_{etype}_p{pn}': np.array(v).swapaxes(0, 1)
                for (etype, pn), v in spt_px.items()}

    def _partition_soln(self, soln, vetimap, vparts):
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
        return {f'soln_{etype}_p{pn}': np.dstack(v)
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
        for f in mesh:
            m = re.match('bcon_(.+?)_p0$', f)
            if m:
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

        # Combine any pre-existing parititons
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

        # Annotate the shape point arrays
        self._prepare_attrs(newmesh)

        # Build the solution converter
        def partition_soln(soln):
            # Check the UUID
            if curruuid != soln['mesh_uuid']:
                raise ValueError('Mismatched solution/mesh')

            # Combine any pre-existing parititons
            soln = self._combine_soln_parts(soln)

            # Partition
            if self.nparts > 1:
                newsoln = self._partition_soln(soln, vetimap, vparts)
            else:
                newsoln = soln

            # Handle the metadata
            newsoln['config'] = soln['config']
            newsoln['stats'] = soln['stats']
            newsoln['mesh_uuid'] = newuuid

            return newsoln

        return newmesh, rnum, partition_soln
