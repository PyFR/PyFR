from collections import Counter, defaultdict, namedtuple
import itertools as it
import re
from uuid import UUID

import numpy as np

from pyfr.inifile import Inifile
from pyfr.progress import NullProgressSequence
from pyfr.util import digest


Graph = namedtuple('Graph', ['vtab', 'etab', 'vwts', 'ewts'])


class BasePartitioner:
    def __init__(self, partwts, elewts=None, nsubeles=64, opts={}):
        self.partwts = partwts
        self.elewts = elewts
        self.nparts = len(partwts)
        self.nsubeles = nsubeles

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
                    linf[en].append(mesh[f'spt_{en}_p{i}', 'linear'])
                    rnum[en].update(((i, j), (0, off + j)) for j in range(n))

        def offset_con(con, pr):
            con = con.copy().astype('U4,i8,i1,i2')

            for en, pn in pinf.items():
                if pn[pr] > 0:
                    con['f1'][np.where(con['f0'] == en)] += offs[en][pr]

            return con

        # Connectivity
        intcon, mpicon, bccon = [], {}, defaultdict(list)
        intcon_p, nintcon = defaultdict(list), 0

        for f in mesh:
            if (mi := re.match(r'con_p(\d+)$', f)):
                intcon.append(offset_con(mesh[f], int(mi[1])))

                for k, v in mesh.attrs(f).items():
                    if k.startswith('periodic_'):
                        intcon_p[k].append(v + nintcon)

                nintcon += intcon[-1].shape[1]
            elif (mm := re.match(r'con_p(\d+)p(\d+)$', f)):
                l, r = int(mm[1]), int(mm[2])
                lcon = offset_con(mesh[f], l)

                if (r, l) in mpicon:
                    rcon = mpicon.pop((r, l))
                    intcon.append(np.vstack([lcon, rcon]))
                else:
                    mpicon[l, r] = lcon
            elif (bc := re.match(r'bcon_(.+?)_p(\d+)$', f)):
                name, l = bc[1], int(bc[2])
                bccon[name].append(offset_con(mesh[f], l))

        # Output data type
        dtype = 'U4,i8,i1,i2'

        # Concatenate these arrays to from the new mesh
        newmesh = {'con_p0': np.hstack(intcon).astype(dtype)}

        for en in spts:
            newmesh[f'spt_{en}_p0'] = np.hstack(spts[en])
            newmesh[f'spt_{en}_p0', 'linear'] = np.hstack(linf[en])

        for k, v in intcon_p.items():
            newmesh['con_p0', k] = np.hstack(v)

        for k, v in bccon.items():
            newmesh[f'bcon_{k}_p0'] = np.hstack(v).astype(dtype)

        return newmesh, rnum

    def _get_elewts(self, mesh):
        # If we have an element weighting table then use it
        if self.elewts is not None:
            return self.elewts
        # Else, use multiple constraints for a balanced partitioning
        else:
            etypes = []
            for f in mesh:
                if isinstance(f, str) and f.startswith('spt'):
                    etypes.append(f.split('_')[1])

            return dict(zip(etypes, np.eye(len(etypes), dtype=int)))

    def _combine_soln_parts(self, soln, prefix):
        newsoln = defaultdict(list)

        for f, (en, shape) in soln.array_info(prefix).items():
            newsoln[f'{prefix}_{en}_p0'].append(soln[f])

        return {k: np.dstack(v) for k, v in newsoln.items()}

    def _construct_graph(self, con, elewts, exwts={}):
        # Edges of the dual graph
        con = con[['f0', 'f1']]
        con = np.hstack([con, con[::-1]])

        # Sort by the left hand side
        idx = np.lexsort([con['f0'][0], con['f1'][0]])
        con = con[:, idx]

        # Left and right hand side element types/indicies
        lhs, rhs = con

        # Compute vertex offsets
        vtab = np.where(lhs[1:] != lhs[:-1])[0]
        vtab = np.concatenate(([0], vtab + 1, [len(lhs)]))

        # Compute the element type/index to vertex number map
        vetimap = lhs[vtab[:-1]].tolist()
        etivmap = {k: v for v, k in enumerate(vetimap)}

        # Prepare the edges and their weights
        etab = np.array([etivmap[r] for r in rhs.tolist()])
        ewts = np.ones_like(etab)

        # Prepare the vertex weights
        vwts = np.array([exwts.get(ti, elewts[ti[0]]) for ti in vetimap])
        vwts = vwts.reshape(len(vwts), -1)

        return Graph(vtab, etab, vwts, ewts), vetimap

    def _partition_graph(self, graph, partwts):
        pass

    def _group_periodic_eles(self, mesh, elewts):
        con = mesh['con_p0'][['f0', 'f1']]
        pmerge, pnames = {}, {}

        # Extract the periodic connectivity from the mesh
        for f, v in mesh.items():
            if f[0] == 'con_p0' and f[1].startswith('periodic'):
                for i, (l, r) in zip(v, con[:, v].T.tolist()):
                    pnames[i] = f[1]
                    if l != r:
                        if l in pmerge and r in pmerge:
                            pr = pmerge[r]
                            for ll, rr in pmerge.items():
                                if rr == pr:
                                    pmerge[ll] = pmerge[l]
                            pmerge[pr] = pmerge[l]
                        elif l in pmerge:
                            if r != pmerge[l]:
                                pmerge[r] = pmerge[l]
                        elif r in pmerge:
                            if l != pmerge[r]:
                                pmerge[l] = pmerge[r]
                        else:
                            pmerge[l] = r

        if pmerge:
            # Eliminate periodic connectivity
            con = np.delete(con, list(pnames), axis=1)

            # Merge the associated elements
            for i, (l, r) in enumerate(con.T.tolist()):
                if l in pmerge:
                    con[0, i] = pmerge[l]
                if r in pmerge:
                    con[1, i] = pmerge[r]

        # Tally up the weights for the merged elements
        exwts = {(t, i): elewts[t] for t, i in set(pmerge.values())}
        for (lt, li), r in pmerge.items():
            exwts[r] += elewts[lt]

        return con, exwts, pmerge, pnames

    def _ungroup_periodic_eles(self, pmerge, vetimap, vparts):
        # Invert the merge dictionary
        punmerge = defaultdict(list)
        for l, r in pmerge.items():
            punmerge[r].append(l)

        # Create new vertices for the unmerged periodic elements
        vnetimap, vnparts = list(vetimap), list(vparts)
        for i, j in enumerate(vetimap):
            if j in punmerge:
                for k in punmerge[j]:
                    vnetimap.append(k)
                    vnparts.append(vparts[i])

        return vnetimap, vnparts

    def _renumber_verts(self, mesh, vetimap, vparts):
        pscon = [[] for i in range(self.nparts)]
        elewts = defaultdict(lambda: 1)
        vpartmap, bndeti = dict(zip(vetimap, vparts)), set()

        # Construct per-partition connectivity arrays and tag elements
        # which are on partition boundaries
        for l, r in zip(*mesh['con_p0'][['f0', 'f1']].tolist()):
            if vpartmap[l] == vpartmap[r]:
                pscon[vpartmap[l]].append([l, r])
            else:
                pscon[vpartmap[l]].append([l, r])
                pscon[vpartmap[r]].append([l, r])
                bndeti |= {l, r}

        # Start by assigning the lowest numbers to these boundary elements
        nvetimap = sorted(bndeti)
        nvparts = [vpartmap[eti] for eti in nvetimap]

        # Use sub-partitioning to assign interior element numbers
        for part, scon in enumerate(pscon):
            # Construct a graph for this partition
            scon = np.array(scon, dtype='U4,i4').T
            sgraph, svetimap = self._construct_graph(scon, elewts)

            # Determine the number of sub-partitions
            nsp = len(svetimap) // self.nsubeles + 1

            # Partition the graph
            if nsp == 1:
                svparts = [0]*len(svetimap)
            else:
                svparts = self._partition_graph(sgraph, [1]*nsp)

            # Group elements according to their type (linear vs curved)
            # and sub-partition number
            linsvetimap = [[] for i in range(nsp)]
            cursvetimap = [[] for i in range(nsp)]
            for (etype, eidx), spart in zip(svetimap, svparts):
                if (etype, eidx) in bndeti:
                    continue

                if mesh[f'spt_{etype}_p0', 'linear'][eidx]:
                    linsvetimap[spart].append((etype, eidx))
                else:
                    cursvetimap[spart].append((etype, eidx))

            # Append to the global list
            nvetimap.extend(it.chain(*cursvetimap, *linsvetimap))
            nvparts.extend([part]*sum(map(len, cursvetimap + linsvetimap)))

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

    def _partition_con(self, mesh, vetimap, vparts, pnames):
        con_px = defaultdict(list)
        con_pxpy = defaultdict(list)
        con_px_periodic = defaultdict(lambda: defaultdict(list))
        bcon_px = defaultdict(list)

        # Global-to-local element index map
        eleglmap = {}
        pcounter = Counter()

        for (etype, eidxg), part in zip(vetimap, vparts):
            eleglmap[etype, eidxg] = (part, pcounter[etype, part])
            pcounter[etype, part] += 1

        # Generate the face connectivity
        for i, (l, r) in enumerate(mesh['con_p0'].T.tolist()):
            letype, leidxg, lfidx, lflags = l
            retype, reidxg, rfidx, rflags = r

            lpart, leidxl = eleglmap[letype, leidxg]
            rpart, reidxl = eleglmap[retype, reidxg]

            conl = (letype, leidxl, lfidx, lflags)
            conr = (retype, reidxl, rfidx, rflags)

            if lpart == rpart:
                # If this face is periodic, then tag it as such
                if (pname := pnames.get(i, None)):
                    con_px_periodic[lpart][pname].append(len(con_px[lpart]))

                con_px[lpart].append([conl, conr])
            else:
                con_pxpy[lpart, rpart].append(conl)
                con_pxpy[rpart, lpart].append(conr)

        # Generate boundary conditions
        for f in filter(lambda f: isinstance(f, str), mesh):
            if (m := re.match('bcon_(.+?)_p0$', f)):
                for lpetype, leidxg, lfidx, lflags in mesh[f].tolist():
                    lpart, leidxl = eleglmap[lpetype, leidxg]
                    conl = (lpetype, leidxl, lfidx, lflags)

                    bcon_px[m[1], lpart].append(conl)

        # Output data type
        dtype = 'S4,i8,i1,i2'

        # Output
        con = {}

        for px, v in con_px.items():
            con[f'con_p{px}'] = np.array(v, dtype=dtype).T

        for (px, py), v in con_pxpy.items():
            con[f'con_p{px}p{py}'] = np.array(v, dtype=dtype)

        for (etype, px), v in bcon_px.items():
            con[f'bcon_{etype}_p{px}'] = np.array(v, dtype=dtype)

        for px, v in con_px_periodic.items():
            for name, idxs in v.items():
                con[f'con_p{px}', name] = np.array(idxs)

        return con, eleglmap

    def partition(self, mesh, progress=NullProgressSequence):
        # Extract the current UUID from the mesh
        curruuid = mesh['mesh_uuid']

        # Combine any pre-existing partitions
        with progress.start('Combine mesh parts'):
            mesh, rnum = self._combine_mesh_parts(mesh)

        # Obtain the element weighting table
        elewts = self._get_elewts(mesh)

        # Merge periodic elements
        with progress.start('Group periodic elements'):
            con, exwts, pmerge, pnames = self._group_periodic_eles(mesh,
                                                                   elewts)

        # Obtain the dual graph for this mesh
        with progress.start('Construct graph'):
            graph, vetimap = self._construct_graph(con, elewts, exwts=exwts)

        # Partition the graph
        with progress.start('Partition graph'):
            if self.nparts > 1:
                vparts = self._partition_graph(graph, self.partwts).tolist()

                if (n := len(set(vparts))) != self.nparts:
                    raise RuntimeError(f'Partitioner error: mesh has {n} '
                                       f'parts versus goal of {self.nparts}')
            else:
                vparts = [0]*len(vetimap)

        # Unmerge periodic elements
        with progress.start('Ungroup periodic elements'):
            vetimap, vparts = self._ungroup_periodic_eles(pmerge, vetimap,
                                                          vparts)

        # Renumber vertices
        with progress.start('Renumber vertices'):
            vetimap, vparts = self._renumber_verts(mesh, vetimap, vparts)

        # Repartition the mesh
        with progress.start('Repartition mesh'):
            # Partition the connectivity portion of the mesh
            newmesh, eleglmap = self._partition_con(mesh, vetimap, vparts,
                                                    pnames)

            # Handle the shape points
            newmesh |= self._partition_spts(mesh, vetimap, vparts)

            # Update the renumbering table
            for etype, emap in rnum.items():
                for k, (pidx, eidx) in emap.items():
                    emap[k] = eleglmap[etype, eidx]

        # Generate a new UUID for the mesh
        newmesh['mesh_uuid'] = newuuid = str(UUID(digest(curruuid, rnum)[:32]))

        # Build the solution converter
        def partition_soln(soln):
            # Check the UUID
            if curruuid != soln['mesh_uuid']:
                raise ValueError('Mismatched solution/mesh')

            # Obtain the prefix
            prefix = Inifile(soln['stats']).get('data', 'prefix')

            # Combine and repartition the solution
            newsoln = self._combine_soln_parts(soln, prefix)
            newsoln = self._partition_soln(newsoln, prefix, vetimap, vparts)

            # Copy over the metadata
            for f in soln:
                if re.match('stats|config|plugins', f):
                    newsoln[f] = soln[f]

            # Apply the new UUID
            newsoln['mesh_uuid'] = newuuid

            return newsoln

        return newmesh, rnum, partition_soln
