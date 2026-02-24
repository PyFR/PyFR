from dataclasses import dataclass, field, replace
import re

import h5py
import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import (AlltoallMixin, Scatterer, SparseScatterer, autofree,
                          get_comm_rank_root)
from pyfr.nputil import iter_struct


@dataclass
class _Mesh:
    fname: str
    raw: object

    ndims: int = None
    subset: bool = False

    creator: str = None
    codec: list = None
    uuid: str = None
    version: int = None

    etypes: list = field(default_factory=list)
    eidxs: dict = field(default_factory=dict)

    spts: dict = field(default_factory=dict)
    spts_nodes: dict = field(default_factory=dict)
    spts_curved: dict = field(default_factory=dict)

    con: list = field(default_factory=list)
    con_p: dict = field(default_factory=dict)
    bcon: dict = field(default_factory=dict)

class _MeshInterconnector(AlltoallMixin):
    """
    Given two input mesh eidxs data from _Mesh instances, create a forward mapping 
    from the src mesh to the dest mesh. 
    """

    def __init__(self, eidxs_src, eidxs_dest):

        # Ensure all ranks have all element types data.
        self.comm, self.rank, self.root = get_comm_rank_root()
        
        local_etypes = sorted(set(eidxs_src) | set(eidxs_dest))
        all_etypes = self.comm.allgather(local_etypes)
        self.etypes = sorted(set().union(*all_etypes))

        # allgather eidxs indexed by rank
        self.src_all  = {et: self.comm.allgather(self._as64(eidxs_src.get(et, ())))  for et in self.etypes}
        self.dest_all = {et: self.comm.allgather(self._as64(eidxs_dest.get(et, ()))) for et in self.etypes}

        self.send_idxs = {}
        self.recv_idxs = {}
        self.scount = {}
        self.sdisp = {}
        self.rcount = {}
        self.rdisp = {}

        # Prepare per-etype storage for mapping data
        self.set_relocation_idxs()

    def _as64(self, a):
        # Get int64 indices even if the arrays for the etype are empty
        return np.asarray(a if a is not None else (), dtype=np.int64)

    def set_relocation_idxs(self):
        comm = self.comm

        for et in self.etypes:
            src_local = np.asarray(self.src_all[et][self.rank], dtype=np.int64)

            send_lists = []
            for p in range(comm.size):
                dest_p = np.asarray(self.dest_all[et][p], dtype=np.int64)
                if src_local.size and dest_p.size:
                    mask = np.isin(src_local, dest_p, assume_unique=False)
                    sel = src_local[mask]
                else:
                    sel = np.empty(0, np.int64)
                send_lists.append(sel)

            sidxs  = np.concatenate(send_lists) if any(a.size for a in send_lists) else np.empty(0, np.int64)
            scount = np.array([a.size for a in send_lists], dtype=np.int32)
            sdisp  = self._count_to_disp(scount)

            # Learn receive sizes
            _recv_dummy, (rcount, rdisp) = self._alltoallcv(comm, sidxs, scount, sdisp)

            # Exchange promised global indices once; keep for final reordering later
            ridxs = np.empty(int(rcount.sum()), dtype=np.int64)
            self._alltoallv(comm, (sidxs, (scount, sdisp)), (ridxs, (rcount, rdisp)))

            self.send_idxs[et] = sidxs
            self.recv_idxs[et] = ridxs
            self.scount[et] = scount
            self.sdisp[et]  = sdisp
            self.rcount[et] = rcount
            self.rdisp[et]  = rdisp

            # global to local id on src
            src_local = np.asarray(self.src_all[et][self.rank], dtype=np.int64)
            self.src_pos = getattr(self, 'src_pos', {})
            self.src_pos[et] = {int(g): i for i, g in enumerate(src_local)}

            # recv -> local dest permutation
            dest_local = np.asarray(self.dest_all[et][self.rank], dtype=np.int64)
            rpos = {int(g): i for i, g in enumerate(self.recv_idxs[et])}
            self.recv_to_dest = getattr(self, 'recv_to_dest', {})
            self.recv_to_dest[et] = np.fromiter((rpos[int(g)] for g in dest_local),
                                            count=dest_local.size, dtype=np.int64)

            self.n_dest = getattr(self, 'n_dest', {})
            self.n_dest[et] = int(dest_local.size)

            self.ssum = getattr(self, 'ssum', {})
            self.rsum = getattr(self, 'rsum', {})
            self.ssum[et] = int(self.scount[et].sum())
            self.rsum[et] = int(self.rcount[et].sum())

    def relocate(self, edict_src, edim):
        src0 = self.preproc_edict(edict_src, edim=edim)
        dst0 = self.relocate_edict(src0)
        return self.postproc_edict(dst0, edim=edim)

    def preproc_edict(self, edict_in, *, edim):
        comm = self.comm
        edict0 = {}

        # Local specs for dtype/trailing shape
        specs = {}
        for et in self.etypes:
            ax = edim
            Ne = len(self.src_all[et][self.rank])

            a = edict_in.get(et, None)
            if a is not None:
                a = np.asarray(a)
                if a.ndim <= ax:
                    raise ValueError("[pre] r={} et={} invalid edim={} for shape={}"
                                    .format(self.rank, et, ax, a.shape))
                a0 = np.moveaxis(a, ax, 0) if ax else a
                if a0.shape[0] != Ne:
                    raise ValueError("[pre] r={} et={} mismatch rows={} vs Ne_src={}"
                                    .format(self.rank, et, a0.shape[0], Ne))
                edict0[et] = np.ascontiguousarray(a0)
                specs[et] = (str(edict0[et].dtype), edict0[et].shape[1:])
            else:
                specs[et] = (None, None)

        # Global agreement on dtype/shape; synthesize empties where needed
        gspecs = comm.allgather(specs)
        for et in self.etypes:
            gdt, gtr = None, None
            for d in gspecs:
                dt, tr = d[et]
                if dt is not None:
                    if gdt is None:
                        gdt, gtr = dt, tr
                    elif dt != gdt or tr != gtr:
                        raise ValueError("[pre] inconsistent dtype/shape for etype {} across ranks: {} vs {}"
                                        .format(et, (gdt, gtr), (dt, tr)))

            if gdt is None:
                # No one provided this etype. If mapping expects traffic, bail.
                ssum = int(self.scount.get(et, np.array([], np.int32)).sum()) if et in self.scount else 0
                rsum = int(self.rcount.get(et, np.array([], np.int32)).sum()) if et in self.rcount else 0
                Ne   = len(self.src_all[et][self.rank])
                if ssum or rsum or Ne:
                    raise ValueError("[pre] no array anywhere for etype '{}' ".format(et))
                continue

            if et not in edict0:
                Ne = len(self.src_all[et][self.rank])
                edict0[et] = np.empty((Ne,) + tuple(gtr), dtype=np.dtype(gdt))

        return edict0

    def relocate_edict(self, edict0):
        out0 = {}
        for et, a0 in (edict0 or {}).items():
            if et not in self.send_idxs or et not in self.recv_to_dest:
                raise ValueError("relocate_edict: mapping not ready for etype '{}'".format(et))
            if a0.ndim == 0:
                raise ValueError("relocate_edict: scalar array for etype '{}'".format(et))

            gids = self.send_idxs[et]
            pos = np.fromiter((self.src_pos[et].get(int(g), -1) for g in gids),
                            count=self.ssum[et], dtype=np.int64)
            if (pos < 0).any():
                miss = [int(g) for g, p in zip(gids.tolist(), pos.tolist()) if p < 0][:8]
                raise ValueError("[xfer] r={} et={} missing local GIDs (first few): {}"
                                .format(self.rank, et, miss))

            svals = a0[pos]
            sc, sd = self.scount[et], self.sdisp[et]
            rc, rd = self.rcount[et], self.rdisp[et]

            rtot = int(rc.sum())
            rvals = np.empty((rtot, *a0.shape[1:]), dtype=a0.dtype)
            self._alltoallv(self.comm, (svals, (sc, sd)), (rvals, (rc, rd)))

            pr = self.recv_to_dest[et]
            if pr.size != self.n_dest[et]:
                raise ValueError("[xfer] r={} et={} recv_to_dest size mismatch".format(self.rank, et))

            out0[et] = rvals[pr]

        return out0

    def postproc_edict(self, edict0, *, edim):
        out = {}
        for et, a0 in (edict0 or {}).items():
            if a0.shape[0] == 0:
                continue
            # Restore element axis to edim
            out[et] = np.moveaxis(a0, 0, edim) if edim else a0
        return out


class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True):
        self.f = h5py.File(fname, 'r')
        self.mesh = _Mesh(fname=fname, raw=self.f)

        # Read in and transform the various parts of the mesh
        self._read_metadata()
        self._read_partitioning(pname)
        self._read_eles()
        self._read_nodes()

        if construct_con:
            self._construct_con()

    def close(self):
        self.f.close()

    def load_soln(self, sname, prefix=None):
        mesh, soln = self.load_subset_mesh_soln(sname, prefix)

        # Ensure the solution is not subset
        if mesh is not self.mesh:
            raise ValueError('Subset solutions are not supported')

        return soln

    def load_subset_mesh_soln(self, sname, prefix=None):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(sname, 'r') as f:
            if rank == root:
                # Ensure the solution is from the mesh we are using
                uuid = f['mesh-uuid'][()].decode()
                if uuid != self.mesh.uuid:
                    raise RuntimeError('Invalid solution for mesh')

                # Read any config and stats records
                soln = {fname: f[fname][()].decode()
                        for fname in f if fname.startswith('config')}
                soln['stats'] = f['stats'][()].decode()
            else:
                soln = None

            # Broadcast and parse
            soln = comm.bcast(soln, root=root)
            soln = {k: Inifile(v) for k, v in soln.items()}

            # Read serialised data
            sdata = {}
            if rank == root:
                def svisit(name):
                    if name.startswith(('plugins', 'bcs', 'intg')):
                        if not isinstance(f[name], h5py.Group):
                            sdata[name] = f[name][()]
                f.visit(svisit)

            soln |= comm.bcast(sdata, root=root)

            # Obtain the polynomial order
            order = soln['config'].getint('solver', 'order')

            # If no prefix has been specified then obtain it from the file
            if prefix is None:
                prefix = soln['stats'].get('data', 'prefix')

            # Note if any elements are subset
            subset = {}

            # Read and scatter the solution data
            for etype in self.escatter:
                # If the element is not present, mark it as completely subset
                if (ek := f'{prefix}/p{order}-{etype}') not in f:
                    subset[etype] = []
                    continue
                # If the element is partially subset use a sparse scatterer
                elif (ei := f'{ek}-idxs') in f:
                    try:
                        idxs = self.mesh.eidxs[etype]
                    except KeyError:
                        idxs = np.empty(0, dtype=int)

                    escatter = SparseScatterer(comm, f[ei], idxs)
                    subset[etype] = escatter.ridx
                # Complete element present so reuse the elements scatterer
                else:
                    escatter = self.escatter[etype]

                # Read the solution
                esoln = escatter(f[ek])
                if escatter.cnt:
                    soln[etype] = esoln.swapaxes(0, 2)

                # Read the partition data
                epart = escatter(f[f'{ek}-parts'])
                if escatter.cnt:
                    soln[f'{etype}-parts'] = epart

        # If the solution is subset then subset the mesh, too
        if subset:
            return self._subset_mesh(subset), soln
        else:
            return self.mesh, soln

    def _subset_mesh(self, subset):
        eidxs, spts, spts_nodes, spts_curved = {}, {}, {}, {}

        for etype in self.mesh.spts:
            if etype in subset:
                sidx = subset[etype]
                if len(sidx):
                    eidxs[etype] = self.mesh.eidxs[etype][sidx]
                    spts[etype] = self.mesh.spts[etype][:, sidx]
                    spts_nodes[etype] = self.mesh.spts_nodes[etype][sidx]
                    spts_curved[etype] = self.mesh.spts_curved[etype][sidx]
            else:
                eidxs[etype] = self.mesh.eidxs[etype]
                spts[etype] = self.mesh.spts[etype]
                spts_nodes[etype] = self.mesh.spts_nodes[etype]
                spts_curved[etype] = self.mesh.spts_curved[etype]

        return replace(self.mesh, subset=True, eidxs=eidxs, spts=spts,
                       spts_nodes=spts_nodes, spts_curved=spts_curved,
                       con=None, con_p=None, bcon=None)

    def _read_metadata(self):
        mesh = self.mesh
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            creator = self.f['creator'][()].decode()
            codec = [c.decode() for c in self.f['codec']]
            uuid = self.f['mesh-uuid'][()].decode()
            version = self.f['version'][()]

            meta = (creator, codec, uuid, version)
        else:
            meta = None

        meta = comm.bcast(meta, root=root)
        mesh.creator, mesh.codec, mesh.uuid, mesh.version = meta

    def _read_with_idxs(self, dset, idxs):
        comm, rank, root = get_comm_rank_root()

        # Construct a Scatterer to read in and distribute the data
        s = Scatterer(comm, idxs)

        return s(dset), s

    def _select_partitioning(self, size, pname=None):
        # If a partitioning has been specified then use it
        if pname:
            pinfo = self.f[f'partitionings/{pname}']
            nparts = len(pinfo['eles'].attrs['regions'])
            if nparts != size:
                raise RuntimeError(f'Partitioning {pname} has {nparts} parts '
                                   f'but running with {size} ranks')
        # Otherwise, try to find one
        else:
            for pname, pinfo in self.f['partitionings'].items():
                nparts = len(pinfo['eles'].attrs['regions'])
                if nparts == size:
                    break
            else:
                raise RuntimeError('Mesh does not have any partitionings with '
                                   f'{size} ranks')

        return pname, pinfo

    def _read_partitioning(self, pname=None):
        comm, rank, root = get_comm_rank_root()
        size = comm.size

        # Have the root rank read in the partitioning metadata
        if rank == root:
            pname, pinfo = self._select_partitioning(size, pname)

            # Read the element region data
            einfo = pinfo['eles'].attrs['regions']

            # Read the neighbours data
            if size > 1:
                ninfo = pinfo['neighbours']
                ninfo = np.split(ninfo[()], ninfo.attrs['regions'][1:-1])
            else:
                ninfo = [[]]
        else:
            pname = einfo = ninfo = None

        # Broadcast this metadata
        ppath = 'partitionings/' + comm.bcast(pname, root=root)
        einfo = comm.scatter(einfo, root=root)
        self.neighbours = comm.scatter(ninfo, root=root)

        # Determine the element types in the mesh
        self.mesh.etypes = etypes = sorted(self.f['eles'])

        # Read our portion of the partitioning table
        peles = self.f[f'{ppath}/eles'][einfo[0]:einfo[-1]]
        peles = np.split(peles, [i - einfo[0] for i in einfo[1:-1]])

        # With this determine the indices associated with each element
        self.mesh.eidxs = {et: pe for et, pe in zip(etypes, peles) if pe.size}

    def _read_eles(self):
        self.eles, self.escatter = eles, escatter = {}, {}

        # Collectively read in and distribute each element array
        for etype in self.mesh.etypes:
            dset = self.f[f'eles/{etype}']
            idxs = self.mesh.eidxs.get(etype, [])
            einfo, escatter[etype] = self._read_with_idxs(dset, idxs)

            # If we have any elements of this type then save the einfo
            if len(idxs):
                eles[etype] = einfo

    def _read_nodes(self):
        enodes = [einfo['nodes'] for einfo in self.eles.values()]

        # Determine the overall set of nodes across all element types
        idxs = np.concatenate([en.ravel() for en in enodes])

        # Note how many dimensions we have
        self.mesh.ndims = self.f['nodes'].dtype['location'].shape[0]

        # Read in these nodes
        nodes = self._read_with_idxs(self.f['nodes'], idxs.ravel())[0]

        # Determine where each element type is in the nodes array
        eoffs = np.cumsum([en.size for en in enodes])

        # Use this to split the nodes array back up
        nodes = np.split(nodes['location'], eoffs[:-1])

        # Reshape and add to the mesh
        for (etype, einfo), n in zip(self.eles.items(), nodes):
            spts = n.reshape(*einfo['nodes'].shape, -1).swapaxes(0, 1)

            self.mesh.spts[etype] = spts
            self.mesh.spts_nodes[etype] = einfo['nodes']
            self.mesh.spts_curved[etype] = einfo['curved']

    def _construct_con(self):
        codec = self.mesh.codec
        eidxs = {k: v.tolist() for k, v in self.mesh.eidxs.items()}
        etypes = self.mesh.etypes

        # Create a map from global to local element numbers
        glmap = [{}]*len(etypes)
        for i, etype in enumerate(etypes):
            if etype in eidxs:
                glmap[i] = {k: j for j, k in enumerate(eidxs[etype])}

        # Create cidx indexed maps
        cdone, cefidx = [None]*len(codec), [None]*len(codec)
        for cidx, c in enumerate(codec):
            if (m := re.match(r'eles/(\w+)/(\d+)$', c)):
                etype, fidx = m[1], m[2]
                cdone[cidx] = set()
                cefidx[cidx] = (etype, etypes.index(etype), int(fidx))

        conl, conr = [], []
        bcon = {i: [] for i, c in enumerate(codec) if c.startswith('bc/')}
        resid = {}

        for etype, einfo in self.eles.items():
            i = etypes.index(etype)
            for fidx, eface in enumerate(einfo['faces'].T):
                efcidx = codec.index(f'eles/{etype}/{fidx}')

                for j, (cidx, off) in enumerate(iter_struct(eface)):
                    # Boundary
                    if off == -1:
                        bcon[cidx].append((etype, j, fidx))
                    # Unpaired face
                    elif j not in cdone[efcidx]:
                        # Lookup the element type and face number
                        ketype, ketidx, kfidx = cefidx[cidx]

                        # If our rank has the element then pair it
                        if (k := glmap[ketidx].get(off)) is not None:
                            conl.append((etype, j, fidx))
                            conr.append((ketype, k, kfidx))
                            cdone[cidx].add(k)
                        # Otherwise add it to the residual dict
                        else:
                            resid[efcidx, eidxs[etype][j]] = (cidx, off)

        # Add the internal connectivity to the mesh
        self.mesh.con = (conl, conr)

        for k, v in bcon.items():
            if v:
                self.mesh.bcon[codec[k][3:]] = v

        # Handle inter-partition connectivity
        if resid:
            self._construct_mpi_con(glmap, cefidx, resid)

    def _construct_mpi_con(self, glmap, cefidx, resid):
        comm, rank, root = get_comm_rank_root()

        # Create a neighbourhood collective communicator
        ncomm = autofree(comm.Create_dist_graph_adjacent(self.neighbours,
                                                         self.neighbours))

        # Create a list of our unpaired faces
        unpaired = list(resid.values())

        # Distribute this to each of our neighbours
        nunpaired = ncomm.neighbor_allgather(unpaired)

        # See which of our neighbours unpaired faces we have
        matches = [[resid[j] for j in nunp if j in resid]
                   for nunp in nunpaired]

        # Distribute this information back to our neighbours
        nmatches = ncomm.neighbor_alltoall(matches)

        for nrank, nmatch in zip(self.neighbours, nmatches):
            if rank < nrank:
                ncon = sorted([(resid[m], m) for m in nmatch])
                ncon = [r for l, r in ncon]
            else:
                ncon = sorted([(m, resid[m]) for m in nmatch])
                ncon = [l for l, r in ncon]

            nncon = []
            for cidx, off in ncon:
                etype, etidx, fidx = cefidx[cidx]
                nncon.append((etype, glmap[etidx][off], fidx))

            # Add the connectivity to the mesh
            self.mesh.con_p[nrank] = nncon
