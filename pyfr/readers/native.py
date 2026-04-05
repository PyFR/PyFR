from dataclasses import dataclass, field, replace
import re

import h5py
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from pyfr.inifile import Inifile
from pyfr.mpiutil import (Scatterer, SparseScatterer, autofree,
                          get_comm_rank_root)
from pyfr.readers.shared_nodes import SharedNodesFinder
from pyfr.util import first


@dataclass
class Mesh:
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
    colours: dict = field(default_factory=dict)

    con: tuple = field(default_factory=tuple)
    con_p: dict = field(default_factory=dict)
    bcon: dict = field(default_factory=dict)
    cidxmap: dict = field(default_factory=dict)

    # Shared nodes for C0 continuous fields
    node_idxs: np.ndarray = None
    node_valency: np.ndarray = None
    node_locs: np.ndarray = None
    shared_nodes: object = None


@dataclass
class Solution:
    config: object
    stats: object
    fields: list
    data: dict = field(default_factory=dict)
    grad_data: dict = field(default_factory=dict)
    aux: dict = field(default_factory=dict)
    prevcfgs: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)


class Connectivity:
    def __init__(self, cidxs, eidxs, cidxmap):
        self.cidxmap = cidxmap
        self.cidxs = cidxs
        self.eidxs = eidxs
        self._ucidxs = np.unique(cidxs).tolist()

    def __len__(self):
        return len(self.cidxs)

    def items(self):
        for cidx in self._ucidxs:
            etype, fidx = self.cidxmap[cidx]
            yield etype, fidx, self.eidxs[self.cidxs == cidx]

    def foreach(self):
        for cidx in self._ucidxs:
            mask = self.cidxs == cidx
            etype, fidx = self.cidxmap[cidx]
            yield etype, fidx, self.eidxs[mask], np.flatnonzero(mask)

    def map_eles(self, data, dtype=None):
        result = np.empty(len(self), dtype=dtype or first(data.values()).dtype)
        for etype, fidx, eidxs, mask in self.foreach():
            result[mask] = data[etype][eidxs]
        return result


class NativeReader:
    def __init__(self, fname, pname=None, *, construct_con=True):
        self.f = h5py.File(fname, 'r')
        self.mesh = Mesh(fname=fname, raw=self.f)

        # Read in and transform the various parts of the mesh
        self._read_metadata()
        self._read_partitioning(pname)
        self._read_eles()
        self._read_nodes()

        if construct_con:
            self._construct_con()
            self._construct_shared_nodes()

    def close(self):
        self.f.close()

    def load_soln(self, sname, prefix=None):
        mesh, soln = self.load_subset_mesh_soln(sname, prefix)

        # Ensure the solution is not subset
        if mesh is not self.mesh:
            raise ValueError('Subset solutions are not supported')

        return soln

    def _read_soln_header(self, f):
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            # Ensure the solution is from the mesh we are using
            uuid = f['mesh-uuid'][()].decode()
            if uuid != self.mesh.uuid:
                raise RuntimeError('Invalid solution for mesh')

            # Ensure solution format is v2
            if f['version'][()] != 2:
                raise RuntimeError('Solution file must be format v2')

            # Read any config and stats records
            cfgs = {fname: f[fname][()].decode()
                    for fname in f if fname.startswith('config')}
            cfgs['stats'] = f['stats'][()].decode()
        else:
            cfgs = None

        # Broadcast and parse
        cfgs = comm.bcast(cfgs, root=root)
        cfgs = {k: Inifile(v) for k, v in cfgs.items()}

        # Extract previous configs
        prevcfgs = {k: v for k, v in cfgs.items() if k.startswith('config-')}

        # Read serialised state (plugins, bcs, intg)
        if rank == root:
            state = {}
            def svisit(name):
                if name.startswith(('plugins/', 'bcs/', 'intg/')):
                    if not isinstance(f[name], h5py.Group):
                        state[name] = f[name][()]
            f.visit(svisit)
        else:
            state = None

        state = comm.bcast(state, root=root)

        return Solution(config=cfgs['config'], stats=cfgs['stats'],
                        fields=None, prevcfgs=prevcfgs, state=state)

    def _soln_fields(self, dtype):
        fields = []
        for g in dtype.names:
            if g in ('grad', 'aux'):
                continue

            prefix = '' if g == 'soln' else f'{g}-'
            fields.extend(f'{prefix}{n}' for n in dtype[g].names)
        return fields

    def _unpack_esoln(self, soln, etype, esoln, dtype):
        dgroups = [g for g in dtype.names if g not in ('grad', 'aux')]
        ne, nd = len(esoln), self.mesh.ndims

        # Unpack all data groups into a single array
        parts = []
        for g in dgroups:
            arr = s2u(esoln[g]).reshape(ne, len(dtype[g].names), -1)
            parts.append(arr.transpose(2, 1, 0))

        soln.data[etype] = np.concatenate(parts, axis=1)

        # Gradient data
        if 'grad' in dtype.names:
            gv = len(dtype['grad'].names)
            g = s2u(esoln['grad']).reshape(ne, gv, nd, -1)
            soln.grad_data[etype] = g.transpose(2, 3, 1, 0)

        # Auxiliary fields
        if 'aux' in dtype.names:
            soln.aux[etype] = {n: esoln['aux'][n] for n in dtype['aux'].names}

    def load_subset_mesh_soln(self, sname, prefix=None):
        comm, rank, root = get_comm_rank_root()

        with h5py.File(sname, 'r') as f:
            soln = self._read_soln_header(f)

            # Obtain the polynomial order
            order = soln.config.getint('solver', 'order')

            # If no prefix has been specified then obtain it from the file
            if prefix is None:
                prefix = soln.stats.get('data', 'prefix')

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

                # Build field list from the first dataset encountered
                if soln.fields is None:
                    soln.fields = self._soln_fields(f[ek].dtype)

                esoln = escatter(f[ek])
                if escatter.cnt:
                    self._unpack_esoln(soln, etype, esoln, f[ek].dtype)

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

        # Store unique node indices, valency, and locations for vertices
        unique_idxs, first_occ = np.unique(idxs, return_index=True)
        self.mesh.node_idxs = unique_idxs
        self.mesh.node_valency = nodes['valency'][first_occ]
        self.mesh.node_locs = nodes['location'][first_occ]

        # Determine where each element type is in the nodes array
        eoffs = np.cumsum([en.size for en in enodes])

        # Use this to split the nodes array back up
        locs = np.split(nodes['location'], eoffs[:-1])

        # Reshape and add to the mesh
        for (etype, einfo), n in zip(self.eles.items(), locs):
            spts = n.reshape(*einfo['nodes'].shape, -1).swapaxes(0, 1)

            self.mesh.spts[etype] = spts
            self.mesh.spts_nodes[etype] = einfo['nodes']
            self.mesh.spts_curved[etype] = einfo['curved']
            if 'colour' in einfo.dtype.names:
                self.mesh.colours[etype] = einfo['colour']

    def _parse_codec(self):
        codec = self.mesh.codec
        ncodec = len(codec)

        cidxmap = {}
        cetmap = np.full(ncodec, -1, dtype=np.int16)

        for cidx, c in enumerate(codec):
            if (m := re.match(r'eles/(\w+)/(\d+)$', c)):
                cidxmap[cidx] = etype, fidx = m[1], int(m[2])
                cetmap[cidx] = self.mesh.etypes.index(etype)

        self.mesh.cidxmap = cidxmap
        return cidxmap, cetmap

    @staticmethod
    def _pack_pairs(*pairs):
        stride = max(o.max() for _, o in pairs) + 1
        return [c * stride + o for c, o in pairs]

    @staticmethod
    def _pair_finder(lhs, stride):
        # Pack (cidx, idx) pairs into flat keys for binary search
        keys = lhs.cidx.astype(int)*stride + lhs.idx
        sord = np.argsort(keys)

        def find(rec):
            qkeys = rec['cidx'].astype(int)*stride + rec['idx']
            pos = np.searchsorted(keys, qkeys, sorter=sord)
            idx = np.take(sord, pos, mode='clip')
            return idx[keys[idx] == qkeys]

        return find

    def _build_g2l(self):
        g2l = {}

        for etype in self.mesh.etypes:
            if (gi := self.mesh.eidxs.get(etype)) is not None:
                perm = np.argsort(gi)
                g2l[etype] = (gi, gi[perm], perm)

        return g2l

    def _flatten_faces(self, g2l):
        codec = self.mesh.codec
        parts = []
        for etype, einfo in self.eles.items():
            gi = g2l[etype][0]
            for fidx, eface in enumerate(einfo['faces'].T):
                efcidx = codec.index(f'eles/{etype}/{fidx}')
                n = len(eface)
                parts.append((np.broadcast_to(np.int16(efcidx), n),
                              np.arange(n), gi, eface['cidx'], eface['off']))

        return map(np.concatenate, zip(*parts))

    def _construct_con(self):
        cidxmap, cetmap = self._parse_codec()
        g2l = self._build_g2l()
        lcidx, leidx, lgidx, rcidx, rgidx = self._flatten_faces(g2l)

        # Global-to-local lookup for rhs element-neighbour faces
        reidx = np.full(len(lcidx), -1)
        for etidx, etype in enumerate(self.mesh.etypes):
            if etype not in g2l:
                continue

            ordgi, perm = g2l[etype][1:]

            # Select rhs faces whose neighbour is this element type
            mask = cetmap[rcidx] == etidx
            offs = rgidx[mask]

            # Map global element numbers to partition local numbers
            pos = np.searchsorted(ordgi, offs)
            pos = np.clip(pos, 0, len(ordgi) - 1)
            reidx[mask] = np.where(ordgi[pos] == offs, perm[pos], -1)

        # Classify interfaces
        is_boundary, is_local = rgidx == -1, reidx >= 0
        is_mpi = ~(is_boundary | is_local)

        # Deduplicate internal interfaces
        lkey, rkey = self._pack_pairs((lcidx[is_local], leidx[is_local]),
                                      (rcidx[is_local], reidx[is_local]))
        iidxs = np.flatnonzero(is_local)[lkey < rkey]

        con = lambda c, e: Connectivity(c, e, cidxmap)
        self.mesh.con = (con(lcidx[iidxs], leidx[iidxs]),
                         con(rcidx[iidxs], reidx[iidxs]))

        # Boundary connectivity
        for bccidx in np.unique(rcidx[is_boundary]):
            name = self.mesh.codec[bccidx][3:]
            bmask = rcidx == bccidx
            self.mesh.bcon[name] = con(lcidx[bmask], leidx[bmask])

        # MPI connectivity
        if np.any(is_mpi):
            dt = [('cidx', np.int16), ('idx', int)]
            lhs = np.rec.fromarrays([lcidx[is_mpi], lgidx[is_mpi]], dtype=dt)
            rhs = np.rec.fromarrays([rcidx[is_mpi], rgidx[is_mpi]], dtype=dt)

            # Stride for packing (cidx, idx) into collision-free keys
            stride = max(len(self.f[f'eles/{et}']) for et in self.mesh.etypes)

            self._construct_mpi_con(g2l, cetmap, cidxmap, lhs, rhs, stride)

    def _construct_mpi_con(self, g2l, cetmap, cidxmap, lhs, rhs, stride):
        comm, rank, root = get_comm_rank_root()

        # Create a neighbourhood collective communicator
        ncomm = autofree(comm.Create_dist_graph_adjacent(self.neighbours,
                                                         self.neighbours))

        # Build a lookup to match (cidx, offset) pairs against lhs faces
        find = self._pair_finder(lhs, stride)

        # See which of our neighbours' unpaired faces we have
        matches = [rhs[find(u)] for u in ncomm.neighbor_allgather(rhs)]

        # Distribute this information back to our neighbours
        nmatches = ncomm.neighbor_alltoall(matches)

        etypes = self.mesh.etypes
        for nrank, nmatch in zip(self.neighbours, nmatches):
            # Find which of our lhs faces match this neighbour
            idx = find(nmatch)

            # Both ranks must agree on face ordering; sort by the
            # lower-ranked side so each rank's pairing is consistent
            ref = rhs if rank < nrank else lhs
            idx = idx[np.lexsort((ref.idx[idx], ref.cidx[idx]))]

            # Codec and element type indices for matched faces
            cidxs, etidxs = lhs.cidx[idx], cetmap[lhs.cidx[idx]]

            # Convert global element offsets to partition-local indices
            eidxs = np.empty(len(idx), dtype=int)
            for ti in np.unique(etidxs):
                ordgi, perm = g2l[etypes[ti]][1:]
                mask = etidxs == ti
                pos = np.searchsorted(ordgi, lhs.idx[idx[mask]])
                eidxs[mask] = perm[pos]

            self.mesh.con_p[nrank] = Connectivity(cidxs, eidxs, cidxmap)

    def _construct_shared_nodes(self):
        snf = SharedNodesFinder(self.eles, self.mesh.node_idxs,
                                self.mesh.node_valency)
        self.mesh.shared_nodes = snf.compute()
