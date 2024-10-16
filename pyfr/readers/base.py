from collections import defaultdict
from itertools import chain
from uuid import UUID

import h5py
import numpy as np

from pyfr._version import __version__
from pyfr.nputil import iter_struct, fuzzysort
from pyfr.polys import get_polybasis
from pyfr.progress import NullProgressSequence
from pyfr.shapes import BaseShape
from pyfr.util import digest, first, subclass_where


class BaseReader:
    def __init__(self, progress):
        self.progress = progress

    def _get_default_partitioning(self, eles, codec):
        # Allocate the partitioning array
        partitioning = np.empty(sum(len(einfo) for einfo in eles.values()),
                                dtype=np.int64)

        # Allocate the partition region array
        pregions = np.zeros((1, len(eles) + 1), dtype=np.int64)

        poff = 0
        for i, etype in enumerate(sorted(eles)):
            einfo = eles[etype]

            # Move curved elements to the start of the list
            ceidx = einfo['curved'].nonzero()[0]
            leidx = (~einfo['curved']).nonzero()[0]

            # Add them to the array
            if len(ceidx):
                partitioning[poff:poff + len(ceidx)] = ceidx
                poff += len(ceidx)
            if len(leidx):
                partitioning[poff:poff + len(leidx)] = leidx
                poff += len(leidx)

            # Note the offsets
            pregions[0, i + 1] = poff

        return partitioning, pregions

    def _get_nodes(self, nodes, eles):
        # Allocate the new nodes table
        dtype = [('location', float, nodes.shape[1]), ('valency', np.uint16)]
        nnodes = np.zeros(len(nodes), dtype=dtype)

        # Copy over the nodes
        nnodes['location'] = nodes

        # Tally up the valencies
        for etype, ele in eles.items():
            k, v = np.unique(ele['nodes'], return_counts=True)
            nnodes['valency'][k] += v.astype(np.uint16)

        return nnodes

    def _to_raw_mesh(self, lintol):
        pass

    def write(self, fname, lintol):
        nodes, eles, codec, pmap = mesh = self._to_raw_mesh(lintol)

        # Compute the UUID
        with self.progress.start('Computing UUID'):
            uuid = UUID(digest(mesh)[:32])

        # Write out the file
        with self.progress.start('Writing mesh'):
            with h5py.File(fname, 'w', libver='latest') as f:
                # Generate a default partitioning
                parts, pregions = self._get_default_partitioning(eles, codec)

                # Write out the metadata
                f['codec'] = np.array(codec, dtype='S')
                f['creator'] = np.array(f'pyfr {__version__}', dtype='S')
                f['mesh-uuid'] = np.array(str(uuid), dtype='S')
                f['version'] = 1

                # Write out the nodes
                f['nodes'] = self._get_nodes(nodes, eles)

                # Write out the elements
                for etype, einfo in eles.items():
                    f[f'eles/{etype}'] = einfo

                # Write out the periodic boundary information
                for pname, pidx in pmap.items():
                    f[f'periodic/{pname}'] = pidx

                # Write out the partitioning
                f['partitionings/1/eles'] = parts
                f['partitionings/1/eles'].attrs['regions'] = pregions


class NodalMeshAssembler:
    # Dimensionality of each element type
    _petype_ndim = {'tri': 2, 'quad': 2,
                    'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}

    # Face numberings for each element type
    _petype_fnums = {
        'tri': {'line': [0, 1, 2]},
        'quad': {'line': [0, 1, 2, 3]},
        'tet': {'tri': [0, 1, 2, 3]},
        'hex': {'quad': [0, 1, 2, 3, 4, 5]},
        'pri': {'quad': [2, 3, 4], 'tri': [0, 1]},
        'pyr': {'quad': [0], 'tri': [1, 2, 3, 4]}
    }

    # Number of nodes in the first-order representation an element
    _petype_focount = {'line': 2, 'tri': 3, 'quad': 4,
                       'tet': 4, 'pyr': 5, 'pri': 6, 'hex': 8}

    def __init__(self, nodepts, elenodes, pents, maps):
        self._nodepts = nodepts
        self._elenodes = elenodes
        self._felespent, self._bfacespents, self._pfacespents = pents
        self._etype_map, self._petype_fnmap, self._nodemaps = maps

    def _check_pyr_parallelogram(self, foeles):
        # Find PyFR node map for the quad face
        fnmap = self._petype_fnmap['pyr']['quad'][0]
        pfnmap = [self._nodemaps['quad', 4][i] for i in fnmap]

        # Face nodes
        fpts = self._nodepts[foeles[:, pfnmap]].swapaxes(0, 1)

        # Check if parallelogram or not
        if np.any(np.abs(fpts[0] - fpts[1] - fpts[2] + fpts[3]) > 1e-10):
            raise ValueError('Pyramids with non-parallelogram bases are '
                             'currently unsupported')

    def _to_first_order(self, elemap):
        foelemap = {}
        for (etype, epent), eles in elemap.items():
            # PyFR element type ('hex', 'tri', &c)
            petype = self._etype_map[etype][0]

            # Number of nodes in the first-order representation
            focount = self._petype_focount[petype]

            foelemap[petype, epent] = eles[:, :focount]

            # Check if pyramids have a parallelogram base or not
            if petype == 'pyr':
                self._check_pyr_parallelogram(foelemap[petype, epent])

        return foelemap

    def _split_fluid(self, elemap):
        selemap = defaultdict(dict)

        for (petype, epent), eles in elemap.items():
            selemap[epent][petype] = eles

        return selemap.pop(self._felespent), selemap

    def _foface_info(self, petype, pftype, codec, foeles):
        # Face numbers of faces of this type on this element
        fnums = np.array(self._petype_fnums[petype][pftype])

        # Lookup these faces in the codec
        cidx = np.array([codec.index(f'eles/{petype}/{f}') for f in fnums])

        # First-order nodes associated with this face type
        fnmap = self._petype_fnmap[petype][pftype]

        # Element, face, and first-order node count
        neles, nfaces, nnodes = len(foeles), len(fnums), len(fnmap[0])

        # Extract the first-order nodes
        nodes = np.sort(foeles[:, fnmap]).reshape(-1, nnodes)

        # To improve face-pairing performance sort the faces by their
        # node numbers such that any connected faces are adjacent
        nodeix = np.lexsort(nodes.T)
        nodes = nodes[nodeix].view([('', nodes.dtype)]*nnodes).squeeze()

        # Generate the associated connectivity information
        eidx, fidx = divmod(nodeix, nfaces)

        return petype, (cidx[fidx], fnums[fidx], eidx), nodes

    def _codec_conn(self, eles, codec):
        cconn = [None]*len(codec)
        for petype, einfo in eles.items():
            for i, fcon in enumerate(einfo['faces'].T):
                cconn[codec.index(f'eles/{petype}/{i}')] = fcon

        return cconn

    def _extract_faces(self, foeles, codec):
        fofaces = defaultdict(list)

        for petype, eles in foeles.items():
            for pftype in self._petype_fnums[petype]:
                fofinf = self._foface_info(petype, pftype, codec, eles)
                fofaces[pftype].append(fofinf)

        return fofaces

    def _pair_fluid_faces(self, ffofaces, codec, eles):
        # Map from codec numbers to per-element face connectivity arrays
        cconn = self._codec_conn(eles, codec)

        resid = {}

        for pftype, faces in ffofaces.items():
            for petype, (cidx, fidx, eidx), nodes in faces:
                # Pair adjacent elements
                padj = nodes[:-1] == nodes[1:]
                plft = np.concatenate((padj, [False]))
                prgt = np.concatenate(([False], padj))

                # Index into the cidx and off arrays for this element type
                ecidx = eles[petype]['faces']['cidx']
                eoff = eles[petype]['faces']['off']

                # Construct the index arrays for the left and right hand sides
                lf = eidx[plft], fidx[plft]
                rf = eidx[prgt], fidx[prgt]

                # With these, pair the adjacent faces
                ecidx[lf], eoff[lf] = cidx[prgt], eidx[prgt]
                ecidx[rf], eoff[rf] = cidx[plft], eidx[plft]

                # Mask the paired faces
                mask = ~(plft | prgt)

                # Use a lookup table to pair residual faces
                con = np.column_stack([cidx[mask], eidx[mask]])
                con = con.view([('', cidx.dtype)]*2).squeeze()
                for rf, n in zip(iter_struct(con), iter_struct(nodes[mask])):
                    # If the nodes are in resid then pair the faces
                    if (lf := resid.pop(n, None)):
                        lcidx, loff = lf
                        rcidx, roff = rf

                        cconn[lcidx][loff] = rf
                        cconn[rcidx][roff] = lf
                    # Otherwise add them to the unpaired dict
                    else:
                        resid[n] = rf

        return resid, cconn

    def _pair_periodic_fluid_faces(self, bpart, cconn, resid):
        pmap = {}
        pdtype = [('cidx', np.int16), ('off', np.int64)]

        for k, (lpent, rpent) in self._pfacespents.items():
            plist = []

            for pftype in bpart[lpent]:
                lfnodes = bpart[lpent][pftype]
                rfnodes = bpart[rpent][pftype]

                lfpts = self._nodepts[lfnodes]
                rfpts = self._nodepts[rfnodes]

                lfidx = fuzzysort(lfpts.mean(axis=1).T, range(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, range(len(rfnodes)))

                for lfn, rfn in zip(lfnodes[lfidx], rfnodes[rfidx]):
                    lf = lcidx, loff = resid.pop(tuple(sorted(lfn)))
                    rf = rcidx, roff = resid.pop(tuple(sorted(rfn)))

                    cconn[lcidx][loff] = rf
                    cconn[rcidx][roff] = lf

                    plist.append([lf, rf])

            pmap[k] = np.array(plist, dtype=pdtype)

        return pmap

    def _ident_boundary_faces(self, bpart, cconn, codec, resid):
        # Create a map from boundary entities to names
        bpents = {v: k for k, v in self._bfacespents.items()}

        for epent, fnodes in bpart.items():
            if epent in bpents:
                # Obtain the codec number associated with this boundary
                cidx = codec.index(f'bc/{bpents[epent]}')

                # Tag the associated faces
                for fn in chain.from_iterable(fnodes.values()):
                    lcidx, loff = resid.pop(tuple(sorted(fn)))

                    cconn[lcidx][loff] = cidx, -1

    def get_eles(self, lintol, progress=NullProgressSequence()):
        eles, codec = {}, []

        with progress.start('Creating elements'):
            for etype, pent in sorted(self._elenodes):
                if pent != self._felespent:
                    continue

                # Elements and type information
                enodes = self._elenodes[etype, pent]
                petype, nnodes = self._etype_map[etype]

                # Determine the number of faces
                fnmap = self._petype_fnmap[petype]
                nfaces = sum(len(fn) for fn in fnmap.values())

                # Add the element type to the codec
                codec.append(f'eles/{petype}')

                # Add the face info to the codec
                codec.extend(f'eles/{petype}/{i}' for i in range(nfaces))

                # Elements array data type
                fdtype = [('cidx', np.int16), ('off', np.int64)]
                edtype = [('nodes', np.int64, nnodes), ('curved', bool),
                          ('faces', fdtype, nfaces)]

                # Allocate the elements array
                eles[petype] = einfo = np.empty(len(enodes), dtype=edtype)

                # Map and copy over the node numbers
                einfo['nodes'] = enodes[:, self._nodemaps[petype, nnodes]]

        # Add the boundary conditions to the codec
        codec.extend(f'bc/{bname}' for bname in self._bfacespents)

        # Add in connectivity information
        with progress.start_with_spinner('Connecting elements') as spinner:
            pmap = self._connect_eles(eles, codec, spinner)

        # Apply linearisation
        with progress.start_with_spinner('Linearising elements') as spinner:
            nodepts = self._linearise_eles(eles, lintol, spinner)

        return nodepts, eles, codec, pmap

    def _connect_eles(self, eles, codec, spinner):
        # For connectivity a first-order representation is sufficient
        foeles = self._to_first_order(self._elenodes)
        spinner()

        # Split into fluid and boundary parts
        fpart, bpart = self._split_fluid(foeles)
        spinner()

        # Extract the faces of the first-order fluid elements
        ffofaces = self._extract_faces(fpart, codec)
        spinner()

        # Pair the fluid-fluid faces
        resid, cconn = self._pair_fluid_faces(ffofaces, codec, eles)
        spinner()

        # Tag and pair periodic boundary faces
        pmap = self._pair_periodic_fluid_faces(bpart, cconn, resid)
        spinner()

        # Identify the fixed boundary faces
        self._ident_boundary_faces(bpart, cconn, codec, resid)
        spinner()

        if any(resid.values()):
            raise ValueError('Unpaired faces in mesh')

        return pmap

    def _linearise_eles(self, emap, lintol, spinner):
        # Create a copy of the node points
        ndim = self._petype_ndim[first(emap)]
        nodepts = self._nodepts[:, :ndim].copy()

        for petype, einfo in emap.items():
            elesix = einfo['nodes']
            nnodes = elesix.shape[-1]

            # Construct the element array
            eles = nodepts[elesix].swapaxes(0, 1)

            # Generate the associated polynomial bases
            shape = subclass_where(BaseShape, name=petype)
            order = shape.order_from_npts(nnodes)
            hbasis = get_polybasis(petype, order + 1, shape.std_ele(order))
            lbasis = get_polybasis(petype, 2, shape.std_ele(1))

            htol = hbasis.nodal_basis_at(lbasis.pts)
            ltoh = lbasis.nodal_basis_at(hbasis.pts)

            leles = (ltoh @ htol) @ eles.reshape(nnodes, -1)
            leles = leles.reshape(nnodes, -1, ndim)

            # Use this to determine which elements are linear
            num = np.max(np.abs(eles - leles), axis=0)
            den = np.max(eles, axis=0) - np.min(eles, axis=0)
            lin = np.all(num / den < lintol, axis=1)

            # Snap the nodes associated with linear elements
            nodepts[elesix[lin]] = leles[:, lin].swapaxes(0, 1)

            # Note which elements are curved
            einfo['curved'] = ~lin

            spinner()

        return nodepts
