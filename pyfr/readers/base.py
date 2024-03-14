from collections import defaultdict
from itertools import chain
from uuid import UUID

import numpy as np

from pyfr.nputil import iter_struct, fuzzysort
from pyfr.polys import get_polybasis
from pyfr.progress import NullProgressSpinner
from pyfr.shapes import BaseShape
from pyfr.util import digest, subclass_where


class BaseReader:
    def __init__(self):
        pass

    def _to_raw_pyfrm(self, lintol):
        pass

    def to_pyfrm(self, lintol):
        mesh = self._to_raw_pyfrm(lintol)

        # Add metadata
        mesh['mesh_uuid'] = np.array(str(UUID(digest(mesh)[:32])), dtype='S')

        return mesh


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

    def _foface_info(self, petype, pftype, foeles):
        # Face numbers of faces of this type on this element
        fnums = np.array(self._petype_fnums[petype][pftype])

        # First-order nodes associated with this face
        fnmap = self._petype_fnmap[petype][pftype]

        # Element, face, and first-order node count
        neles, nfaces, nnodes = len(foeles), len(fnums), len(fnmap[0])

        # Extract the first-order nodes
        nodes = np.sort(foeles[:, fnmap]).reshape(-1, nnodes)

        # To improve face-pairing performance sort the faces
        nodeix = np.argsort(nodes, axis=0)[:, 0]
        nodes = nodes[nodeix].view([('', nodes.dtype)]*nnodes).squeeze()

        # Generate the connectivity
        eidx, fidx = divmod(nodeix, nfaces)
        con = [(petype, e, f, 0)
               for e, f in np.column_stack([eidx, fnums[fidx]]).tolist()]

        return con, nodes

    def _extract_faces(self, foeles):
        fofaces = defaultdict(list)

        for petype, eles in foeles.items():
            for pftype in self._petype_fnums[petype]:
                fofinf = self._foface_info(petype, pftype, eles)
                fofaces[pftype].append(fofinf)

        return fofaces

    def _pair_fluid_faces(self, ffofaces):
        pairs = defaultdict(list)
        resid = {}

        for pftype, faces in ffofaces.items():
            append, pop = pairs[pftype].append, resid.pop
            fnit = (zip(f, iter_struct(n)) for f, n in faces)
            for f, n in chain.from_iterable(fnit):
                # See if the nodes are in resid
                if (lf := pop(n, None)):
                    append([lf, f])
                # Otherwise add them to the unpaired dict
                else:
                    resid[n] = f

        return pairs, resid

    def _pair_periodic_fluid_faces(self, bpart, resid):
        pfaces, pmap = defaultdict(list), {}

        for k, (lpent, rpent) in self._pfacespents.items():
            for pftype in bpart[lpent]:
                lfnodes = bpart[lpent][pftype]
                rfnodes = bpart[rpent][pftype]

                lfpts = self._nodepts[lfnodes]
                rfpts = self._nodepts[rfnodes]

                lfidx = fuzzysort(lfpts.mean(axis=1).T, range(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, range(len(rfnodes)))

                for lfn, rfn in zip(lfnodes[lfidx], rfnodes[rfidx]):
                    lf = resid.pop(tuple(sorted(lfn)))
                    rf = resid.pop(tuple(sorted(rfn)))

                    pfaces[pftype].append([lf, rf])
                    pmap[lf, rf] = k

        return pfaces, pmap

    def _ident_boundary_faces(self, bpart, resid):
        bfaces = defaultdict(list)

        bpents = set(self._bfacespents.values())

        for epent, fnodes in bpart.items():
            if epent in bpents:
                for fn in chain.from_iterable(fnodes.values()):
                    bfaces[epent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def get_connectivity(self, spinner=NullProgressSpinner()):
        # For connectivity a first-order representation is sufficient
        eles = self._to_first_order(self._elenodes)
        spinner()

        # Split into fluid and boundary parts
        fpart, bpart = self._split_fluid(eles)
        spinner()

        # Extract the faces of the fluid elements
        ffaces = self._extract_faces(fpart)
        spinner()

        # Pair the fluid-fluid faces
        fpairs, resid = self._pair_fluid_faces(ffaces)
        spinner()

        # Tag and pair periodic boundary faces
        pfpairs, pmap = self._pair_periodic_fluid_faces(bpart, resid)
        spinner()

        # Identify the fixed boundary faces
        bf = self._ident_boundary_faces(bpart, resid)
        spinner()

        if any(resid.values()):
            raise ValueError('Unpaired faces in mesh')

        # Flattern the face-pair dicts
        pairs = chain(chain.from_iterable(fpairs.values()),
                      chain.from_iterable(pfpairs.values()))

        # Generate the internal connectivity array
        con = list(pairs)

        # Extract the names of periodic interfaces
        con_pnames = defaultdict(list)
        for i, (l, r) in enumerate(con):
            if (l, r) in pmap:
                con_pnames[pmap[l, r]].append(i)

        # Generate boundary condition connectivity arrays
        bcon = {}
        for pbcrgn, pent in self._bfacespents.items():
            bcon[pbcrgn] = bf[pent]

        spinner()

        # Output
        ret = {'con_p0': np.array(con, dtype='S4,i8,i1,i2').T}

        for k, v in con_pnames.items():
            ret['con_p0', f'periodic_{k}'] = np.array(v, dtype=np.int64)

        for k, v in bcon.items():
            ret[f'bcon_{k}_p0'] = np.array(v, dtype='S4,i8,i1,i2')

        return ret

    def _linearise_eles(self, lintol, spinner):
        lidx = {}

        for etype, pent in self._elenodes:
            if pent != self._felespent:
                continue

            # Elements and type information
            elesix = self._elenodes[etype, pent]
            petype, nnodes = self._etype_map[etype]

            # Obtain the dimensionality of the element type
            ndim = self._petype_ndim[petype]

            # Node maps between input and PyFR orderings
            itop = self._nodemaps[petype, nnodes]
            ptoi = np.argsort(itop)

            # Construct the element array
            eles = self._nodepts[elesix[:, itop], :ndim].swapaxes(0, 1)

            # Generate the associated polynomial bases
            shape = subclass_where(BaseShape, name=petype)
            order = shape.order_from_nspts(nnodes)
            hbasis = get_polybasis(petype, order, shape.std_ele(order - 1))
            lbasis = get_polybasis(petype, 2, shape.std_ele(1))

            htol = hbasis.nodal_basis_at(lbasis.pts)
            ltoh = lbasis.nodal_basis_at(hbasis.pts)

            leles = (ltoh @ htol) @ eles.reshape(nnodes, -1)
            leles = leles.reshape(nnodes, -1, ndim)

            # Use this to determine which elements are linear
            num = np.max(np.abs(eles - leles), axis=0)
            den = np.max(eles, axis=0) - np.min(eles, axis=0)
            lin = lidx[petype] = np.all(num / den < lintol, axis=1)

            for ix in np.nonzero(lin)[0]:
                self._nodepts[elesix[ix], :ndim] = leles[ptoi, ix]

            spinner()

        return lidx

    def get_shape_points(self, lintol, spinner=NullProgressSpinner()):
        spts = {}

        # Apply tolerance-based linearisation to the elements
        lidx = self._linearise_eles(lintol, spinner)

        for etype, pent in self._elenodes:
            if pent != self._felespent:
                continue

            # Elements and type information
            eles = self._elenodes[etype, pent]
            petype, nnodes = self._etype_map[etype]

            # Go from Gmsh to PyFR node ordering
            peles = eles[:, self._nodemaps[petype, nnodes]]

            # Obtain the dimensionality of the element type
            ndim = self._petype_ndim[petype]

            # Build the array
            arr = self._nodepts[peles, :ndim].swapaxes(0, 1)

            spts[f'spt_{petype}_p0'] = arr
            spts[f'spt_{petype}_p0', 'linear'] = lidx[petype]

            spinner()

        return spts
