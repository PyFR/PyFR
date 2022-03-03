# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
import uuid

import numpy as np

from pyfr.nputil import fuzzysort
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class BaseReader:
    def __init__(self):
        pass

    def _to_raw_pyfrm(self, lintol):
        pass

    def to_pyfrm(self, lintol):
        mesh = self._to_raw_pyfrm(lintol)

        # Add metadata
        mesh['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')

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
        fnums = self._petype_fnums[petype][pftype]

        # First-order nodes associated with this face
        fnmap = self._petype_fnmap[petype][pftype]

        # Connectivity: (petype, eidx, fidx, flags)
        con = [(petype, i, j, 0) for i in range(len(foeles)) for j in fnums]

        # Nodes
        nodes = np.sort(foeles[:, fnmap]).reshape(len(con), -1)

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
            for f, n in chain.from_iterable(zip(f, n) for f, n in faces):
                sn = tuple(n)

                # See if the nodes are in resid
                if sn in resid:
                    pairs[pftype].append([resid.pop(sn), f])
                # Otherwise add them to the unpaired dict
                else:
                    resid[sn] = f

        return pairs, resid

    def _pair_periodic_fluid_faces(self, bpart, resid):
        pfaces = defaultdict(list)

        for k, (lpent, rpent) in self._pfacespents.items():
            for pftype in bpart[lpent]:
                lfnodes = bpart[lpent][pftype]
                rfnodes = bpart[rpent][pftype]

                lfpts = self._nodepts[lfnodes]
                rfpts = self._nodepts[rfnodes]

                lfidx = fuzzysort(lfpts.mean(axis=1).T, range(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, range(len(rfnodes)))

                for lfn, rfn in zip(lfnodes[lfidx], rfnodes[rfidx]):
                    # Add periodic face flags
                    flg = int(k) + 1

                    # Left = +, right = -
                    lf = resid.pop(tuple(sorted(lfn)))[:-1] + (flg,)
                    rf = resid.pop(tuple(sorted(rfn)))[:-1] + (-flg,)

                    pfaces[pftype].append([lf, rf])

        return pfaces

    def _ident_boundary_faces(self, bpart, resid):
        bfaces = defaultdict(list)

        bpents = set(self._bfacespents.values())

        for epent, fnodes in bpart.items():
            if epent in bpents:
                for fn in chain.from_iterable(fnodes.values()):
                    bfaces[epent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def get_connectivity(self):
        # For connectivity a first-order representation is sufficient
        eles = self._to_first_order(self._elenodes)

        # Split into fluid and boundary parts
        fpart, bpart = self._split_fluid(eles)

        # Extract the faces of the fluid elements
        ffaces = self._extract_faces(fpart)

        # Pair the fluid-fluid faces
        fpairs, resid = self._pair_fluid_faces(ffaces)

        # Tag and pair periodic boundary faces
        pfpairs = self._pair_periodic_fluid_faces(bpart, resid)

        # Identify the fixed boundary faces
        bf = self._ident_boundary_faces(bpart, resid)

        if any(resid.values()):
            raise ValueError('Unpaired faces in mesh')

        # Flattern the face-pair dicts
        pairs = chain(chain.from_iterable(fpairs.values()),
                      chain.from_iterable(pfpairs.values()))

        # Generate the internal connectivity array
        con = list(pairs)

        # Generate boundary condition connectivity arrays
        bcon = {}
        for pbcrgn, pent in self._bfacespents.items():
            bcon[pbcrgn] = bf[pent]

        # Output
        ret = {'con_p0': np.array(con, dtype='S4,i4,i1,i2').T}

        for k, v in bcon.items():
            ret[f'bcon_{k}_p0'] = np.array(v, dtype='S4,i4,i1,i2')

        return ret

    def _linearise_eles(self, lintol):
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

        return lidx

    def get_shape_points(self, lintol):
        spts = {}

        # Apply tolerance-based linearisation to the elements
        lidx = self._linearise_eles(lintol)

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

        return spts
