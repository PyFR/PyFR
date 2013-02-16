# -*- coding: utf-8 -*-

import re

from collections import defaultdict, Counter
from itertools import chain, ifilter, izip

import numpy as np

from pyfr.readers import BaseReader

def msh_section(mshit, section):
    endln = '$End%s\n' % section
    count = int(next(mshit))

    for i,l in enumerate(mshit):
        if l == endln:
            raise ValueError('Unexpected end of section $%s' % section)

        yield l

        if i == count - 1:
            break
    else:
        raise ValueError('Unexpected EOF')

    if next(mshit) != endln:
        raise ValueError('Expected $End%s' % section)

class GmshReader(BaseReader):
    # Supported file types and extensions
    name = 'gmsh'
    extn = ['.msh']

    # Gmsh element types to PyFR type (petype) + sizes
    _etype_map = {3: ('quad', 4), 5: ('hex', 8), 12: ('hex', 27)}

    # Number of nodes per first-order element of each type
    _petype_focount = {'quad': 4, 'hex': 8, 'tet': 4, 'pri': 6, 'pyr': 5}

    # Number of faces of each type per element
    _petype_ftcount = {'hex': {'quad': 6},
                       'tet': {'tri': 4},
                       'pri': {'quad': 3, 'tri': 2},
                       'pyr': {'quad': 1, 'tri': 4}}

    # Supported fixed boundary conditions
    _bctype_map = {'supersonic_inflow': 'sup_inflow',
                   'supersonic_outflow': 'sup_outflow',
                   'subsonic_inflow': 'sub_inflow',
                   'subsonic_outflow': 'sub_outflow',
                   'isothermal_noslip': 'isotherm_noslip'}

    # Remappers (Gmsh node ordering to PyFR node ordering)
    _node_maps = {('hex',  8): np.array([ 0,  1,  3,  2,  4,  5,  7,  6]),
                  ('hex', 27): np.array([ 0,  2,  8,  6, 18, 20, 26, 24,  1,
                                          3,  9,  5, 11,  7, 17, 15, 19, 21,
                                         23, 25,  4, 10, 12, 14, 16, 22, 13])}

    def __init__(self, msh):
        if isinstance(msh, basestring):
            msh = open(msh)

        # Get an iterator over the lines of the mesh
        mshit = iter(msh)

        # Required section readers
        sect_map = {'MeshFormat': self._read_mesh_format,
                    'Nodes': self._read_nodes,
                    'Elements': self._read_eles,
                    'PhysicalNames': self._read_phys_names}
        req_sect = frozenset(sect_map)

        # Seen sections
        seen_sect = set()

        for l in ifilter(lambda l: l != '\n', mshit):
            # Ensure we have encountered a section
            if not l.startswith('$'):
                raise ValueError('Expected a mesh section')

            # Strip the '$' and '\n' to get the section name
            sect = l[1:-1]

            # If the section is known then read it
            if sect in sect_map:
                sect_map[sect](mshit)
                seen_sect.add(sect)
            # Else skip over it
            else:
                endsect = '$End%s\n' % sect

                for el in mshit:
                    if el == endsect:
                        break
                else:
                    raise ValueError('Expected $End%s' % sect)

        # Check that all of the required sections are present
        if seen_sect != req_sect:
            missing = req_sect - seen_sect
            raise ValueError('Required sections: %s not found' % missing)

    def _read_mesh_format(self, mshit):
        ver, ftype, dsize = next(mshit).split()

        if ver != '2.2':
            raise ValueError('Invalid mesh version')
        if ftype != '0':
            raise ValueError('Invalid file type')
        if dsize != '8':
            raise ValueError('Invalid data size')

        if next(mshit) != '$EndMeshFormat\n':
            raise ValueError('Expected $EndMeshFormat')

    def _read_phys_names(self, msh):
        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        #  - periodic faces
        # with each mesh having one set of fluid elements, zero or more
        # boundary faces and up to three periodic boundary face pairs.
        self._felespent = None
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Extract the physical names
        for l in msh_section(msh, 'PhysicalNames'):
            m = re.match(r'^(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError('Malformed physical entity')

            pent, name = int(m.group(2)), m.group(3).lower()

            # Fluid elements
            if name == 'fluid':
                self._felespent = pent
            # Boundary condition faces
            elif name in self._bctype_map:
                bctype = self._bctype_map[name]
                self._bfacespents[bctype] = pent
            # Periodic boundary faces
            elif name.startswith('periodic'):
                p = re.match(r'periodic[ -_]?(x|y|z)[ -_]?(l|r)', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p.group(1)].append(pent)
            else:
                raise ValueError('Invalid physical entity name')

        if self._felespent is None:
            raise ValueError('No fluid elements in mesh')

        if any(len(pf) != 2 for pf in self._pfacespents.itervalues()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def _read_nodes(self, msh):
        self._nodepts = nodepts = {}

        for i,l in enumerate(msh_section(msh, 'Nodes')):
            nv = l.split()
            nodepts[int(nv[0])] = [float(x) for x in nv[1:]]

    def _read_eles(self, msh):
        elenodes = defaultdict(list)
        eleparts = defaultdict(list)

        for l in msh_section(msh, 'Elements'):
            # Extract the raw element data
            elei = map(int, l.split())
            enum, etype, entags = elei[:3]
            etags, enodes = elei[3:3 + entags], elei[3 + entags:]

            if etype not in self._etype_map:
                raise ValueError('Unsupported element type %d' % etype)

            # Physical entity type (used for BCs)
            epent = etags[0]

            # Determine the partition number (defaults to 0)
            epart = etags[3] - 1 if entags > 2 else 0

            elenodes[(etype, epent)].append(enodes)
            eleparts[(etype, epent)].append(epart)

        self._elenodes = {k: np.array(v) for k,v in elenodes.iteritems()}
        self._eleparts = {k: np.array(v) for k,v in eleparts.iteritems()}

    def _to_first_order(self, elemap):
        foelemap = {}
        for (etype, epent), eles in elemap.iteritems():
            # PyFR element type ('hex', 'tri', &c)
            petype = self._etype_map[etype][0]

            # Number of nodes in the first-order representation
            focount = self._petype_focount[petype]

            foelemap[(petype, epent)] = eles[:,:focount]

        return foelemap

    def _split_fluid(self, elemap):
        selemap = defaultdict(dict)

        for (petype, epent), eles in elemap.iteritems():
            selemap[epent][petype] = eles

        return selemap.pop(self._felespent), selemap


    def _extract_faces(self, foeles):
        extractors = {'hex': self._extract_faces_hex}

        fofaces = defaultdict(list)
        for petype, eles in foeles.iteritems():
            for pftype, faces in extractors[petype](eles):
                fofaces[pftype].append(faces.ravel())

        return fofaces

    def _foface_array(self, peletype, pftype, neles):
        # Number of nodes per face and number of faces of this type per ele
        nfnodes = self._petype_focount[pftype]
        nfcount = self._petype_ftcount[peletype][pftype]

        dtype = [('petype', 'S4'), ('eidx', 'i4'), ('fidx', 'i1'),
                 ('rtag', 'i1'), ('nodes', 'i4', nfnodes)]

        arr = np.recarray((neles, nfcount), dtype=dtype)
        arr.petype = peletype
        arr.rtag = 0

        return arr

    def _extract_faces_hex(self, fohexes):
        # Gmsh nodes offsets for each of the six faces
        fnmap = np.array([[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                          [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]])

        qf = self._foface_array('hex', 'quad', len(fohexes))

        qf.eidx = np.arange(len(fohexes))[...,None]
        qf.fidx = np.arange(6)
        qf.nodes = fohexes[:,fnmap]

        return [('quad', qf)]

    def _pair_fluid_faces(self, ffofaces):
        pairs = defaultdict(list)
        resid = {}

        for pftype, faces in ffofaces.iteritems():
            for f in chain(*faces):
                sn = tuple(sorted(f.nodes))

                # See if the nodes are in resid
                if sn in resid:
                    pairs[pftype].append((resid.pop(sn), f))
                # Otherwise add them to the unpaired dict
                else:
                    resid[sn] = f

        return pairs, resid

    def _calc_fluid_rtags(self, fpairs):
        rtaggers = {'quad': self._calc_fluid_rtags_quad}

        for pftype, pairs in fpairs.iteritems():
            rtaggers[pftype](pairs)

    def _calc_fluid_rtags_quad(self, qpairs):
        # To standard order
        hexlut = np.array([[1, 0, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
                           [0, 1, 2, 3], [1, 0, 3, 2], [0, 1, 3, 2]])

        for l, r in qpairs:
            ordl = l.nodes[hexlut[l.fidx]]
            ordr = r.nodes[hexlut[r.fidx]]

            # Rotation tag is index of ordl[0] in ordr mod four
            r.rtag = np.where(ordr==ordl[0])[0][0] % 4

    def _pair_periodic_fluid_faces(self, bpart, resid):
        pfaces = defaultdict(list)

        for lpent, rpent in self._pfacespents.itervalues():
            for pftype in bpart[lpent]:
                lfnodes = bpart[lpent][pftype]
                rfnodes = bpart[rpent][pftype]

                for lfn, rfn in izip(lfnodes, rfnodes):
                    lf = resid.pop(tuple(sorted(lfn)))
                    rf = resid.pop(tuple(sorted(rfn)))

                    pfaces[pftype].append((lf, rf))

        return pfaces

    def _calc_periodic_fluid_rtags(self, pfpairs):
        rtaggers = {'quad': self._calc_periodic_fluid_rtags_quad}

        for pftype, pairs in pfpairs.iteritems():
            rtaggers[pftype](pairs)

    def _calc_periodic_fluid_rtags_quad(self, qpairs):
        # TODO: Compute this properly
        for l, r in qpairs:
            r.rtag = 1

    def _ident_boundary_faces(self, bpart, resid):
        bfaces = defaultdict(list)

        bpents = set(self._bfacespents.itervalues())

        for epent, fnodes in bpart.iteritems():
            if epent in bpents:
                for fn in chain.from_iterable(fnodes.itervalues()):
                    bfaces[epent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def _partition_pairs(self, pairs, bcf):
        con_px = defaultdict(list)
        con_pxpy = defaultdict(list)
        bcon_px = defaultdict(list)

        # Connectivity in PyFR is specified in terms of partition-local
        # element numbers.  As the element indices in pairs are
        # global it is first necessary to produce a global-to-local
        # mapping for each element type.
        eleglmap = defaultdict(list)
        pcounter = Counter()

        feleparts = self._split_fluid(self._eleparts)[0]
        for etype, eleps in feleparts.iteritems():
            petype = self._etype_map[etype][0]

            for p in eleps:
                eleglmap[petype].append((p, pcounter[(petype, p)]))
                pcounter[(petype, p)] += 1

        # Generate the face connectivity
        for l, r in pairs:
            lpart, leidx = eleglmap[l.petype][l.eidx]
            rpart, reidx = eleglmap[r.petype][r.eidx]

            conl = (l.petype, leidx, l.fidx, l.rtag)
            conr = (r.petype, reidx, r.fidx, r.rtag)

            if lpart == rpart:
                con_px[lpart].append([conl, conr])
            else:
                con_pxpy[(lpart, rpart)].append(conl)
                con_pxpy[(rpart, lpart)].append(conr)

        # Generate boundary conditions
        for pbctype, pent in self._bfacespents.iteritems():
            for l in bcf[pent]:
                lpart, leidx = eleglmap[l.petype][l.eidx]
                conl = (l.petype, leidx, l.fidx, 0)

                bcon_px[(pbctype, lpart)].append(conl)

        return con_px, con_pxpy, bcon_px

    def _gen_connectivity(self):
        # For connectivity a first-order representation is sufficient
        eles = self._to_first_order(self._elenodes)

        # Split into fluid and boundary parts
        fpart, bpart = self._split_fluid(eles)

        # Extract the faces of the fluid elements
        ffaces = self._extract_faces(fpart)

        # Pair the fluid-fluid faces
        fpairs, resid = self._pair_fluid_faces(ffaces)

        # Identify periodic boundary face pairs
        pfpairs = self._pair_periodic_fluid_faces(bpart, resid)

        # Identify the fixed boundary faces
        bf = self._ident_boundary_faces(bpart, resid)

        if any(resid.itervalues()):
            raise ValueError('Unpaired faces in mesh')

        # Determine the rotation tags of these fluid-fluid faces
        self._calc_fluid_rtags(fpairs)
        self._calc_periodic_fluid_rtags(pfpairs)

        # Flattern the face-pair dicts
        pairs = chain(chain.from_iterable(fpairs.itervalues()),
                      chain.from_iterable(pfpairs.itervalues()))

        # Process these face pairs into the connectivity arrays
        con_px, con_pxpy, bcon_px = self._partition_pairs(pairs, bf)

        # Output
        retcon = {}

        for k, v in con_px.iteritems():
            retcon['con_p%d' % k] = np.array(v, dtype='S5,i4,i1,i1').T

        for k, v in con_pxpy.iteritems():
            retcon['con_p%dp%d' % k] = np.array(v, dtype='S5,i4,i1,i1')

        for k, v in bcon_px.iteritems():
            retcon['bcon_%s_p%d' % k] = np.array(v, dtype='S5,i4,i1,i1')

        return retcon

    def _gen_shape_points(self):
        spts = defaultdict(list)

        # Global node map (node index to coords)
        nodepts = self._nodepts

        for etype, pent in self._elenodes:
            if pent != self._felespent:
                continue

            # Elements and corresponding partition numbers
            eles = self._elenodes[(etype, pent)]
            prts = self._eleparts[(etype, pent)]

            petype, nnodes = self._etype_map[etype]

            # Go from Gmsh to PyFR node ordering
            peles = eles[:,self._node_maps[(petype, nnodes)]]

            for n, p in izip(peles, prts):
                spts[(petype, p)].append([nodepts[i] for i in n])

        return {'spt_%s_p%d' % k: np.array(arr).swapaxes(0, 1)
                for k, arr in spts.iteritems()}

    def _to_raw_pyfrm(self):
        rawm = {}
        rawm.update(self._gen_connectivity())
        rawm.update(self._gen_shape_points())
        return rawm

