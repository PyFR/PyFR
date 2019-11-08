# -*- coding: utf-8 -*-

from collections import defaultdict
from ctypes import POINTER, create_string_buffer, c_char_p, c_int, c_void_p
import re

import numpy as np

from pyfr.ctypesutil import load_library
from pyfr.readers import BaseReader, NodalMeshAssembler
from pyfr.readers.nodemaps import CGNSNodeMaps


# Possible CGNS exception types
CGNSError = type('CGNSError', (Exception,), {})
CGNSNodeNotFound = type('CGNSNodeNotFound', (CGNSError,), {})
CGNSIncorrectPath = type('CGNSIncorrectPath', (CGNSError,), {})
CGNSNoIndexDim = type('CGNSNoIndexDim', (CGNSError,), {})


class CGNSWrappers(object):
    # Possible return codes
    _statuses = {
        -1: CGNSError,
        -2: CGNSNodeNotFound,
        -3: CGNSIncorrectPath,
        -4: CGNSNoIndexDim
    }

    def __init__(self):
        # Load CGNS
        self.lib = lib = load_library('cgns')

        # Constants (from cgnslib.h)
        self.CG_MODE_READ = 0
        self.RealDouble = 4
        self.Unstructured = 3
        self.PointRange, self.ElementRange = 4, 6

        # cg_open
        lib.cg_open.argtypes = [c_char_p, c_int, POINTER(c_int)]
        lib.cg_open.errcheck = self._errcheck

        # cg_close
        lib.cg_close.argtypes = [c_int]
        lib.cg_close.errcheck = self._errcheck

        # cg_base_read
        lib.cg_base_read.argtypes = [c_int, c_int, c_char_p, POINTER(c_int),
                                     POINTER(c_int)]
        lib.cg_base_read.errcheck = self._errcheck

        # cg_nzones
        lib.cg_nzones.argtypes = [c_int, c_int, POINTER(c_int)]
        lib.cg_nzones.errcheck = self._errcheck

        # cg_zone_read
        lib.cg_zone_read.argtypes = [c_int, c_int, c_int, c_char_p,
                                     POINTER(c_int)]
        lib.cg_zone_read.errcheck = self._errcheck

        # cg_zone_type
        lib.cg_zone_type.argtypes = [c_int, c_int, c_int, POINTER(c_int)]
        lib.cg_zone_type.errcheck = self._errcheck

        # cg_coord_read
        lib.cg_coord_read.argtypes = [
            c_int, c_int, c_int, c_char_p, c_int, POINTER(c_int),
            POINTER(c_int), c_void_p
        ]
        lib.cg_coord_read.errcheck = self._errcheck

        # cg_nbocos
        lib.cg_nbocos.argtypes = [c_int, c_int, c_int, POINTER(c_int)]
        lib.cg_nbocos.errcheck = self._errcheck

        # cg_boco_info
        lib.cg_boco_info.argtypes = [
            c_int, c_int, c_int, c_int, c_char_p, POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int)
        ]
        lib.cg_boco_info.errcheck = self._errcheck

        # cg_boco_read
        lib.cg_boco_read.argtypes = [c_int, c_int, c_int, c_int,
                                     POINTER(c_int), c_void_p]
        lib.cg_boco_read.errcheck = self._errcheck

        # cg_nsections
        lib.cg_nsections.argtypes = [c_int, c_int, c_int, POINTER(c_int)]
        lib.cg_nsections.errcheck = self._errcheck

        # cg_section_read
        lib.cg_section_read.argtypes = [
            c_int, c_int, c_int, c_int, c_char_p, POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)
        ]
        lib.cg_section_read.errcheck = self._errcheck

        # cg_ElementDataSize
        lib.cg_ElementDataSize.argtypes = [c_int, c_int, c_int, c_int,
                                           POINTER(c_int)]
        lib.cg_ElementDataSize.errcheck = self._errcheck

        # cg_elements_read
        lib.cg_elements_read.argtypes = [c_int, c_int, c_int, c_int,
                                         c_void_p, c_void_p]
        lib.cg_elements_read.errcheck = self._errcheck

    def _errcheck(self, status, fn, arg):
        if status != 0:
            try:
                raise self._statuses[status]
            except KeyError:
                raise CGNSError

    def open(self, name):
        file = c_int()
        self.lib.cg_open(bytes(name, 'utf-8'), self.CG_MODE_READ, file)
        return file

    def close(self, file):
        self.lib.cg_close(file)

    def base_read(self, file, idx):
        celldim, physdim = c_int(), c_int()
        name = create_string_buffer(32)

        self.lib.cg_base_read(file, idx + 1, name, celldim, physdim)

        return {'file': file, 'idx': idx + 1,
                'name': name.value.decode('utf-8'),
                'CellDim': celldim.value, 'PhysDim': physdim.value}

    def nzones(self, base):
        n = c_int()
        self.lib.cg_nzones(base['file'], base['idx'], n)
        return n.value

    def zone_read(self, base, idx):
        zonetype = c_int()
        name = create_string_buffer(32)
        size = (c_int * 3)()

        self.lib.cg_zone_read(base['file'], base['idx'], idx + 1, name, size)

        # Check zone type
        self.lib.cg_zone_type(base['file'], base['idx'], idx + 1, zonetype)
        if zonetype.value != self.Unstructured:
            raise RuntimeError('ReadCGNS_read: Incorrect zone type for file')

        return {'base': base, 'idx': idx + 1,
                'name': name.value.decode('utf-8'),
                'size': list(size)}

    def coord_read(self, zone, name, x):
        i = c_int(1)
        j = c_int(zone['size'][0])

        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']

        # The data type does not need to be the same as the one in which the
        # coordinates are stored in the file
        # http://cgns.github.io/CGNS_docs_current/midlevel/grid.html
        datatype = self.RealDouble

        self.lib.cg_coord_read(file, base, zone, bytes(name, 'utf-8'),
                               datatype, i, j, x.ctypes.data)

    def nbocos(self, zone):
        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']
        n = c_int()

        self.lib.cg_goto(file, base, b'Zone_t', 1, b'ZoneBC_t', 1, b'end')
        self.lib.cg_nbocos(file, base, zone, n)
        return n.value

    def boco_read(self, zone, idx):
        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']

        name = create_string_buffer(32)
        bocotype = c_int()
        ptset_type = c_int()
        npnts = c_int()
        normalindex = (c_int * 3)()
        normallistsize = c_int()
        normaldatatype = c_int()
        ndataset = c_int()
        bcrange = (c_int * 2)()

        self.lib.cg_boco_info(
            file, base, zone, idx + 1, name, bocotype, ptset_type, npnts,
            normalindex, normallistsize, normaldatatype, ndataset
        )

        if ptset_type.value not in [self.PointRange, self.ElementRange]:
            raise RuntimeError('Only element range BC is supported')

        self.lib.cg_boco_read(file, base, zone, idx + 1, bcrange, None)

        return {'name': name.value.decode('utf-8'),
                'range': tuple(bcrange)}

    def nsections(self, zone):
        file = zone['base']['file']
        base = zone['base']['idx']
        zone = zone['idx']

        n = c_int()
        self.lib.cg_nsections(file, base, zone, n)

        return n.value

    def section_read(self, zone, idx):
        file = zone['base']['file']
        base = zone['base']['idx']
        zidx = zone['idx']

        name = create_string_buffer(32)
        etype, start, end, nbdry = c_int(), c_int(), c_int(), c_int()
        pflag, cdim = c_int(), c_int()

        self.lib.cg_section_read(
            file, base, zidx, idx + 1, name, etype, start, end, nbdry, pflag
        )

        self.lib.cg_ElementDataSize(file, base, zidx, idx + 1, cdim)

        return {'zone': zone, 'idx': idx + 1, 'dim': cdim.value,
                'etype': etype.value, 'range': (start.value, end.value)}

    def elements_read(self, sect, conn):
        file = sect['zone']['base']['file']
        base = sect['zone']['base']['idx']
        zone = sect['zone']['idx']
        idx = sect['idx']

        self.lib.cg_elements_read(file, base, zone, idx,
                                  conn.ctypes.data, None)


class CGNSZoneReader(object):
    # CGNS element types to PyFR type (petype) and node counts
    cgns_map = {
        3: ('line', 2), 4: ('line', 3), 24: ('line', 4), 40: ('line', 5),
        5: ('tri', 3), 6: ('tri', 6), 26: ('tri', 10), 42: ('tri', 15),
        7: ('quad', 4), 9: ('quad', 9), 28: ('quad', 16), 44: ('quad', 25),
        10: ('tet', 4), 11: ('tet', 10), 30: ('tet', 20), 47: ('tet', 35),
        12: ('pyr', 5), 13: ('pyr', 14), 33: ('pyr', 30), 50: ('pyr', 55),
        14: ('pri', 6), 16: ('pri', 18), 36: ('pri', 40), 53: ('pri', 75),
        17: ('hex', 8), 19: ('hex', 27), 39: ('hex', 64), 56: ('hex', 125),
        20: ('mixed',),
    }

    def __init__(self, cgns, base, idx):
        self._cgns = cgns
        zone = cgns.zone_read(base, idx)

        # Read nodes
        self.nodepts = self._read_nodepts(zone)

        # Read bc
        bc = self._read_bc(zone)

        # Read elements
        self.elenodes = elenodes = {}
        self.pents = pents = {}

        # Construct elenodes and physical entity
        for idx in range(cgns.nsections(zone)):
            elerng, elenode = self._read_element(zone, idx)

            for bcname, bcrng in bc.items():
                if elerng[0] >= bcrng[0] and elerng[1] <= bcrng[1]:
                    name = bcname
                    break
            else:
                name = 'fluid'

            pent = pents.setdefault(name, idx)

            elenodes.update({(k, pent): v for k, v in elenode.items()})

    def _read_nodepts(self, zone):
        nnode = zone['size'][0]
        nodepts = np.zeros((3, nnode))

        self._cgns.coord_read(zone, 'CoordinateX', nodepts[0])
        self._cgns.coord_read(zone, 'CoordinateY', nodepts[1])
        self._cgns.coord_read(zone, 'CoordinateZ', nodepts[2])

        return nodepts

    def _read_bc(self, zone):
        nbc = self._cgns.nbocos(zone)
        bc = {}

        for idx_bc in range(nbc):
            boco = self._cgns.boco_read(zone, idx_bc)
            name = boco['name'].lower()
            bc[name] = boco['range']

        return bc

    def _read_element(self, zone, idx):
        s = self._cgns.section_read(zone, idx)

        elerng = s['range']
        conn = np.zeros(s['dim'], dtype=np.int32)
        self._cgns.elements_read(s, conn)

        cgns_type = s['etype']
        petype = self.cgns_map[cgns_type][0]

        elenode = {}

        if petype == 'mixed':
            i = 0
            mele = defaultdict(list)

            while i < s['dim']:
                try:
                    cgns_type = conn[i]
                    petype, spts = self.cgns_map[cgns_type]
                except KeyError:
                    raise

                mele[cgns_type].append(conn[i + 1: i + 1 + spts])
                i += 1 + spts

            elenode = {k: np.array(v) for k, v in mele.items()}
        else:
            spts = self.cgns_map[cgns_type][1]
            elenode[cgns_type] = conn.reshape(-1, spts)

        return elerng, elenode


class CGNSReader(BaseReader):
    # Supported file types and extensions
    name = 'cgns'
    extn = ['.cgns']

    # CGNS element types to PyFR type (petype) and node counts
    _etype_map = CGNSZoneReader.cgns_map

    # First-order node numbers associated with each element face
    _petype_fnmap = {
        'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
        'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
        'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
        'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                         [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
        'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
                'tri': [[0, 1, 2], [3, 4, 5]]},
        'pyr': {'quad': [[0, 1, 2, 3]],
                'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}
    }

    # Mappings between the node ordering of PyFR and that of CGNS
    _nodemaps = CGNSNodeMaps

    def __init__(self, msh):
        # Load and wrap CGNS
        self._cgns = cgns = CGNSWrappers()

        # Read CGNS mesh file
        self._file = file = cgns.open(msh.name)
        base = cgns.base_read(file, 0)

        if cgns.nzones(base) > 1:
            raise ValueError('Only single zone is supported')

        # Read the single CGNS Zone
        zone = CGNSZoneReader(cgns, base, 0)

        self._nodepts = {i + 1: v
                         for i, v in enumerate(zone.nodepts.swapaxes(0, 1))}
        self._elenodes = zone.elenodes
        pents = zone.pents

        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        #  - periodic faces
        self._felespent = pents.pop('fluid')
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        for name, pent in pents.items():
            if name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p.group(1)].append(name)
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

        if any(len(pf) != 2 for pf in self._pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def __del__(self):
        if hasattr(self, '_file'):
            self._cgns.close(self._file)

    def _to_raw_pyfrm(self):
        # Assemble a nodal mesh
        maps = self._etype_map, self._petype_fnmap, self._nodemaps
        pents = self._felespent, self._bfacespents, self._pfacespents
        mesh = NodalMeshAssembler(self._nodepts, self._elenodes, pents, maps)

        rawm = {}
        rawm.update(mesh.get_connectivity())
        rawm.update(mesh.get_shape_points())
        return rawm
