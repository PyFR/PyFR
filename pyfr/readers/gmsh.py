# -*- coding: utf-8 -*-

from collections import defaultdict
import re

import numpy as np

from pyfr.readers import BaseReader, NodalMeshAssembler
from pyfr.readers.nodemaps import GmshNodeMaps


def msh_section(mshit, section):
    endln = '$End{}\n'.format(section)
    endix = int(next(mshit)) - 1

    for i, l in enumerate(mshit):
        if l == endln:
            raise ValueError('Unexpected end of section $' + section)

        yield l.strip()

        if i == endix:
            break
    else:
        raise ValueError('Unexpected EOF')

    if next(mshit) != endln:
        raise ValueError('Expected $End' + section)


class GmshReader(BaseReader):
    # Supported file types and extensions
    name = 'gmsh'
    extn = ['.msh']

    # Gmsh element types to PyFR type (petype) and node counts
    _etype_map = {
        1: ('line', 2), 8: ('line', 3), 26: ('line', 4), 27: ('line', 5),
        2: ('tri', 3), 9: ('tri', 6), 21: ('tri', 10), 23: ('tri', 15),
        3: ('quad', 4), 10: ('quad', 9), 36: ('quad', 16), 37: ('quad', 25),
        4: ('tet', 4), 11: ('tet', 10), 29: ('tet', 20), 30: ('tet', 35),
        5: ('hex', 8), 12: ('hex', 27), 92: ('hex', 64), 93: ('hex', 125),
        6: ('pri', 6), 13: ('pri', 18), 90: ('pri', 40), 91: ('pri', 75),
        7: ('pyr', 5), 14: ('pyr', 14), 118: ('pyr', 30), 119: ('pyr', 55)
    }

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

    # Mappings between the node ordering of PyFR and that of Gmsh
    _nodemaps = GmshNodeMaps

    def __init__(self, msh):
        if isinstance(msh, str):
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

        for l in filter(lambda l: l != '\n', mshit):
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
                endsect = '$End{}\n'.format(sect)

                for el in mshit:
                    if el == endsect:
                        break
                else:
                    raise ValueError('Expected $End' + sect)

        # Check that all of the required sections are present
        if seen_sect != req_sect:
            missing = req_sect - seen_sect
            raise ValueError('Required sections: {} not found'
                             .format(missing))

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
        self._felespent = None
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Seen physical names
        seen = set()

        # Extract the physical names
        for l in msh_section(msh, 'PhysicalNames'):
            m = re.match(r'(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError('Malformed physical entity')

            pent, name = int(m.group(2)), m.group(3).lower()

            # Ensure we have not seen this name before
            if name in seen:
                raise ValueError('Duplicate physical name: {}'.format(name))

            # Fluid elements
            if name == 'fluid':
                self._felespent = pent
            # Periodic boundary faces
            elif name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p.group(1)].append(pent)
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

            seen.add(name)

        if self._felespent is None:
            raise ValueError('No fluid elements in mesh')

        if any(len(pf) != 2 for pf in self._pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def _read_nodes(self, msh):
        self._nodepts = nodepts = {}

        for l in msh_section(msh, 'Nodes'):
            nv = l.split()
            nodepts[int(nv[0])] = np.array([float(x) for x in nv[1:]])

    def _read_eles(self, msh):
        elenodes = defaultdict(list)

        for l in msh_section(msh, 'Elements'):
            # Extract the raw element data
            elei = [int(i) for i in l.split()]
            enum, etype, entags = elei[:3]
            etags, enodes = elei[3:3 + entags], elei[3 + entags:]

            if etype not in self._etype_map:
                raise ValueError('Unsupported element type {}'.format(etype))

            # Physical entity type (used for BCs)
            epent = etags[0]

            elenodes[etype, epent].append(enodes)

        self._elenodes = {k: np.array(v) for k, v in elenodes.items()}

    def _to_raw_pyfrm(self):
        # Assemble a nodal mesh
        maps = self._etype_map, self._petype_fnmap, self._nodemaps
        pents = self._felespent, self._bfacespents, self._pfacespents
        mesh = NodalMeshAssembler(self._nodepts, self._elenodes, pents, maps)

        rawm = {}
        rawm.update(mesh.get_connectivity())
        rawm.update(mesh.get_shape_points())
        return rawm
