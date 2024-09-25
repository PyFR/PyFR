from collections import defaultdict
import re

import numpy as np

from pyfr.readers import BaseReader, NodalMeshAssembler


def msh_section(mshit, section):
    endln = f'$End{section}\n'
    endix = int(next(mshit))

    for i, l in enumerate(mshit, start=1):
        if l == endln:
            raise ValueError(f'Unexpected end of section ${section}')

        yield l.strip()

        if i == endix:
            break
    else:
        raise ValueError('Unexpected EOF')

    if next(mshit) != endln:
        raise ValueError(f'Expected $End{section}')


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
    _nodemaps = {
        ('tet', 4): [0, 1, 2, 3],
        ('tet', 10): [0, 4, 1, 6, 5, 2, 7, 9, 8, 3],
        ('tet', 20): [0, 4, 5, 1, 9, 16, 6, 8, 7, 2, 11, 17, 15, 18, 19, 13,
                      10, 14, 12, 3],
        ('tet', 35): [0, 4, 5, 6, 1, 12, 22, 24, 7, 11, 23, 8, 10, 9, 2, 15,
                      25, 26, 21, 28, 34, 32, 30, 33, 18, 14, 27, 20, 29, 31,
                      17, 13, 19, 16, 3],
        ('tet', 56): [0, 4, 5, 6, 7, 1, 15, 28, 33, 30, 8, 14, 31, 32, 9, 13,
                      29, 10, 12, 11, 2, 19, 34, 37, 35, 27, 40, 52, 53, 47,
                      45, 54, 50, 42, 48, 23, 18, 39, 38, 26, 43, 55, 49, 44,
                      51, 22, 17, 36, 25, 41, 46, 21, 16, 24, 20, 3],
        ('tet', 84): [
            0, 4, 5, 6, 7, 8, 1, 18, 34, 42, 41, 36, 9, 17, 37, 43, 40, 10, 16,
            38, 39, 11, 15, 35, 12, 14, 13, 2, 23, 44, 47, 48, 45, 33, 54, 74,
            78, 75, 65, 62, 80, 79, 69, 61, 76, 70, 56, 66, 28, 22, 52, 53, 49,
            32, 57, 81, 83, 68, 63, 82, 73, 60, 71, 27, 21, 51, 50, 31, 58, 77,
            67, 59, 72, 26, 20, 46, 30, 55, 64, 25, 19, 29, 24, 3
        ],
        ('pri', 6): [0, 1, 2, 3, 4, 5],
        ('pri', 18): [0, 6, 1, 7, 9, 2, 8, 15, 10, 16, 17, 11, 3, 12, 4, 13,
                      14, 5],
        ('pri', 40): [0, 6, 7, 1, 8, 24, 12, 9, 13, 2, 10, 26, 27, 14, 30, 38,
                      34, 33, 35, 16, 11, 29, 28, 15, 31, 39, 37, 32, 36, 17,
                      3, 18, 19, 4, 20, 25, 22, 21, 23, 5],
        ('pri', 75): [
            0, 6, 7, 8, 1, 9, 33, 35, 15, 10, 34, 16, 11, 17, 2, 12, 39, 43,
            40, 18, 48, 66, 69, 57, 55, 72, 61, 51, 58, 21, 13, 46, 47, 44,
            19, 52, 68, 71, 64, 56, 74, 65, 54, 62, 22, 14, 42, 45, 41, 20, 49,
            67, 70, 60, 53, 73, 63, 50, 59, 23, 3, 24, 25, 26, 4, 27, 36, 37,
            30, 28, 38, 31, 29, 32, 5
        ],
        ('pri', 126): [
            0, 6, 7, 8, 9, 1, 10, 42, 47, 44, 18, 11, 45, 46, 19, 12, 43, 20,
            13, 21, 2, 14, 54, 58, 59, 55, 22, 70, 102, 114, 106, 86, 81, 122,
            118, 90, 80, 110, 91, 73, 87, 26, 15, 65, 66, 67, 60, 23, 74, 104,
            116, 108, 97, 82, 124, 120, 98, 85, 112, 99, 79, 92, 27, 16, 64,
            69, 68, 61, 24, 75, 105, 117, 109, 96, 83, 125, 121, 101, 84, 113,
            100, 78, 93, 28, 17, 57, 63, 62, 56, 25, 71, 103, 115, 107, 89, 76,
            123, 119, 95, 77, 111, 94, 72, 88, 29, 3, 30, 31, 32, 33, 4, 34,
            48, 51, 49, 38, 35, 53, 52, 39, 36, 50, 40, 37, 41, 5
        ],
        ('pri', 196): [
            0, 6, 7, 8, 9, 10, 1, 11, 51, 59, 58, 53, 21, 12, 54, 60, 57, 22,
            13, 55, 56, 23, 14, 52, 24, 15, 25, 2, 16, 71, 75, 76, 77, 72, 26,
            96, 146, 161, 166, 151, 121, 111, 186, 191, 171, 125, 110, 181,
            176, 126, 109, 156, 127, 99, 122, 31, 17, 86, 87, 91, 88, 78, 27,
            100, 148, 163, 168, 153, 136, 112, 188, 193, 173, 137, 119, 183,
            178, 141, 115, 158, 138, 108, 128, 32, 18, 85, 94, 95, 92, 79, 28,
            101, 149, 164, 169, 154, 135, 116, 189, 194, 174, 144, 120, 184,
            179, 145, 118, 159, 142, 107, 129, 33, 19, 84, 90, 93, 89, 80, 29,
            102, 150, 165, 170, 155, 134, 113, 190, 195, 175, 140, 117, 185,
            180, 143, 114, 160, 139, 106, 130, 34, 20, 74, 83, 82, 81, 73, 30,
            97, 147, 162, 167, 152, 124, 103, 187, 192, 172, 133, 104, 182,
            177, 132, 105, 157, 131, 98, 123, 35, 3, 36, 37, 38, 39, 40, 4, 41,
            61, 64, 65, 62, 46, 42, 69, 70, 66, 47, 43, 68, 67, 48, 44, 63, 49,
            45, 50, 5
        ],
        ('pyr', 5): [0, 1, 3, 2, 4],
        ('pyr', 14): [0, 5, 1, 6, 13, 8, 3, 10, 2, 7, 9, 12, 11, 4],
        ('pyr', 30): [0, 5, 6, 1, 7, 25, 28, 11, 8, 26, 27, 12, 3, 16, 15, 2,
                      9, 21, 13, 22, 29, 23, 19, 24, 17, 10, 14, 20, 18, 4],
        ('pyr', 55): [0, 5, 6, 7, 1, 8, 41, 48, 44, 14, 9, 45, 49, 47, 15, 10,
                      42, 46, 43, 16, 3, 22, 21, 20, 2, 11, 29, 30, 17, 33, 50,
                      51, 35, 32, 53, 52, 36, 26, 39, 38, 23, 12, 31, 18, 34,
                      54, 37, 27, 40, 24, 13, 19, 28, 25, 4],
        ('pyr', 91): [
            0, 5, 6, 7, 8, 1, 9, 61, 72, 71, 64, 17, 10, 65, 73, 76, 70, 18,
            11, 66, 74, 75, 69, 19, 12, 62, 67, 68, 63, 20, 3, 28, 27, 26, 25,
            2, 13, 37, 40, 38, 21, 44, 77, 82, 78, 49, 46, 83, 90, 85, 52, 43,
            80, 87, 79, 50, 33, 56, 58, 55, 29, 14, 42, 41, 22, 47, 84, 86, 54,
            48, 89, 88, 53, 34, 59, 60, 30, 15, 39, 23, 45, 81, 51, 35, 57, 31,
            16, 24, 36, 32, 4
        ],
        ('pyr', 140): [
            0, 5, 6, 7, 8, 9, 1, 10, 85, 100, 99, 98, 88, 20, 11, 89, 101, 108,
            104, 97, 21, 12, 90, 105, 109, 107, 96, 22, 13, 91, 102, 106, 103,
            95, 23, 14, 86, 92, 93, 94, 87, 24, 3, 34, 33, 32, 31, 30, 2, 15,
            45, 48, 49, 46, 25, 56, 110, 115, 116, 111, 65, 59, 117, 135, 138,
            121, 68, 58, 118, 136, 137, 122, 69, 55, 113, 126, 125, 112, 66,
            40, 76, 79, 78, 75, 35, 16, 53, 54, 50, 26, 60, 119, 131, 123, 73,
            64, 132, 139, 133, 74, 63, 129, 134, 127, 70, 41, 80, 84, 83, 36,
            17, 52, 51, 27, 61, 120, 124, 72, 62, 130, 128, 71, 42, 81, 82, 37,
            18, 47, 28, 57, 114, 67, 43, 77, 38, 19, 29, 44, 39, 4
        ],
        ('hex', 8): [0, 1, 3, 2, 4, 5, 7, 6],
        ('hex', 27): [0, 8, 1, 9, 20, 11, 3, 13, 2, 10, 21, 12, 22, 26, 23, 15,
                      24, 14, 4, 16, 5, 17, 25, 18, 7, 19, 6],
        ('hex', 64): [
            0, 8, 9, 1, 10, 32, 35, 14, 11, 33, 34, 15, 3, 19, 18, 2, 12, 36,
            37, 16, 40, 56, 57, 44, 43, 59, 58, 45, 22, 49, 48, 20, 13, 39, 38,
            17, 41, 60, 61, 47, 42, 63, 62, 46, 23, 50, 51, 21, 4, 24, 25, 5,
            26, 52, 53, 28, 27, 55, 54, 29, 7, 31, 30, 6
        ],
        ('hex', 125): [
            0, 8, 9, 10, 1, 11, 44, 51, 47, 17, 12, 48, 52, 50, 18, 13, 45, 49,
            46, 19, 3, 25, 24, 23, 2, 14, 53, 57, 54, 20, 62, 98, 106, 99, 71,
            69, 107, 118, 109, 75, 65, 101, 111, 100, 72, 29, 81, 84, 80, 26,
            15, 60, 61, 58, 21, 66, 108, 119, 110, 78, 70, 120, 124, 121, 79,
            68, 113, 122, 112, 76, 30, 85, 88, 87, 27, 16, 56, 59, 55, 22, 63,
            102, 114, 103, 74, 67, 115, 123, 116, 77, 64, 105, 117, 104, 73,
            31, 82, 86, 83, 28, 4, 32, 33, 34, 5, 35, 89, 93, 90, 38, 36, 96,
            97, 94, 39, 37, 92, 95, 91, 40, 7, 43, 42, 41, 6
        ],
        ('hex', 216): [
            0, 8, 9, 10, 11, 1, 12, 56, 67, 66, 59, 20, 13, 60, 68, 71, 65, 21,
            14, 61, 69, 70, 64, 22, 15, 57, 62, 63, 58, 23, 3, 31, 30, 29, 28,
            2, 16, 72, 76, 77, 73, 24, 88, 152, 160, 161, 153, 104, 99, 162,
            184, 187, 166, 108, 98, 163, 185, 186, 167, 109, 91, 155, 171, 170,
            154, 105, 36, 121, 125, 124, 120, 32, 17, 83, 84, 85, 78, 25, 92,
            164, 188, 189, 168, 115, 100, 192, 208, 209, 196, 116, 103, 195,
            211, 210, 197, 117, 97, 174, 201, 200, 172, 110, 37, 126, 133, 132,
            131, 33, 18, 82, 87, 86, 79, 26, 93, 165, 191, 190, 169, 114, 101,
            193, 212, 213, 199, 119, 102, 194, 215, 214, 198, 118, 96, 175,
            202, 203, 173, 111, 38, 127, 134, 135, 130, 34, 19, 75, 81, 80, 74,
            27, 89, 156, 176, 177, 157, 107, 94, 178, 204, 205, 180, 113, 95,
            179, 207, 206, 181, 112, 90, 159, 183, 182, 158, 106, 39, 122, 128,
            129, 123, 35, 4, 40, 41, 42, 43, 5, 44, 136, 140, 141, 137, 48, 45,
            147, 148, 149, 142, 49, 46, 146, 151, 150, 143, 50, 47, 139, 145,
            144, 138, 51, 7, 55, 54, 53, 52, 6
        ],
        ('tri', 3): [0, 1, 2],
        ('tri', 6): [0, 3, 1, 5, 4, 2],
        ('tri', 10): [0, 3, 4, 1, 8, 9, 5, 7, 6, 2],
        ('tri', 15): [0, 3, 4, 5, 1, 11, 12, 13, 6, 10, 14, 7, 9, 8, 2],
        ('tri', 21): [0, 3, 4, 5, 6, 1, 14, 15, 16, 17, 7, 13, 20, 18, 8, 12,
                      19, 9, 11, 10, 2],
        ('quad', 4): [0, 1, 3, 2],
        ('quad', 9): [0, 4, 1, 7, 8, 5, 3, 6, 2],
        ('quad', 16): [0, 4, 5, 1, 11, 12, 13, 6, 10, 15, 14, 7, 3, 9, 8, 2],
        ('quad', 25): [0, 4, 5, 6, 1, 15, 16, 20, 17, 7, 14, 23, 24, 21, 8, 13,
                       19, 22, 18, 9, 3, 12, 11, 10, 2],
        ('quad', 36): [0, 4, 5, 6, 7, 1, 19, 20, 24, 25, 21, 8, 18, 31, 32, 33,
                       26, 9, 17, 30, 35, 34, 27, 10, 16, 23, 29, 28, 22, 11,
                       3, 15, 14, 13, 12, 2]
    }

    def __init__(self, msh, progress):
        super().__init__(progress)

        if isinstance(msh, str):
            msh = open(msh)

        with progress.start_with_spinner('Reading .msh') as pspinner:
            # Get an iterator over the lines of the mesh
            mshit = iter(msh)

            # Have our spinner flashed every 10,000 lines
            mshit = pspinner.wrap_file_lines(mshit, 10000)

            # Section readers
            sect_map = {
                'MeshFormat': self._read_mesh_format,
                'PhysicalNames': self._read_phys_names,
                'Entities': self._read_entities,
                'Nodes': self._read_nodes,
                'Elements': self._read_eles
            }

            for l in filter(lambda l: l != '\n', mshit):
                # Ensure we have encountered a section
                if not l.startswith('$'):
                    raise ValueError('Expected a mesh section')

                # Strip the '$' and '\n' to get the section name
                sect = l[1:-1]

                # Try to read the section
                try:
                    sect_map[sect](mshit)
                # Else skip over it
                except KeyError:
                    endsect = f'$End{sect}\n'

                    for el in mshit:
                        if el == endsect:
                            break
                    else:
                        raise ValueError(f'Expected $End{sect}')

        # Account for any starting node offsets
        for k, v in self._elenodes.items():
            v -= self._nodeoff

    def _read_mesh_format(self, mshit):
        ver, ftype, dsize = next(mshit).split()

        if ver == '2.2':
            self._read_nodes_impl = self._read_nodes_impl_v2
            self._read_eles_impl = self._read_eles_impl_v2
        elif ver == '4.1':
            self._read_nodes_impl = self._read_nodes_impl_v41
            self._read_eles_impl = self._read_eles_impl_v41
        else:
            raise ValueError('Invalid mesh version')

        if ftype != '0':
            raise ValueError('Invalid file type')
        if dsize != '8':
            raise ValueError('Invalid data size')

        if next(mshit) != '$EndMeshFormat\n':
            raise ValueError('Expected $EndMeshFormat')

    def _read_phys_names(self, mshit):
        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        #  - periodic faces
        self._felespent = None
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Seen physical names and IDs
        seen_names = set()
        seen_ids = set()

        # Extract the physical names
        for l in msh_section(mshit, 'PhysicalNames'):
            m = re.match(r'(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError('Malformed physical entity')

            pent, name = int(m[2]), m[3].lower()

            # Ensure we have not seen this name before
            if name in seen_names:
                raise ValueError(f'Duplicate physical name: {name}')

            # Ensure physical entitiy IDs are unique
            if pent in seen_ids:
                raise ValueError(f'Duplicate physical entity ID: {pent}')

            # Fluid elements
            if name == 'fluid':
                self._felespent = pent
            # Periodic boundary faces
            elif name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p[1]].append(pent)
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

            seen_names.add(name)
            seen_ids.add(pent)

        if self._felespent is None:
            raise ValueError('No fluid elements in mesh')

        if any(len(pf) != 2 for pf in self._pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def _read_entities(self, mshit):
        self._tagpents = tagpents = {}

        # Obtain the entity counts
        npts, *ents = (int(i) for i in next(mshit).split())

        # Skip over the point entities
        for i in range(npts):
            next(mshit)

        # Iterate through the curves, surfaces, and volume entities
        for ndim, nent in enumerate(ents, start=1):
            for j in range(nent):
                ent = next(mshit).split()
                etag, enphys = int(ent[0]), int(ent[7])

                if enphys == 0:
                    continue
                elif enphys == 1:
                    tagpents[ndim, etag] = abs(int(ent[8]))
                else:
                    raise ValueError('Invalid physical tag count for entity')

        if next(mshit) != '$EndEntities\n':
            raise ValueError('Expected $EndEntities')

    def _read_nodes(self, mshit):
        self._read_nodes_impl(mshit)

    def _read_nodes_impl_v2(self, mshit):
        nodemap = {}

        # Read in the nodes as a dict
        for l in msh_section(mshit, 'Nodes'):
            nv = l.split()
            nodemap[int(nv[0])] = nv[1:]

        # Determine the minimum and maximum node numbers
        ixl, ixu = min(nodemap), max(nodemap)

        # Pack them into a dense array
        self._nodepts = nodepts = np.empty((ixu - ixl + 1, 3))
        nodepts.fill(np.nan)

        for k, nv in nodemap.items():
            nodepts[k - ixl] = nv

        # Save the starting node offset
        self._nodeoff = ixl

    def _read_nodes_impl_v41(self, mshit):
        # Entity count, node count, minimum and maximum node numbers
        ne, nn, ixl, ixu = (int(i) for i in next(mshit).split())

        self._nodepts = nodepts = np.empty((ixu - ixl + 1, 3))
        nodepts.fill(np.nan)

        for i in range(ne):
            nen = int(next(mshit).split()[-1])
            nix = [int(next(mshit)) for _ in range(nen)]

            for j in nix:
                nodepts[j - ixl] = next(mshit).split()

        # Save the starting node offset
        self._nodeoff = ixl

        if next(mshit) != '$EndNodes\n':
            raise ValueError('Expected $EndNodes')

    def _read_eles(self, mshit):
        self._read_eles_impl(mshit)

    def _read_eles_impl_v2(self, mshit):
        elenodes = defaultdict(list)

        for l in msh_section(mshit, 'Elements'):
            # Extract the raw element data
            elei = [int(i) for i in l.split()]
            enum, etype, entags = elei[:3]
            etags, enodes = elei[3:3 + entags], elei[3 + entags:]

            if etype not in self._etype_map:
                raise ValueError(f'Unsupported element type {etype}')

            # Physical entity type (used for BCs)
            epent = etags[0]

            elenodes[etype, epent].append(enodes)

        self._elenodes = {k: np.array(v) for k, v in elenodes.items()}

    def _read_eles_impl_v41(self, mshit):
        elenodes = defaultdict(list)

        # Block and total element count
        nb, ne = (int(i) for i in next(mshit).split()[:2])

        for i in range(nb):
            edim, etag, etype, ecount = (int(j) for j in next(mshit).split())

            if etype not in self._etype_map:
                raise ValueError(f'Unsupported element type {etype}')

            # Determine the number of nodes associated with each element
            nnodes = self._etype_map[etype][1]

            # Lookup the physical entity type
            epent = self._tagpents[edim, etag]

            # Allocate space for, and read in, these elements
            enodes = np.empty((ecount, nnodes), dtype=np.int64)
            for j in range(ecount):
                enodes[j] = next(mshit).split()[1:]

            elenodes[etype, epent].append(enodes)

        if ne != sum(len(vv) for v in elenodes.values() for vv in v):
            raise ValueError('Invalid element count')

        if next(mshit) != '$EndElements\n':
            raise ValueError('Expected $EndElements')

        self._elenodes = {k: np.vstack(v) for k, v in elenodes.items()}

    def _to_raw_mesh(self, lintol):
        # Assemble a nodal mesh
        maps = self._etype_map, self._petype_fnmap, self._nodemaps
        pents = self._felespent, self._bfacespents, self._pfacespents
        mesh = NodalMeshAssembler(self._nodepts, self._elenodes, pents, maps)

        return mesh.get_eles(lintol, self.progress)
