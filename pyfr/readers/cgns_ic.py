"""CGNS IC reader — reads cell-centered RANS data for PyFR IC injection.

Reads a CGNS-HDF5 file containing cell-centered primitive variables
(Density, VelocityX/Y/Z, Pressure) and provides nearest-neighbor
interpolation to arbitrary query points via scipy cKDTree.

Supports both simple CGNS (cell-centred coords) and Fluent CGNS
(vertex coords + MIXED/HEXA_8 connectivity + cell-centred fields).

Usage from PyFR INI::

    [soln-ics]
    type = cgns
    cgns-file = /path/to/rans_solution.cgns
"""

import numpy as np
import h5py
from scipy.spatial import cKDTree


class CGNSICReader:
    """Read cell-centered RANS data from CGNS and interpolate to query pts."""

    # Standard CGNS DataArray names → primitive variable keys
    _field_aliases = {
        # Density
        'Density': 'rho',
        'density': 'rho',
        # Velocity components
        'VelocityX': 'u',
        'VelocityY': 'v',
        'VelocityZ': 'w',
        'velocityx': 'u',
        'velocityy': 'v',
        'velocityz': 'w',
        'Velocity_X': 'u',
        'Velocity_Y': 'v',
        'Velocity_Z': 'w',
        # Pressure
        'Pressure': 'p',
        'pressure': 'p',
        # Momentum (will be converted)
        'MomentumX': 'rhou',
        'MomentumY': 'rhov',
        'MomentumZ': 'rhow',
        # Energy
        'EnergyStagnationDensity': 'E',
        'Energy': 'E',
    }

    # CGNS element type codes
    _HEXA_8 = 17
    _MIXED = 20

    def __init__(self, cgns_path):
        self.path = cgns_path
        self._centres = None
        self._fields = None
        self._tree = None

        self._read_cgns()
        self._build_tree()

    def _read_cgns(self):
        """Read cell centres and flow fields from CGNS-HDF5."""
        with h5py.File(self.path, 'r') as f:
            base = self._find_cgns_base(f)
            zone = self._find_zone(base)

            # Read grid coordinates (may be vertex or cell-centred)
            coords = self._read_coords(zone)

            # Find volume element section and count
            n_vol_cells, vol_elem = self._find_volume_elements(zone)

            # Read flow solution
            raw_fields = self._read_solution(zone)

            # Determine if we need cell-centre computation
            n_coords = len(coords)
            n_fields = len(next(iter(raw_fields.values()))) if raw_fields else 0

            if vol_elem is not None and n_coords != n_vol_cells:
                # Vertex coords + connectivity → compute cell centres
                self._centres = self._compute_cell_centres(
                    vol_elem, coords
                )
                # Truncate fields to volume cells only
                self._fields = {k: v[:n_vol_cells]
                                for k, v in raw_fields.items()}
            elif self._is_vertex_data(zone) and vol_elem is not None:
                self._centres = self._compute_cell_centres(
                    vol_elem, coords
                )
                self._fields = raw_fields
            else:
                self._centres = coords
                self._fields = raw_fields

        n = len(self._centres)
        nf = {k: len(v) for k, v in self._fields.items()}
        print(f'  CGNS IC: {n} cells, fields: {nf}')

    def _find_cgns_base(self, f):
        """Find the CGNSBase_t node."""
        for name in f:
            node = f[name]
            if isinstance(node, h5py.Group):
                label = self._get_label(node)
                if label == 'CGNSBase_t':
                    return node
        raise RuntimeError('No CGNSBase_t found in CGNS file')

    def _find_zone(self, base):
        """Find the first Zone_t."""
        for name in base:
            node = base[name]
            if isinstance(node, h5py.Group):
                label = self._get_label(node)
                if label == 'Zone_t':
                    return node
        raise RuntimeError('No Zone_t found in CGNS base')

    @staticmethod
    def _get_label(node):
        """Get the CGNS label string from an HDF5 node."""
        label = node.attrs.get('label', b'')
        if isinstance(label, bytes):
            label = label.decode()
        return label.strip()

    def _read_coords(self, zone):
        """Read GridCoordinates → (N, ndims) array."""
        gc = None
        for name in zone:
            node = zone[name]
            if isinstance(node, h5py.Group):
                if self._get_label(node) == 'GridCoordinates_t':
                    gc = node
                    break

        if gc is None:
            raise RuntimeError('No GridCoordinates_t found')

        x = y = z = None
        for name in gc:
            node = gc[name]
            if isinstance(node, h5py.Group):
                data = self._read_data_array(node)
                if data is None:
                    continue
                if 'CoordinateX' in name:
                    x = data
                elif 'CoordinateY' in name:
                    y = data
                elif 'CoordinateZ' in name:
                    z = data

        if x is None or y is None:
            raise RuntimeError('Missing coordinate arrays in CGNS')

        if z is not None:
            return np.column_stack([x, y, z])
        else:
            return np.column_stack([x, y])

    def _read_data_array(self, node):
        """Read a DataArray_t node's data."""
        for key in [' data', 'data']:
            if key in node:
                return np.asarray(node[key], dtype=np.float64).ravel()
        if isinstance(node, h5py.Dataset):
            return np.asarray(node, dtype=np.float64).ravel()
        return None

    def _read_int_array(self, node):
        """Read integer data from a DataArray_t node."""
        for key in [' data', 'data']:
            if key in node:
                return np.asarray(node[key], dtype=np.int64).ravel()
        if isinstance(node, h5py.Dataset):
            return np.asarray(node, dtype=np.int64).ravel()
        return None

    def _find_volume_elements(self, zone):
        """Find the volume Elements_t section (largest cell count).

        Returns (n_cells, element_group) or (0, None).
        """
        best_ncells = 0
        best_elem = None

        for name in zone:
            node = zone[name]
            if not isinstance(node, h5py.Group):
                continue
            if self._get_label(node) != 'Elements_t':
                continue

            # Read element type
            etype_data = None
            if ' data' in node:
                etype_data = np.asarray(node[' data'], dtype=np.int32).ravel()

            if etype_data is None:
                continue

            etype = int(etype_data[0])

            # Get element count from ElementRange or connectivity
            er_node = node.get('ElementRange')
            if er_node is not None:
                er = self._read_int_array(er_node)
                if er is not None:
                    ncells = int(er[1] - er[0] + 1)
                else:
                    ncells = 0
            else:
                # Infer from connectivity
                conn_node = node.get('ElementConnectivity')
                if conn_node is None:
                    continue
                if etype == self._HEXA_8:
                    if isinstance(conn_node, h5py.Group):
                        cd = self._read_int_array(conn_node)
                        ncells = len(cd) // 8 if cd is not None else 0
                    elif isinstance(conn_node, h5py.Dataset):
                        ncells = conn_node.shape[0] // 8
                    else:
                        ncells = 0
                elif etype == self._MIXED:
                    eso = node.get('ElementStartOffset')
                    if eso is not None:
                        if isinstance(eso, h5py.Group):
                            od = self._read_int_array(eso)
                            ncells = len(od) - 1 if od is not None else 0
                        else:
                            ncells = eso.shape[0] - 1
                    else:
                        ncells = 0
                else:
                    ncells = 0

            # Volume elements: HEXA_8 or MIXED with hex
            if etype in (self._HEXA_8, self._MIXED) and ncells > best_ncells:
                best_ncells = ncells
                best_elem = node

        return best_ncells, best_elem

    def _is_vertex_data(self, zone):
        """Check if FlowSolution is at vertices."""
        for name in zone:
            node = zone[name]
            if isinstance(node, h5py.Group):
                if self._get_label(node) == 'FlowSolution_t':
                    loc = self._get_grid_location(node)
                    return 'Vertex' in loc
        return False

    @staticmethod
    def _get_grid_location(node):
        """Extract GridLocation string from a FlowSolution_t node."""
        loc = node.attrs.get('GridLocation', b'')
        if isinstance(loc, bytes):
            loc = loc.decode()
        if 'GridLocation' in node:
            gl = node['GridLocation']
            if isinstance(gl, h5py.Group) and ' data' in gl:
                loc = gl[' data'][()].tobytes().decode().strip()
            elif isinstance(gl, h5py.Dataset):
                loc = gl[()].tobytes().decode().strip()
        return loc

    def _read_solution(self, zone):
        """Read FlowSolution fields → dict of 1D arrays."""
        fields = {}
        for name in zone:
            node = zone[name]
            if isinstance(node, h5py.Group):
                if self._get_label(node) == 'FlowSolution_t':
                    fields = self._read_solution_node(node)
                    if fields:
                        break
        return fields

    def _read_solution_node(self, sol_node):
        """Read all DataArray_t children of a FlowSolution_t."""
        fields = {}
        for name in sol_node:
            child = sol_node[name]
            if isinstance(child, h5py.Group):
                if self._get_label(child) == 'DataArray_t':
                    data = self._read_data_array(child)
                    if data is not None:
                        cname = name.strip()
                        key = self._field_aliases.get(cname, cname)
                        fields[key] = data
        return fields

    def _compute_cell_centres(self, elem_node, vertex_coords):
        """Compute cell centres from vertex coords + connectivity.

        Handles HEXA_8 (type 17) and MIXED (type 20) element sections.
        MIXED format: connectivity = [type0, n0..n7, type1, n1..n7, ...]
        with ElementStartOffset giving per-element offsets.
        """
        etype_data = np.asarray(elem_node[' data'], dtype=np.int32).ravel()
        etype = int(etype_data[0])

        # Read connectivity
        conn_node = elem_node.get('ElementConnectivity')
        if conn_node is None:
            raise RuntimeError('No ElementConnectivity found')

        if isinstance(conn_node, h5py.Group):
            conn_data = self._read_int_array(conn_node)
        elif isinstance(conn_node, h5py.Dataset):
            conn_data = np.asarray(conn_node, dtype=np.int64).ravel()
        else:
            raise RuntimeError('Cannot read ElementConnectivity')

        if etype == self._HEXA_8:
            # Pure HEXA_8: connectivity is (N*8,) with 1-based indices
            conn = conn_data.reshape(-1, 8) - 1
            return vertex_coords[conn].mean(axis=1)

        elif etype == self._MIXED:
            # MIXED format with ElementStartOffset
            eso_node = elem_node.get('ElementStartOffset')
            if eso_node is None:
                raise RuntimeError('MIXED elements require ElementStartOffset')

            if isinstance(eso_node, h5py.Group):
                offsets = self._read_int_array(eso_node)
            else:
                offsets = np.asarray(eso_node, dtype=np.int64).ravel()

            n_cells = len(offsets) - 1
            centres = np.empty((n_cells, vertex_coords.shape[1]))

            # Check if all elements have the same stride (common case)
            stride = int(offsets[1] - offsets[0])
            uniform = np.all(np.diff(offsets) == stride)

            if uniform and stride == 9:
                # All HEXA_8 in MIXED format: [type, n0..n7] per element
                # Extract node indices (skip type byte at each element start)
                all_data = conn_data.reshape(n_cells, 9)
                node_ids = all_data[:, 1:] - 1  # skip type, 0-based
                centres = vertex_coords[node_ids].mean(axis=1)
            else:
                # General MIXED: iterate per element
                for i in range(n_cells):
                    s = int(offsets[i])
                    e = int(offsets[i + 1])
                    elem = conn_data[s:e]
                    # First value is element type, rest are node indices
                    node_ids = elem[1:].astype(int) - 1
                    centres[i] = vertex_coords[node_ids].mean(axis=0)

            return centres

        raise RuntimeError(f'Unsupported element type {etype} for '
                           'cell centre computation')

    def _build_tree(self):
        """Build cKDTree for nearest-neighbor lookup."""
        self._tree = cKDTree(self._centres)

    def get_primitives(self, query_pts, gamma=None):
        """Interpolate primitive variables to query points.

        Parameters
        ----------
        query_pts : ndarray, shape (N, ndims)
            Physical coordinates to interpolate to.
        gamma : float, optional
            Ratio of specific heats (needed if data is conservative).

        Returns
        -------
        dict : {varname: ndarray} with keys 'rho', 'u', 'v', 'w', 'p'
        """
        _, idx = self._tree.query(query_pts)

        result = {}
        for key, data in self._fields.items():
            result[key] = data[idx]

        # Convert conservative → primitive if needed
        if 'rhou' in result and 'u' not in result:
            rho = result['rho']
            result['u'] = result.pop('rhou') / rho
            result['v'] = result.pop('rhov') / rho
            result['w'] = result.pop('rhow') / rho

        if 'E' in result and 'p' not in result:
            if gamma is None:
                raise ValueError('gamma required for conservative→primitive')
            rho = result['rho']
            E = result.pop('E')
            ke = 0.5 * rho * (result['u']**2 + result['v']**2
                              + result['w']**2)
            result['p'] = (gamma - 1) * (E - ke)

        # Validate
        if np.any(result.get('rho', 1) <= 0):
            raise ValueError('CGNS IC: non-positive density detected')
        if np.any(result.get('p', 1) <= 0):
            raise ValueError('CGNS IC: non-positive pressure detected')

        return result

    @property
    def n_cells(self):
        return len(self._centres)

    @property
    def field_names(self):
        return list(self._fields.keys())

    @property
    def centres(self):
        return self._centres


def read_cgns_ic(path):
    """Convenience function to create a CGNSICReader."""
    return CGNSICReader(path)
