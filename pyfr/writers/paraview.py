# -*- coding: utf-8 -*-

"""Converts .pyfr[m, s] files to a Paraview VTK UnstructuredGrid File"""

from collections import defaultdict
import os

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.solvers import BaseSystem
from pyfr.util import subclass_where
from pyfr.writers import BaseWriter


class ParaviewWriter(BaseWriter):
    """Wrapper for writing serial .vtu  and parallel .pvtu Paraview files"""
    # Supported file types and extensions
    name = 'paraview'
    extn = ['.vtu', '.pvtu']

    # PyFR to VTK element types and number of nodes
    vtk_types = {
        'tri': (5, 3), 'quad': (9, 4),
        'tet': (10, 4), 'pyr': (14, 5), 'pri': (13, 6), 'hex': (12, 8)
    }

    def __init__(self, args):
        super().__init__(args)

        self.dtype = np.dtype(args.precision).type
        self.divisor = args.divisor or self.cfg.getint('solver', 'order')

    def _get_npts_ncells_nnodes(self, mk):
        m_inf = self.mesh_inf[mk]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=m_inf[0])
        subdvcls = subclass_where(BaseShapeSubDiv, name=m_inf[0])

        # Number of vis points
        npts = shapecls.nspts_from_order(self.divisor + 1)*m_inf[1][1]

        # Number of sub cells and nodes
        ncells = len(subdvcls.subcells(self.divisor))*m_inf[1][1]
        nnodes = len(subdvcls.subnodes(self.divisor))*m_inf[1][1]

        return npts, ncells, nnodes

    def _get_vtu_array_attrs(self, mk=None):
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'
        dsize = np.dtype(self.dtype).itemsize
        ndims = self.ndims

        names = ['', 'connectivity', 'offsets', 'types', 'Density',
                'Velocity', 'Pressure']
        types = [dtype, 'Int32', 'Int32', 'UInt8'] + [dtype]*3
        comps = ['3', '', '', '', '1', str(ndims), '1']

        # If a mesh has been given the compute the sizes
        if mk:
            npts, ncells, nnodes = self._get_npts_ncells_nnodes(mk)

            sizes = np.array([3, 4*nnodes, 4*ncells, ncells, 1, ndims, 1])
            sizes[[0, 4, 5, 6]] *= dsize*npts

            return names, types, comps, sizes
        else:
            return names, types, comps

    def write_out(self):
        name, extn = os.path.splitext(self.outf)
        parallel = extn == '.pvtu'

        parts = defaultdict(list)
        for mk, sk in zip(self.mesh_inf, self.soln_inf):
            prt = mk.split('_')[-1]
            pfn = '{0}_{1}.vtu'.format(name, prt) if parallel else self.outf

            parts[pfn].append((mk, sk))

        write_s_to_fh = lambda s: fh.write(s.encode('utf-8'))

        for pfn, misil in parts.items():
            with open(pfn, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="UnstructuredGrid" '
                              'version="0.1">\n<UnstructuredGrid>\n')

                # Running byte-offset for appended data
                off = 0

                # Header
                for mk, sk in misil:
                    off = self._write_serial_vtu_header(fh, mk, off)

                write_s_to_fh('</UnstructuredGrid>\n'
                              '<AppendedData encoding="raw">\n_')

                # Data
                for mk, sk in misil:
                    self._write_vtu_data(fh, mk, sk)

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        if parallel:
            with open(self.outf, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="PUnstructuredGrid" '
                              'version="0.1">\n<PUnstructuredGrid>\n')

                # Header
                self._write_parallel_vtu_header(fh)

                # Constitutent pieces
                for pfn in parts:
                    write_s_to_fh('<Piece Source="{0}"/>\n'
                                  .format(os.path.basename(pfn)))

                write_s_to_fh('</PUnstructuredGrid>\n</VTKFile>\n')


    def _write_vtk_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)

        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _write_serial_vtu_header(self, vtuf, mk, off):
        names, types, comps, sizes = self._get_vtu_array_attrs(mk)
        npts, ncells = self._get_npts_ncells_nnodes(mk)[:2]

        write_s = lambda s: vtuf.write(s.encode('utf-8'))
        write_s('<Piece NumberOfPoints="{0}" NumberOfCells="{1}">\n'
                .format(npts, ncells))
        write_s('<Points>\n')

        # Write vtk DaraArray headers
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s('<DataArray Name="{0}" type="{1}" '
                    'NumberOfComponents="{2}" '
                    'format="appended" offset="{3}"/>\n'
                    .format(n, t, c, off))

            off += 4 + s

            # Write ends/starts of vtk file objects
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            elif i == 3:
                write_s('</Cells>\n<PointData>\n')

        # Write end of vtk element data
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_parallel_vtu_header(self, vtuf):
        names, types, comps = self._get_vtu_array_attrs()

        write_s = lambda s: vtuf.write(s.encode('utf-8'))
        write_s('<PPoints>\n')

        # Write vtk DaraArray headers
        for i, (n, t, s) in enumerate(zip(names, types, comps)):
            write_s('<PDataArray Name="{0}" type="{1}" '
                    'NumberOfComponents="{2}"/>\n'.format(n, t, s))

            if i == 0:
                write_s('</PPoints>\n<PCells>\n')
            elif i == 3:
                write_s('</PCells>\n<PPointData>\n')

        write_s('</PPointData>\n')

    def _write_vtu_data(self, vtuf, mk, sk):
        name = self.mesh_inf[mk][0]
        mesh = self.mesh[mk]
        soln = self.soln[sk]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=name)
        subdvcls = subclass_where(BaseShapeSubDiv, name=name)

        # Get the system class
        syscls = subclass_where(BaseSystem,
                                name=self.cfg.get('solver', 'system'))

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.divisor)
        nsvpts = len(svpts)

        # Shape
        soln_b = shapecls(nspts, self.cfg)

        # Generate the operator matrices
        mesh_vtu_op = soln_b.sbasis.nodal_basis_at(svpts)
        soln_vtu_op = soln_b.ubasis.nodal_basis_at(svpts)

        # Calculate node locations of vtu elements
        vpts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Calculate solution at node locations of vtu elements
        vsol = np.dot(soln_vtu_op, soln.reshape(-1, self.nvars*neles))
        vsol = vsol.reshape(nsvpts, self.nvars, -1).swapaxes(0, 1)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.append(vpts, np.zeros(pts.shape[:-1])[..., None],
                             axis=2)

        # Write element node locations to file
        self._write_vtk_darray(vpts.swapaxes(0, 1), vtuf, self.dtype)

        # Perform the sub division
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare vtu cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

        # Tile vtu cell type numbers
        vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)

        # Write vtu node connectivity, connectivity offsets and cell types
        self._write_vtk_darray(vtu_con, vtuf, np.int32)
        self._write_vtk_darray(vtu_off, vtuf, np.int32)
        self._write_vtk_darray(vtu_typ, vtuf, np.uint8)

        # Convert from conservative to primitive variables
        vsol = np.array(syscls.elementscls.conv_to_pri(vsol, self.cfg))

        # Write Density, Velocity and Pressure
        self._write_vtk_darray(vsol[0].T, vtuf, self.dtype)
        self._write_vtk_darray(vsol[1:-1].T, vtuf, self.dtype)
        self._write_vtk_darray(vsol[-1].T, vtuf, self.dtype)


class BaseShapeSubDiv(object):
    vtk_types = dict(tri=5, quad=9, tet=10, pyr=14, pri=13, hex=12)
    vtk_nodes = dict(tri=3, quad=4, tet=4, pyr=5, pri=6, hex=8)

    @classmethod
    def subcells(cls, n):
        pass

    @classmethod
    def subcelloffs(cls, n):
        return np.cumsum([cls.vtk_nodes[t] for t in cls.subcells(n)])

    @classmethod
    def subcelltypes(cls, n):
        return np.array([cls.vtk_types[t] for t in cls.subcells(n)])

    @classmethod
    def subnodes(cls, n):
        pass


class TensorProdShapeSubDiv(BaseShapeSubDiv):
    @classmethod
    def subnodes(cls, n):
        conbase = np.array([0, 1, n + 2, n + 1])

        # Extend quad mapping to hex mapping
        if cls.ndim == 3:
            conbase = np.hstack((conbase, conbase + (1 + n)**2))

        # Calculate offset of each subdivided element's nodes
        nodeoff = np.zeros((n,)*cls.ndim)
        for dim, off in enumerate(np.ix_(*(range(n),)*cls.ndim)):
            nodeoff += off*(n + 1)**dim

        # Tile standard element node ordering mapping, then apply offsets
        internal_con = np.tile(conbase, (n**cls.ndim, 1))
        internal_con += nodeoff.T.flatten()[:, None]

        return np.hstack(internal_con)


class QuadShapeSubDiv(TensorProdShapeSubDiv):
    name = 'quad'
    ndim = 2

    @classmethod
    def subcells(cls, n):
        return ['quad']*(n**2)


class HexShapeSubDiv(TensorProdShapeSubDiv):
    name = 'hex'
    ndim = 3

    @classmethod
    def subcells(cls, n):
        return ['hex']*(n**3)


class TriShapeSubDiv(BaseShapeSubDiv):
    name = 'tri'

    @classmethod
    def subcells(cls, n):
        return ['tri']*(n**2)

    @classmethod
    def subnodes(cls, n):
        conlst = []

        for row in range(n, 0, -1):
            # Lower and upper indices
            l = (n - row)*(n + row + 3) // 2
            u = l + row + 1

            # Base offsets
            off = [l, l + 1, u, u + 1, l + 1, u]

            # Generate current row
            subin = np.ravel(np.arange(row - 1)[..., None] + off)
            subex = [ix + row - 1 for ix in off[:3]]

            # Extent list
            conlst.extend([subin, subex])

        return np.hstack(conlst)


class TetShapeSubDiv(BaseShapeSubDiv):
    name = 'tet'

    @classmethod
    def subcells(cls, nsubdiv):
        return ['tet']*(nsubdiv**3)

    @classmethod
    def subnodes(cls, nsubdiv):
        conlst = []
        jump = 0

        for n in range(nsubdiv, 0, -1):
            for row in range(n, 0, -1):
                # Lower and upper indices
                l = (n - row)*(n + row + 3) // 2 + jump
                u = l + row + 1

                # Lower and upper for one row up
                ln = (n + 1)*(n + 2) // 2 + l - n + row
                un = ln + row

                rowm1 = np.arange(row - 1)[..., None]

                # Base offsets
                offs = [(l, l + 1, u, ln), (l + 1, u, ln, ln + 1),
                        (u, u + 1, ln + 1, un), (u, ln, ln + 1, un),
                        (l + 1, u, u+1, ln + 1), (u + 1, ln + 1, un, un + 1)]

                # Current row
                conlst.extend(rowm1 + off for off in offs[:-1])
                conlst.append(rowm1[:-1] + offs[-1])
                conlst.append([ix + row - 1 for ix in offs[0]])

            jump += (n + 1)*(n + 2) // 2

        return np.hstack(np.ravel(c) for c in conlst)


class PriShapeSubDiv(BaseShapeSubDiv):
    name = 'pri'

    @classmethod
    def subcells(cls, n):
        return ['pri']*(n**3)

    @classmethod
    def subnodes(cls, n):
        # Triangle connectivity
        tcon = TriShapeSubDiv.subnodes(n).reshape(-1, 3)

        # Layer these rows of triangles to define prisms
        loff = (n + 1)*(n + 2) // 2
        lcon = [[tcon + i*loff, tcon + (i + 1)*loff] for i in range(n)]

        return np.hstack(np.hstack(l).flat for l in lcon)


class PyrShapeSubDiv(BaseShapeSubDiv):
    name = 'pyr'

    @classmethod
    def subcells(cls, n):
        cells = []

        for i in range(n, 0, -1):
            cells += ['pyr']*(i**2 + (i - 1)**2)
            cells += ['tet']*(2*i*(i - 1))

        return cells

    @classmethod
    def subnodes(cls, nsubdiv):
        lcon = []

        # Quad connectivity
        qcon = [QuadShapeSubDiv.subnodes(n + 1).reshape(-1, 4)
                for n in range(nsubdiv)]

        # Simple functions
        def _row_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*i + j + 1)
                             for i in range(a, n + b)
                             for j in range(n - 1)])

        def _col_in_quad(n, a=0, b=0):
            return np.array([(n*i + j, n*(i + 1) + j)
                             for i in range(n - 1)
                             for j in range(a, n + b)])

        u = 0
        for n in range(nsubdiv, 0, -1):
            l = u
            u += (n + 1)**2

            lower_quad = qcon[n - 1] + l
            upper_pts = np.arange(n**2) + u

            # First set of pyramids
            lcon.append([lower_quad, upper_pts])

            if n > 1:
                upper_quad = qcon[n - 2] + u
                lower_pts = np.hstack(range(k*(n + 1)+1, (k + 1)*n + k)
                                      for k in range(1, n)) + l

                # Second set of pyramids
                lcon.append([upper_quad[:, ::-1], lower_pts])

                lower_row = _row_in_quad(n + 1, 1, -1) + l
                lower_col = _col_in_quad(n + 1, 1, -1) + l

                upper_row = _row_in_quad(n) + u
                upper_col = _col_in_quad(n) + u

                # Tetrahedra
                lcon.append([lower_col, upper_row])
                lcon.append([lower_row[:, ::-1], upper_col])

        return np.hstack(np.column_stack(l).flat for l in lcon)
