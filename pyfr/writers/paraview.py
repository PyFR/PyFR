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

    def write_out(self):
        filerootname, extn = os.path.splitext(self.outf)
        parallel = extn == '.pvtu'

        # Set default divisor to solution order
        if self.args.divisor == 0:
            self.args.divisor = self.cfg.getint('solver', 'order')

        parts = defaultdict(list)
        for mk, sk in zip(self.mesh_inf, self.soln_inf):
            part = mk.split('_')[-1]

            if parallel:
                pfn = '{}_{}.vtu'.format(filerootname, part)
            else:
                pfn = self.outf

            parts[pfn].append((mk, sk))

        write_s_to_fh = lambda s: fh.write(s.encode('utf-8'))

        for pfn, misil in parts.items():
            with open(pfn, 'wb') as fh:
                # Write .vtu file header
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="UnstructuredGrid" '
                              'version="0.1">\n<UnstructuredGrid>\n')

                off = 0
                for mk, sk in misil:
                    # Write data description header.
                    # element in a partition.
                    off += _write_vtu_header(self.args, fh, self.mesh_inf[mk],
                                             self.soln_inf[sk], off)

                # Write end/start of header/data sections
                write_s_to_fh('</UnstructuredGrid>\n'
                              '<AppendedData encoding="raw">\n_')
                for mk, sk in misil:
                    # Write data "piece"wise
                    _write_vtu_data(self.args, fh, self.cfg, self.mesh[mk],
                                    self.mesh_inf[mk], self.soln[sk],
                                    self.soln_inf[sk])

                # Write .vtu file footer
                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        if parallel:
            with open(self.outf, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="PUnstructuredGrid" '
                              'version="0.1">\n<PUnstructuredGrid>\n')
                mk, sk = next(iter(parts.values()))[0]
                _write_vtu_header(self.args, fh, self.mesh_inf[mk],
                                  self.soln_inf[sk], 0, True)
                for pfn in parts:
                    write_s_to_fh('<Piece Source="{}"/>\n'
                                  .format(os.path.basename(pfn)))
                write_s_to_fh('</PUnstructuredGrid>\n</VTKFile>\n')


def _write_vtk_darray(array, vtuf, numtyp):
    array = array.astype(numtyp)

    np.uint32(array.nbytes).tofile(vtuf)
    array.tofile(vtuf)


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


def _write_vtu_header(args, vtuf, m_inf, s_inf, off, parallel=False):
    # Set vtk name for float, set size in bytes
    if args.precision == 'single':
        flt = ['Float32', 4]
    else:
        flt = ['Float64', 8]

    write_s = lambda s: vtuf.write(s.encode('utf-8'))

    prefix = 'P' if parallel else ''

    # Get the shape and sub division classes
    shapecls = subclass_where(BaseShape, name=m_inf[0])
    subdvcls = subclass_where(BaseShapeSubDiv, name=m_inf[0])

    npts = shapecls.nspts_from_order(args.divisor + 1)*m_inf[1][1]

    cells = subdvcls.subcells(args.divisor)
    nodes = subdvcls.subnodes(args.divisor)

    ncells = len(cells)*m_inf[1][1]
    nnodes = len(nodes)*m_inf[1][1]

    # Standard template for vtk DataArray string
    if parallel:
        darray = ('<PDataArray Name="%s" type="%s" '
                  'NumberOfComponents="%s"/>\n')
    else:
        darray = ('<DataArray Name="%s" type="%s" '
                  'NumberOfComponents="%s" format="appended" offset="%d"/>\n')

    if not parallel:
        # Write headers for vtu elements
        write_s('<Piece NumberOfPoints="%s" NumberOfCells="%s">\n'
                % (npts, ncells))

    write_s('<{prefix}Points>\n'.format(prefix=prefix))

    # Lists of DataArray "names", "types" and "NumberOfComponents"
    nams = ['', 'connectivity', 'offsets', 'types', 'Density', 'Velocity',
            'Pressure']
    typs = [flt[0], 'Int32', 'Int32', 'UInt8'] + [flt[0]] * 3
    ncom = ['3', '', '', '', '1', str(m_inf[1][2]), '1']

    # Calculate size of described DataArrays (in bytes)
    offs = np.array([0, 3, 4*nnodes, 4*ncells, ncells, 1, m_inf[1][2], 1])
    offs[[1, 5, 6, 7]] *= flt[1]*npts

    # Write vtk DaraArray headers
    for i in range(len(nams)):
        if parallel:
            write_s(darray % (nams[i], typs[i], ncom[i]))
        else:
            write_s(darray % (nams[i], typs[i], ncom[i],
                              sum(offs[:i+1]) + i*4 + off))

        # Write ends/starts of vtk file objects
        if i == 0:
            write_s('</{prefix}Points>\n<{prefix}Cells>\n'
                    .format(prefix=prefix))
        elif i == 3:
            write_s('</{prefix}Cells>\n<{prefix}PointData>\n'
                    .format(prefix=prefix))

    # Write end of vtk element data
    write_s('</{prefix}PointData>\n'.format(prefix=prefix))

    if not parallel:
        write_s('</Piece>\n')

    # Return the number of bytes appended
    return sum(offs) + 4*len(nams)


def _write_vtu_data(args, vtuf, cfg, mesh, m_inf, soln, s_inf):
    dtype = 'float32' if args.precision == 'single' else 'float64'

    # Get the shape and sub division classes
    shapecls = subclass_where(BaseShape, name=m_inf[0])
    subdvcls = subclass_where(BaseShapeSubDiv, name=m_inf[0])

    # Get the system class
    syscls = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))

    nspts, neles, ndims = m_inf[1]
    nvpts = shapecls.nspts_from_order(args.divisor + 1)

    # Generate basis objects for solution and vtu output
    soln_b = shapecls(nspts, cfg)
    vtu_b = shapecls(nvpts, cfg)

    # Generate operator matrices to move points and solutions to vtu nodes
    mesh_vtu_op = soln_b.sbasis.nodal_basis_at(vtu_b.spts)
    soln_vtu_op = soln_b.ubasis.nodal_basis_at(vtu_b.spts)

    # Calculate node locations of vtu elements
    pts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
    pts = pts.reshape(nvpts, -1, ndims)

    # Calculate solution at node locations of vtu elements
    sol = np.dot(soln_vtu_op, soln.reshape(s_inf[1][0], -1))
    sol = sol.reshape(nvpts, s_inf[1][1], -1).swapaxes(0, 1)

    # Append dummy z dimension for points in 2-d (required by Paraview)
    if ndims == 2:
        pts = np.append(pts, np.zeros(pts.shape[:-1])[..., None], axis=2)

    # Write element node locations to file
    _write_vtk_darray(pts.swapaxes(0, 1), vtuf, dtype)

    # Perform the sub division
    nodes = subdvcls.subnodes(args.divisor)

    # Prepare vtu cell arrays (connectivity, offsets, types):
    # Generate and extend vtu sub-cell node connectivity across all elements
    vtu_con = np.tile(nodes, (neles, 1))
    vtu_con += (np.arange(neles)*nvpts)[:, None]

    # Generate offset into the connectivity array for the end of each element
    vtu_off = np.tile(subdvcls.subcelloffs(args.divisor), (neles, 1))
    vtu_off += (np.arange(neles)*len(nodes))[:, None]

    # Tile vtu cell type numbers
    vtu_typ = np.tile(subdvcls.subcelltypes(args.divisor), neles)

    # Write vtu node connectivity, connectivity offsets and cell types
    _write_vtk_darray(vtu_con, vtuf, 'int32')
    _write_vtk_darray(vtu_off, vtuf, 'int32')
    _write_vtk_darray(vtu_typ, vtuf, 'uint8')

    # Convert from conservative to primitive variables
    sol = np.array(syscls.elementscls.conv_to_pri(sol, cfg))

    # Write Density, Velocity and Pressure
    _write_vtk_darray(sol[0].T, vtuf, dtype)
    _write_vtk_darray(sol[1:-1].T, vtuf, dtype)
    _write_vtk_darray(sol[-1].T, vtuf, dtype)
