# -*- coding: utf-8 -*-

"""Converts .pyfr[m, s] files to a Paraview VTK UnstructuredGrid File"""
import numpy as np

from pyfr.inifile import Inifile
from pyfr.readers.nodemaps import GmshNodeMaps
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers import BaseWriter


class ParaviewWriter(BaseWriter):
    """Wrapper for writing serial .vtu Paraview files"""
    # Supported file types and extensions
    name = 'paraview'
    extn = ['.vtu']

    # PyFR to VTK element types and number of nodes
    vtk_types = {
        'tri': (5, 3), 'quad': (9, 4),
        'tet': (10, 4), 'pyr': (14, 5), 'pri': (13, 6), 'hex': (12, 8)
    }

    def write_out(self):
        """Controls the writing of serial .vtu Paraview files

        Writes .vtu pieces for each element type, in each partition of
        the PyFR files.  The Paraview data type used is "appended",
        which concatenates all data into a single block of binary data
        at the end of the file.  ASCII headers written at the top of
        the file describe the structure of this data.
        """
        # Set default divisor to solution order
        if self.args.divisor == 0:
            self.args.divisor = self.cfg.getint('solver', 'order')

        # Write .vtu file header
        self.outf.write('<?xml version="1.0" ?>\n<VTKFile '
                        'byte_order="LittleEndian" type="UnstructuredGrid" '
                        'version="0.1">\n<UnstructuredGrid>\n')

        # Initialise offset (in bytes) to end of appended data
        off = 0

        # Write data description header.  A vtk "piece" is used for each
        # element in a partition.
        for mk, sk in zip(self.mesh_inf, self.soln_inf):
            off += _write_vtu_header(self.args, self.outf, self.mesh_inf[mk],
                                     self.soln_inf[sk], off)

        # Write end/start of header/data sections
        self.outf.write('</UnstructuredGrid>\n<AppendedData '
                        'encoding="raw">\n_')

        # Write data "piece"wise
        for mk, sk in zip(self.mesh_inf, self.soln_inf):
            _write_vtu_data(self.args, self.outf, self.cfg, self.mesh[mk],
                            self.mesh_inf[mk], self.soln[sk],
                            self.soln_inf[sk])

        # Write .vtu file footer
        self.outf.write('\n</AppendedData>\n</VTKFile>')


def _write_vtk_darray(array, vtuf, numtyp):
    array = array.astype(numtyp)

    np.uint32(array.nbytes).tofile(vtuf)
    array.tofile(vtuf)


def _component_to_physical_soln(soln, gamma):
    """Convert PyFR solution of rho, rho(u, v, [w]), E to rho, u, v, [w], p

    :param soln: PyFR solution array to be converted.
    :param gamma: Ratio of specific heats.
    :type soln: numpy.ndarray of shape [nupts, neles, ndims + 2]
    :type gamma: integer

    """
    # Convert rhou, rhov, [rhow] to u, v, [w]
    soln[:,1:-1] /= soln[:,0,None]

    # Convert total energy to pressure
    soln[:,-1] -= 0.5*soln[:,0]*np.sum(soln[:,1:-1]**2, axis=1)
    soln[:,-1] *= gamma - 1


def _ncells_after_subdiv(ms_inf, divisor):
    """Calculates total number of vtu cells in partition after subdivision

    :param ms_inf: Mesh/solninformation. ('ele_type', [npts, nele, ndims])
    :type ms_inf: tuple: (str, list)
    :rtype: integer

    """
    # Calculate the number of low-order cells in each high order element
    n_sub_ele = divisor ** ms_inf[1][2]

    # Pyramids require the further addition of an arithmetic series
    if ms_inf[0] == 'pyr':
        n_sub_ele += (divisor - 1) * divisor / 2

    # Multiply by number of elements
    return n_sub_ele * ms_inf[1][1]


def _quadcube_con(ndim, nsubdiv):
    """Generate node connectivity for vtu hex/quad in high-order elements

    :param ndim: Number of dimensions [2,3]
    :param nsubdiv: Number of subdivisions (equal to element shape order)
    :type ndim: integer
    :type nsubdiv: integer
    :rtype: list

    """
    # Mapping from pyfr to vtk quad nodes
    conbase = np.array([0, 1, nsubdiv + 2, nsubdiv + 1], dtype=int)

    # Extend quad mapping to hex mapping if 3-d
    if ndim == 3:
        conbase = np.hstack((conbase, conbase + (1 + nsubdiv) ** 2))

    # Calculate offset of each subdivided element's nodes from std. mapping
    nodeoff = np.zeros((nsubdiv,)*ndim)
    for dim, off in enumerate(np.ix_(*(xrange(nsubdiv),)*ndim)):
        nodeoff += off * (nsubdiv + 1) ** dim

    # Tile standard element node ordering mapping, then apply offsets
    internal_con = np.tile(conbase, (nsubdiv ** ndim, 1))
    internal_con += nodeoff.T.flatten()[:, None]

    return np.hstack(internal_con)


def _tri_con(nsubdiv):
    """Generate node connectivity for vtu triangles in high-order elements

    :param nsubdiv: Number of subdivisions (equal to element shape order)
    :type ndim: integer
    :type nsubdiv: integer
    :rtype: list

    """

    conlst = []

    for row in xrange(nsubdiv, 0, -1):
        # Lower and upper indices
        l = (nsubdiv - row)*(nsubdiv + row + 3) // 2
        u = l + row + 1

        # Base offsets
        off = [l, l + 1, u, u + 1, l + 1, u]

        # Generate current row
        subin = np.ravel(np.arange(row - 1)[...,None] + off)
        subex = [ix + row - 1 for ix in off[:3]]

        # Extent list
        conlst.extend([subin, subex])

    return np.hstack(conlst)


def _tet_con(nsubdiv):
    """Generate node connectivity for vtu tet in high-order elements

    :param ndim: Number of dimensions [3]
    :param nsubdiv: Number of subdivisions (equal to element shape order)
    :type nsubdiv: integer
    :rtype: list

    Produce six different mappings for six different cell orientations
    """
    conlst = []
    jump = 0
    for n in xrange(nsubdiv, 0, -1):
        for row in xrange(n, 0, -1):
            # Lower and upper indices
            l = (n - row)*(n + row + 3) // 2 + jump
            u = l + row + 1

            # Lower and upper for one row up
            ln = (n + 1)*(n + 2) // 2 + l - n + row
            un = ln + row

            rowm1 = np.arange(row - 1)[...,None]

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


def _pri_con(nsubdiv):
    # Triangle connectivity
    tcon = _tri_con(nsubdiv).reshape(-1, 3)

    # Layer these rows of triangles to define prisms
    loff = (nsubdiv + 1)*(nsubdiv + 2) // 2
    lcon = [[tcon + i*loff, tcon + (i + 1)*loff] for i in xrange(nsubdiv)]

    return np.hstack(np.hstack(l).flat for l in lcon)


def _pyr_con(nsubdiv):
    if nsubdiv > 1:
        raise RuntimeError('Subdivision is not implemented for pyramids.')

    return [0, 1, 3, 2, 4]


def _base_con(etype, nsubdiv):
    """Switch case to select node connectivity for supported vtu elements

    PyFR high-order elements are subdivided into low-order vtu
    cells to permit visualisation in Paraview.  Cells are defined
    in vtk files by specifying connectivity between nodes.  To
    reduce memory requirements, it is possible to use a single
    high-order node multiple times in defining low-order cells.

    :param etype: PyFR element type
    :param nsubdiv: Number of subdivisions (equal to element shape order)
    :type etype: string
    :type nsubdiv: integer
    :rtype: list

    """
    connec_map = {
        'tri': _tri_con,
        'tet': _tet_con,
        'pri': _pri_con,
        'pyr': _pyr_con,
        'quad': lambda n: _quadcube_con(2, n),
        'hex': lambda n: _quadcube_con(3, n)
    }

    try:
        return connec_map[etype](nsubdiv)
    except KeyError:
        raise RuntimeError('Connectivity not implemented for ' + etype)


def _write_vtu_header(args, vtuf, m_inf, s_inf, off):
    # Set vtk name for float, set size in bytes
    if args.precision == 'single':
        flt = ['Float32', 4]
    else:
        flt = ['Float64', 8]

    # Get the basis class
    basiscls = subclass_where(BaseShape, name=m_inf[0])

    nele = _ncells_after_subdiv(m_inf, args.divisor)
    npts = basiscls.nspts_from_order(args.divisor + 1)*m_inf[1][1]

    # Standard template for vtk DataArray string
    darray = '<DataArray Name="%s" type="%s" NumberOfComponents="%s" '\
             'format="appended" offset="%d"/>\n'

    # Write headers for vtu elements
    vtuf.write('<Piece NumberOfPoints="%s" NumberOfCells="%s">\n<Points>\n'
               % (npts, nele))

    # Lists of DataArray "names", "types" and "NumberOfComponents"
    nams = ['', 'connectivity', 'offsets', 'types', 'Density', 'Velocity',
            'Pressure']
    typs = [flt[0], 'Int32', 'Int32', 'UInt8'] + [flt[0]] * 3
    ncom = ['3', '', '', '', '1', str(m_inf[1][2]), '1']

    # Calculate size of described DataArrays (in bytes)
    offs = np.array([0, 3, 4 * nele * ParaviewWriter.vtk_types[m_inf[0]][1],
                     4 * nele, nele, 1, m_inf[1][2], 1])
    offs[[1,5,6,7]] *= flt[1] * npts

    # Write vtk DaraArray headers
    for i in xrange(len(nams)):
        vtuf.write(darray % (nams[i], typs[i], ncom[i],
                             sum(offs[:i+1]) + i*4 + off))

        # Write ends/starts of vtk file objects
        if i == 0:
            vtuf.write('</Points>\n<Cells>\n')
        elif i == 3:
            vtuf.write('</Cells>\n<PointData>\n')

    # Write end of vtk element data
    vtuf.write('</PointData>\n</Piece>\n')

    # Return the number of bytes appended
    return sum(offs) + 4*len(nams)


def _write_vtu_data(args, vtuf, cfg, mesh, m_inf, soln, s_inf):
    dtype = 'float32' if args.precision == 'single' else 'float64'

    # Get the basis class
    basiscls = subclass_where(BaseShape, name=m_inf[0])

    ndims = m_inf[1][2]
    npts = basiscls.nspts_from_order(args.divisor + 1)

    # Generate basis objects for solution and vtu output
    soln_b = basiscls(m_inf[1][0], cfg)
    vtu_b = basiscls(npts, cfg)

    # Generate operator matrices to move points and solutions to vtu nodes
    mesh_vtu_op = soln_b.sbasis.nodal_basis_at(vtu_b.spts)
    soln_vtu_op = soln_b.ubasis.nodal_basis_at(vtu_b.spts)

    # Calculate node locations of vtu elements
    pts = np.dot(mesh_vtu_op, mesh.reshape(m_inf[1][0],
                                           -1)).reshape(npts, -1, ndims)
    # Calculate solution at node locations of vtu elements
    sol = np.dot(soln_vtu_op, soln.reshape(s_inf[1][0],
                                           -1)).reshape(npts, s_inf[1][1], -1)

    # Append dummy z dimension for points in 2-d (required by Paraview)
    if ndims == 2:
        pts = np.append(pts, np.zeros(pts.shape[:-1])[...,None], axis=2)

    # Write element node locations to file
    _write_vtk_darray(pts.swapaxes(0, 1), vtuf, dtype)

    # Prepare vtu cell arrays (connectivity, offsets, types):
    # Generate and extend vtu sub-cell node connectivity across all elements
    vtu_con = np.tile(_base_con(m_inf[0], args.divisor),
                      (m_inf[1][1], 1))
    vtu_con += (np.arange(m_inf[1][1]) * npts)[:, None]

    # Generate offset into the connectivity array for the end of each element
    vtu_off = np.arange(_ncells_after_subdiv(m_inf, args.divisor)) + 1
    vtu_off *= ParaviewWriter.vtk_types[m_inf[0]][1]

    # Tile vtu cell type numbers
    vtu_typ = np.tile(ParaviewWriter.vtk_types[m_inf[0]][0],
                      _ncells_after_subdiv(m_inf, args.divisor))

    # Write vtu node connectivity, connectivity offsets and cell types
    _write_vtk_darray(vtu_con, vtuf, 'int32')
    _write_vtk_darray(vtu_off, vtuf, 'int32')
    _write_vtk_darray(vtu_typ, vtuf, 'uint8')

    # Convert rhou, rhov, [rhow] to u, v, [w] and energy to pressure
    _component_to_physical_soln(sol, cfg.getfloat('constants', 'gamma'))

    # Write Density, Velocity and Pressure
    _write_vtk_darray(sol[:,0].T, vtuf, dtype)
    _write_vtk_darray(sol[:,1:-1].transpose(2, 0, 1), vtuf, dtype)
    _write_vtk_darray(sol[:,-1].T, vtuf, dtype)
