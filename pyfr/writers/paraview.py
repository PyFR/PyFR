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

    # PyFR to VTK element types, node mapping, (number of nodes)
    vtk_to_pyfr = {'tri': (5, [0, 1, 2], 3),
                   'quad': (9, [0, 1, 3, 2], 4),
                   'tet': (10, [0, 1, 2, 3], 4),
                   'pyr': (14, [0, 1, 3, 2, 4], 5),
                   'pri': (13, [0, 1, 2, 3, 4, 5], 6),
                   'hex': (12, [0, 1, 3, 2, 4, 5, 7, 6], 8)}

    def write_out(self):
        """Controls the writing of serial .vtu Paraview files

        Writes .vtu pieces for each element type, in each partition of
        the PyFR files.  The Paraview data type used is "appended",
        which concatenates all data into a single block of binary data
        at the end of the file.  ASCII headers written at the top of
        the file describe the structure of this data.

        Two different methods for outputting the data are used; one
        that involves subdividing a high order element into low
        order elements (at the shape points), and another that appends
        high-order data to be read by an external plugin detailed here:
        http://perso.uclouvain.be/sebastien.blaise/tools.html

        The methods are switched such that the latter is used when
        self.args.divisor is None. When it is not none, the former
        method is used on a high order element of shape order =
        self.args.divsior. An initial 0 value sets itself to the
        solution order.

        """
        # Set default divisor to solution order
        if self.args.divisor == 0:
            self.args.divisor = self.cfg.getint('solver', 'order')

        # Write .vtu file header
        self.outf.write('<?xml version="1.0" ?>\n<VTKFile '
                        'byte_order="LittleEndian" type="UnstructuredGrid" '
                        'version="0.1">\n<UnstructuredGrid>\n')

        # Initialise offset (in bytes) to end of appended data
        off = np.array([0])

        # Write data description header.  A vtk "piece" is used for each
        # element in a partition.
        for i, mk in enumerate(self.mesh_inf.iterkeys()):
            sk = self.soln_inf.keys()[i]

            _write_vtu_header(self.args, self.outf, self.mesh_inf[mk],
                              self.soln_inf[sk], off)

        # Write end/start of header/data sections
        self.outf.write('</UnstructuredGrid>\n<AppendedData '
                        'encoding="raw">\n_')

        # Write data "piece"wise
        for i, mk in enumerate(self.mesh_inf.iterkeys()):
            sk = self.soln_inf.keys()[i]

            _write_vtu_data(self.args, self.outf, self.cfg, self.mesh[mk],
                            self.mesh_inf[mk], self.soln[sk],
                            self.soln_inf[sk])

        # Write .vtu file footer
        self.outf.write('\n</AppendedData>\n</VTKFile>')


def _write_vtk_darray(array, vtuf, numtyp):
    """Writes a numpy array to a vtu file (in binary as type numtyp)

    .vtu files require the size of the data (in bytes) to be prepended.

    :param array: Array to be written to file.
    :param vtuf: File to write array to.
    :param numtyp: Type of number representation to use. e.g. 'float32'
    :type array: numpy.ndrarray
    :type vtuf: file
    :type numtyp: string

    """
    np.array(array.astype(numtyp).nbytes).astype('uint32').tofile(vtuf)
    array.astype(numtyp).tofile(vtuf)


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
    # Catch all for cases where cell subdivision is not performed
    if divisor is None:
        divisor = 1

    # Calculate the number of low-order cells in each high order element
    n_sub_ele = divisor ** ms_inf[1][2]

    # Pyramids require the further addition of an arithmetic series
    if ms_inf[0] == 'pyr':
        n_sub_ele += (divisor - 1) * divisor / 2

    # Multiply by number of elements
    return n_sub_ele * ms_inf[1][1]


def _npts_from_order(order, m_inf, total=True):
    """Calculates the number of nodes in an element of order n

    :param order: Shape order of element
    :param ms_inf: Mesh/soln information. ('ele_type', [npts, nele, ndims])
    :param total: True/False return nodes in element/nodes in partition
    :type order: integer
    :type ms_inf: tuple: (str, list)
    :type total: bool
    :rtype: integer

    """
    # Calculate number of nodes in element of type and order
    if m_inf[0] in ['quad', 'hex']:
        gen_npts = (order + 1)**m_inf[1][2]
    elif m_inf[0] in ['tri', 'pri']:
        gen_npts = (order + 2) * (order + 1)**(m_inf[1][2] - 1) / 2
    elif m_inf[0] == 'tet':
        gen_npts = (order + 1) * (order + 2) * (order + 3) / 6
    elif m_inf == 'pyr':
        gen_npts = (order + 1) * (order + 2) * (2 * order + 3) / 6

    # Multiply by number of elements
    if total:
        return gen_npts * m_inf[1][1]
    else:
        return gen_npts


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
    :type ndim: integer
    :type nsubdiv: integer
    :rtype: list

    """
    connec_map = {
        'tri': _tri_con,
        'tet': _tet_con,
        'pri': _pri_con,
        'quad': lambda n: _quadcube_con(2, n),
        'hex': lambda n: _quadcube_con(3, n)
    }

    try:
        return connec_map[etype](nsubdiv or 1)
    except KeyError:
        raise RuntimeError('Connectivity not implemented for ' + etype)


def _write_vtu_header(args, vtuf, m_inf, s_inf, off):
    """Write headers for .vtu piece objects and subsequent appended DataArrays

    The .vtk data format used by the Paraview converter is "appended",
    which requires ASCII headers to be written to define the contents
    of the appended data arrays.

    The function handles both "append" and "divide" high-order data
    output options.  The "append" option requires analogous data
    to the "divide" option to define the low-order cells.  The
    high-order data is appended to the end of the piece as "CellData".

    :param args: pyfr-postp command line arguments from argparse
    :param vtuf: .vtu output file
    :param m_inf: Tuple of element type and array shape of mesh
    :param s_inf: Tuple of element type and array shape of soln
    :param off: Offset (in bytes) to end of appended data
    :type args: class 'argparse.Namespace'
    :type vtuf: file
    :type m_inf: tuple
    :type s_inf: tuple
    :type off: type 'numpy.ndarray'

    """
    # Set vtk name for float, set size in bytes
    if args.precision == 'single':
        flt = ['Float32', 4]
    else:
        flt = ['Float64', 8]

    # Assign variables dependent on output mode
    if args.divisor is not None:
        nele = _ncells_after_subdiv(m_inf, args.divisor)
        npts = _npts_from_order(args.divisor, m_inf)
    else:
        nele = m_inf[1][1]
        npts = m_inf[1][1] * ParaviewWriter.vtk_to_pyfr[m_inf[0]][2]

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
    offs = np.array([0, 3, 4 * nele * ParaviewWriter.vtk_to_pyfr[m_inf[0]][2],
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
    vtuf.write('</PointData>\n')

    # Store total offset (bytes) to end of appended data
    off += sum(offs) + 4*len(nams)

    # Write headers for appended high-order data, if required
    if args.divisor is None:
        vtuf.write('<CellData>\n')

        # Size of described high order data (in bytes)
        hooffs = np.array([3, 1, m_inf[1][2], 1]) * nele * flt[1]

        # Number of high-order nodes per ele (less those written as low-order)
        nhpts = s_inf[1][0] - ParaviewWriter.vtk_to_pyfr[m_inf[0]][2]

        # Lists of requisite DataArray "names" and "NumberOfComponents"
        nams = ['HOcoord', 'Density_HOsol', 'Velocity_HOsol', 'Pressure_HOsol']
        ncom = ['3', 1, str(m_inf[1][2]), 1]

        # Equivalent high-order nodes are written to the same array
        for spt in xrange(nhpts):
            # Write DataArrays as named in nams for each node
            for i in xrange(len(nams)):
                vtuf.write(darray % ('_'.join((nams[i], str(spt))), flt[0],
                                     ncom[i], sum(hooffs[:i]) + off + 4*i))

            # Update total byte offset to current end of appended data
            off += sum(hooffs) + 4*len(nams)

        # Write ends of vtk objects
        vtuf.write('</CellData>\n')
    vtuf.write('</Piece>\n')


def _write_vtu_data(args, vtuf, cfg, mesh, m_inf, soln, s_inf):
    """ Writes mesh and solution data for appended (binary) data .vtu files

    :param args: pyfr-postp command line arguments from argparse
    :param vtuf: .vtu output file
    :param cfg: PyFR config file used in the respective simulation
    :param mesh: Single PyFR mesh array (corresponding to soln)
    :param m_inf: Tuple of element type and array shape of mesh
    :param soln: Single PyFR solution array (corresponding to mesh)
    :param s_inf: Tuple of element type and array shape of soln
    :type args: class 'argparse.Namespace'
    :type vtuf: file
    :type cfg: class 'pyfr.inifile.Inifile'
    :type mesh: numpy.ndarray
    :type m_inf: tuple
    :type soln: numpy.ndrayy
    :type s_inf: tuple

    """
    # Set numpy name for float; set size in bytes
    if args.precision == 'single':
        flt = ['float32', 4]
    else:
        flt = ['float64', 8]

    # Get the basis class
    basiscls = subclass_where(BaseShape, name=m_inf[0])
    ndims = m_inf[1][2]

    # Set npts for divide/append cases
    if args.divisor is not None:
        npts = _npts_from_order(args.divisor, m_inf, total=False)
    else:
        npts = ParaviewWriter.vtk_to_pyfr[m_inf[0]][2]

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
    _write_vtk_darray(pts.swapaxes(0,1), vtuf, flt[0])

    # Prepare vtu cell arrays (connectivity, offsets, types):
    # Generate and extend vtu sub-cell node connectivity across all elements
    vtu_con = np.tile(_base_con(m_inf[0], args.divisor),
                      (m_inf[1][1], 1))
    vtu_con += (np.arange(m_inf[1][1]) * npts)[:, None]

    # Generate offset into the connectivity array for the end of each element
    vtu_off = np.arange(_ncells_after_subdiv(m_inf, args.divisor)) + 1
    vtu_off *= ParaviewWriter.vtk_to_pyfr[m_inf[0]][2]

    # Tile vtu cell type numbers
    vtu_typ = np.tile(ParaviewWriter.vtk_to_pyfr[m_inf[0]][0],
                      _ncells_after_subdiv(m_inf, args.divisor))

    # Write vtu node connectivity, connectivity offsets and cell types
    _write_vtk_darray(vtu_con, vtuf, 'int32')
    _write_vtk_darray(vtu_off, vtuf, 'int32')
    _write_vtk_darray(vtu_typ, vtuf, 'uint8')

    # Convert rhou, rhov, [rhow] to u, v, [w] and energy to pressure
    _component_to_physical_soln(sol, cfg.getfloat('constants', 'gamma'))

    # Write Density, Velocity and Pressure
    _write_vtk_darray(sol[:,0].T, vtuf, flt[0])
    _write_vtk_darray(sol[:,1:-1].transpose(2, 0, 1), vtuf, flt[0])
    _write_vtk_darray(sol[:,-1].T, vtuf, flt[0])

    # Append high-order data as CellData if not dividing cells
    if args.divisor is None:
        # Calculate number of points written as low-order, and left to append
        nlpts = ParaviewWriter.vtk_to_pyfr[s_inf[0]][2]
        nhpts = s_inf[1][0] - nlpts

        # Generate basis objects for mesh, solution and vtu output
        mesh_b = basiscls(m_inf[1][0], cfg)

        # Get location of spts in standard element of solution order
        uord = cfg.getint('solver', 'order')
        ele_spts = subclass_where(BaseShape, name=m_inf[0]).std_ele(uord)

        # Generate operator matrices to move points and solutions to vtu nodes
        mesh_hpts_op = mesh_b.sbasis.nodal_basis_at(ele_spts)
        soln_hpts_op = mesh_b.ubasis.nodal_basis_at(ele_spts)

        # Calculate node locations of vtu elements
        pts = np.dot(mesh_hpts_op, mesh.reshape(m_inf[1][0], -1))
        pts = pts.reshape((-1,) + m_inf[1][1:])

        # Append dummy z dimension to 2-d points (required by Paraview)
        if ndims == 2:
            pts = np.append(pts, np.zeros(pts.shape[:-1])[...,None], axis=2)

        # Calculate solution at node locations
        sol = np.dot(soln_hpts_op, soln.reshape(s_inf[1][0], -1))
        sol = sol.reshape((-1,) + s_inf[1][1:])

        # Convert rhou, rhov, [rhow] to u, v, [w] and energy to pressure
        _component_to_physical_soln(sol, cfg.getfloat('constants', 'gamma'))

        # Write data arrays, one set of high order points at a time
        for gmshpt in xrange(nhpts):
            # Convert Gmsh node number to equivalent in PyFR
            pt = GmshNodeMaps.to_pyfr[s_inf[0], s_inf[1][0]][gmshpt + nlpts]

            # Write node locations, density, velocity and pressure
            _write_vtk_darray(pts[pt], vtuf, flt[0])
            _write_vtk_darray(sol[pt,0], vtuf, flt[0])
            _write_vtk_darray(sol[pt,1:-1].T, vtuf, flt[0])
            _write_vtk_darray(sol[pt,-1], vtuf, flt[0])
