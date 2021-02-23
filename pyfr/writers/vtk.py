# -*- coding: utf-8 -*-

from collections import defaultdict
import os
import re

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where
from pyfr.writers import BaseWriter
from pyfr.writers.nodemaps import VTKHONodeMaps


class VTKWriter(BaseWriter):
    # Supported file types and extensions
    name = 'vtk'
    extn = ['.vtu', '.pvtu']

    vtk_types_ho = dict(tri=69, quad=70, tet=71, pri=73, hex=72)

    def __init__(self, args):
        super().__init__(args)

        self.dtype = np.dtype(args.precision).type

        # Choose whether to output subdivided cells or high order VTK cells
        # If -k is given by the user then use high-order VTK cells as output
        # with order equal to the solution order or to the one provided by
        # the user
        # Else if neither -o nor -d are found in the input then use high-order
        # VTK cells with order equal to the simulation order
        # Else use cell subdivision
        if args.order:
            self.ho_output = True
            self.divisor = args.order
        elif args.divisor is None:
            self.ho_output = True
            self.divisor = self.cfg.getint('solver', 'order')
        else:
            self.ho_output = False
            self.divisor = args.divisor

        # If outputting high-order VTK cells choose version 2.1
        # to ensure consistency with VTK9 mappings
        # of high order vtkLagrangeHexahedron cells
        # See nodemaps.py for more information on
        # why VTK9 maps are chosen over those of VTK8
        self.vtkfile_version = '2.1' if self.ho_output else '0.1'

        # Solutions need a separate processing pipeline to other data
        if self.dataprefix == 'soln':
            self._pre_proc_fields = self._pre_proc_fields_soln
            self._post_proc_fields = self._post_proc_fields_soln
            self._soln_fields = list(self.elementscls.privarmap[self.ndims])
            self._vtk_vars = list(self.elementscls.visvarmap[self.ndims])
        # Otherwise we're dealing with simple scalar data
        else:
            self._pre_proc_fields = self._pre_proc_fields_scal
            self._post_proc_fields = self._post_proc_fields_scal
            self._soln_fields = self.stats.get('data', 'fields').split(',')
            self._vtk_vars = [(k, [k]) for k in self._soln_fields]

        if self.ho_output:
            self._get_npts_ncells_nnodes = self._get_npts_ncells_nnodes_ho
        else:
            self._get_npts_ncells_nnodes = self._get_npts_ncells_nnodes_lin

        # See if we are computing gradients
        if args.gradients:
            self._pre_proc_fields_ref = self._pre_proc_fields
            self._pre_proc_fields = self._pre_proc_fields_grad
            self._post_proc_fields = self._post_proc_fields_grad

            # Update list of solution fields
            self._soln_fields.extend(
                f'{f}-{d}'
                for f in list(self._soln_fields) for d in range(self.ndims)
            )

            # Update the list of VTK variables to solution fields
            nf = lambda f: [f'{f}-{d}' for d in range(self.ndims)]
            for var, fields in list(self._vtk_vars):
                if len(fields) == 1:
                    self._vtk_vars.append((f'grad {var}', nf(fields[0])))
                else:
                    self._vtk_vars.extend(
                        (f'grad {var} {f}', nf(f)) for f in fields
                    )

    def _pre_proc_fields_soln(self, name, mesh, soln):
        # Convert from conservative to primitive variables
        return np.array(self.elementscls.con_to_pri(soln, self.cfg))

    def _pre_proc_fields_scal(self, name, mesh, soln):
        return soln

    def _post_proc_fields_soln(self, vsoln):
        # Primitive and visualisation variable maps
        privarmap = self.elementscls.privarmap[self.ndims]
        visvarmap = self.elementscls.visvarmap[self.ndims]

        # Prepare the fields
        fields = []
        for fnames, vnames in visvarmap:
            ix = [privarmap.index(vn) for vn in vnames]

            fields.append(vsoln[ix])

        return fields

    def _post_proc_fields_scal(self, vsoln):
        return [vsoln[self._soln_fields.index(v)] for v, _ in self._vtk_vars]

    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Call the reference pre-processor
        soln = self._pre_proc_fields_ref(name, mesh, soln)

        # Dimensions
        nvars, nupts = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.swapaxes(0, 1).reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.vstack([soln, gradsoln])

    def _post_proc_fields_grad(self, vsoln):
        # Prepare the fields
        fields = []
        for vname, vfields in self._vtk_vars:
            ix = [self._soln_fields.index(vf) for vf in vfields]

            fields.append(vsoln[ix])

        return fields

    def _get_npts_ncells_nnodes_lin(self, sk):
        etype, neles = self.soln_inf[sk][0], self.soln_inf[sk][1][2]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)
        subdvcls = subclass_where(BaseShapeSubDiv, name=etype)

        # Number of vis points
        npts = shapecls.nspts_from_order(self.divisor + 1)*neles

        # Number of sub cells and nodes
        ncells = len(subdvcls.subcells(self.divisor))*neles
        nnodes = len(subdvcls.subnodes(self.divisor))*neles

        return npts, ncells, nnodes

    def _get_npts_ncells_nnodes_ho(self, sk):
        etype, neles = self.soln_inf[sk][0], self.soln_inf[sk][1][2]

        if etype == 'pyr':
            # No Lagrange pyr cells in VTK
            # Therefore, rely on the subdivision mechanism
            # of the vtk writer
            return self._get_npts_ncells_nnodes_lin(sk)

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)

        # Number of vis points
        # which coincides with the number of
        # nodes of the vtkLagrange* correspondent
        # objects
        npts = shapecls.nspts_from_order(self.divisor + 1)*neles

        return npts, neles, npts

    def _get_array_attrs(self, sk=None):
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'
        dsize = np.dtype(self.dtype).itemsize

        vvars = self._vtk_vars

        names = ['', 'connectivity', 'offsets', 'types']
        types = [dtype, 'Int32', 'Int32', 'UInt8']
        comps = ['3', '', '', '']

        for fname, varnames in vvars:
            names.append(fname.title())
            types.append(dtype)
            comps.append(str(len(varnames)))

        # If a solution has been given the compute the sizes
        if sk:
            npts, ncells, nnodes = self._get_npts_ncells_nnodes(sk)
            nb = npts*dsize

            sizes = [3*nb, 4*nnodes, 4*ncells, ncells]
            sizes.extend(len(varnames)*nb for fname, varnames in vvars)

            return names, types, comps, sizes
        else:
            return names, types, comps

    @memoize
    def _get_shape(self, name, nspts):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, self.cfg)

    @memoize
    def _get_std_ele(self, name, nspts):
        return self._get_shape(name, nspts).std_ele(self.divisor)

    @memoize
    def _get_mesh_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    @memoize
    def _get_soln_op(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

    def write_out(self):
        name, extn = os.path.splitext(self.outf)
        parallel = extn == '.pvtu'

        parts = defaultdict(list)
        for sk, (etype, shape) in self.soln_inf.items():
            part = sk.split('_')[-1]
            pname = f'{name}_{part}.vtu' if parallel else self.outf

            parts[pname].append((f'spt_{etype}_{part}', sk))

        write_s_to_fh = lambda s: fh.write(s.encode())

        for pfn, misil in parts.items():
            with open(pfn, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="UnstructuredGrid" '
                              f'version="{self.vtkfile_version}">\n'
                              '<UnstructuredGrid>\n')

                # Running byte-offset for appended data
                off = 0

                # Header
                for mk, sk in misil:
                    off = self._write_serial_header(fh, sk, off)

                write_s_to_fh('</UnstructuredGrid>\n'
                              '<AppendedData encoding="raw">\n_')

                # Data
                for mk, sk in misil:
                    self._write_data(fh, mk, sk)

                write_s_to_fh('\n</AppendedData>\n</VTKFile>')

        if parallel:
            with open(self.outf, 'wb') as fh:
                write_s_to_fh('<?xml version="1.0" ?>\n<VTKFile '
                              'byte_order="LittleEndian" '
                              'type="PUnstructuredGrid" '
                              f'version="{self.vtkfile_version}">\n'
                              '<PUnstructuredGrid>\n')

                # Header
                self._write_parallel_header(fh)

                # Constitutent pieces
                for pfn in parts:
                    bname = os.path.basename(pfn)
                    write_s_to_fh(f'<Piece Source="{bname}"/>\n')

                write_s_to_fh('</PUnstructuredGrid>\n</VTKFile>\n')

    def _write_darray(self, array, vtuf, dtype):
        array = array.astype(dtype)

        np.uint32(array.nbytes).tofile(vtuf)
        array.tofile(vtuf)

    def _process_name(self, name):
        return re.sub(r'\W+', '_', name)

    def _write_serial_header(self, vtuf, sk, off):
        names, types, comps, sizes = self._get_array_attrs(sk)
        npts, ncells = self._get_npts_ncells_nnodes(sk)[:2]

        write_s = lambda s: vtuf.write(s.encode())
        write_s(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')
        write_s('<Points>\n')

        # Write vtk DaraArray headers
        for i, (n, t, c, s) in enumerate(zip(names, types, comps, sizes)):
            write_s(f'<DataArray Name="{self._process_name(n)}" type="{t}" '
                    f'NumberOfComponents="{c}" '
                    f'format="appended" offset="{off}"/>\n')

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

    def _write_parallel_header(self, vtuf):
        names, types, comps = self._get_array_attrs()

        write_s = lambda s: vtuf.write(s.encode())
        write_s('<PPoints>\n')

        # Write vtk DaraArray headers
        for i, (n, t, s) in enumerate(zip(names, types, comps)):
            write_s(f'<PDataArray Name="{self._process_name(n)}" type="{t}" '
                    f'NumberOfComponents="{s}"/>\n')

            if i == 0:
                write_s('</PPoints>\n<PCells>\n')
            elif i == 3:
                write_s('</PCells>\n<PPointData>\n')

        write_s('</PPointData>\n')

    def _write_data(self, vtuf, mk, sk):
        name = self.mesh_inf[mk][0]
        mesh = self.mesh[mk].astype(self.dtype)
        soln = self.soln[sk].swapaxes(0, 1).astype(self.dtype)

        # Handle the case of partial solution files
        if soln.shape[2] != mesh.shape[1]:
            skpre, skpost = sk.rsplit('_', 1)

            mesh = mesh[:, self.soln[f'{skpre}_idxs_{skpost}'], :]

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = self._get_std_ele(name, nspts)
        nsvpts = len(svpts)

        if name != 'pyr' and self.ho_output:
            # Transform PyFR to VTK9 points
            # See nodemaps.py for more information on
            # why VTK9 maps are chosen over those of VTK8
            # High order `pyr` elements are not currently
            # supported in VTK
            svpts = np.array(svpts)[VTKHONodeMaps.to_pyfr[name, nsvpts]]

        # Generate the operator matrices
        mesh_vtu_op = self._get_mesh_op(name, nspts, svpts)
        soln_vtu_op = self._get_soln_op(name, nspts, svpts)

        # Calculate node locations of VTU elements
        vpts = mesh_vtu_op @ mesh.reshape(nspts, -1)
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Pre-process the solution
        soln = self._pre_proc_fields(name, mesh, soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = soln_vtu_op @ soln.reshape(len(soln), -1)
        vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Write element node locations to file
        self._write_darray(vpts.swapaxes(0, 1), vtuf, self.dtype)

        # Perform the sub division
        if name != 'pyr' and self.ho_output:
            nodes = np.arange(nsvpts)
            subcellsoff = nsvpts
            types = self.vtk_types_ho[name]
        else:
            subdvcls = subclass_where(BaseShapeSubDiv, name=name)
            nodes = subdvcls.subnodes(self.divisor)
            subcellsoff = subdvcls.subcelloffs(self.divisor)
            types = subdvcls.subcelltypes(self.divisor)

        # Prepare VTU cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]

        # Generate offset into the connectivity array
        vtu_off = np.tile(subcellsoff, (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

        # Tile VTU cell type numbers
        vtu_typ = np.tile(types, neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, vtuf, np.int32)
        self._write_darray(vtu_off, vtuf, np.int32)
        self._write_darray(vtu_typ, vtuf, np.uint8)

        # Process and write out the various fields
        for arr in self._post_proc_fields(vsoln):
            self._write_darray(arr.T, vtuf, self.dtype)


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
        nodeoff = np.zeros((n,)*cls.ndim, dtype=np.int)
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

        return np.hstack([np.ravel(c) for c in conlst])


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

        return np.hstack([np.hstack(l).flat for l in lcon])


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
                lower_pts = np.hstack([range(k*(n + 1)+1, (k + 1)*n + k)
                                       for k in range(1, n)]) + l

                # Second set of pyramids
                lcon.append([upper_quad[:, ::-1], lower_pts])

                lower_row = _row_in_quad(n + 1, 1, -1) + l
                lower_col = _col_in_quad(n + 1, 1, -1) + l

                upper_row = _row_in_quad(n) + u
                upper_col = _col_in_quad(n) + u

                # Tetrahedra
                lcon.append([lower_col, upper_row])
                lcon.append([lower_row[:, ::-1], upper_col])

        return np.hstack([np.column_stack(l).flat for l in lcon])
