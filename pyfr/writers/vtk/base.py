from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np

from pyfr.cache import clear_memoize, memoize
from pyfr.fields import FieldRecovery, con_block_to_pri
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.shapes import BaseShape, interp_pts
from pyfr.subdiv import get_subdiv
from pyfr.util import subclass_where
from pyfr.writers import BaseWriter
from pyfr.writers.vtk.output import CleanToGridVTKOutput, DirectVTKOutput


FieldMeta = namedtuple('FieldMeta', 'kind ncomps dtype')


class BaseVTKWriter(BaseWriter):
    # Supported file types and extensions
    name = 'vtk'
    extn = ['.vtu', '.pvtu']

    # Type of export (volume/boundary/STL)
    type = None

    # Adapter kind if it differs from the export type
    adapter_kind = None

    # If to output curvature data
    output_curved = False

    def __init__(self, meshf, pname=None, *, prec='single', order=None,
                 divisor=None, fields=[], pp_plugins=[], pp_cfg=None,
                 discontinuous=False):
        super().__init__(meshf, pname)

        self.dtype = np.dtype(prec).type
        self.fields = fields
        self._pp_plugin_names = pp_plugins
        self._pp_cfg = pp_cfg

        # Determine the output filter
        if discontinuous:
            self._output_cls = DirectVTKOutput
        else:
            self._output_cls = CleanToGridVTKOutput

        # Choose whether to output subdivided cells or high order VTK cells
        if order or divisor is None:
            self.ho_output = True
            self.divisor = order
            self.vtkfile_version = '2.1'
            self._get_npts_ncells_nnodes = self._get_npts_ncells_nnodes_ho
        else:
            self.ho_output = False
            self.divisor = divisor
            self.vtkfile_version = '1.0'
            self._get_npts_ncells_nnodes = self._get_npts_ncells_nnodes_lin

    def _build_extra_fields(self):
        self._extra_fields = {}
        self._classify_aux_fields()
        self._register_pp_fields()

    def _classify_aux_fields(self):
        # Classify aux fields by shape
        for etype, dt in self.soln.dtypes.items():
            pshapes = self._extra_point_shapes(etype)
            for name in dt['aux'].names if 'aux' in dt.names else ():
                shape, base = dt['aux'][name].shape, dt['aux'][name].base

                if shape in pshapes:
                    meta = FieldMeta('point', 1, base)
                elif shape[:-1] in pshapes:
                    meta = FieldMeta('point', shape[-1], base)
                else:
                    meta = FieldMeta('cell', int(np.prod(shape) or 1), base)

                self._extra_fields[name] = meta

    def _register_pp_fields(self):
        # Combine requested plugins with any defaults from the file itself
        names = list(self._pp_plugin_names)
        if self.stats.hasopt('data', 'postproc'):
            names.extend(self.stats.get('data', 'postproc').split())

        # Names the writer already supplies and so never derives on demand;
        # gradients and residuals are synthesised by the writer itself
        provided = self._vtk_vars.keys() | self._extra_fields.keys()
        provided |= {f'{p} {v}' for p in ('grad', 'resid')
                     for v in self._vtk_vars}

        # Resolve the pipeline (deriving any unprovided fields) and register it
        cfg = self._pp_cfg or self.cfg
        want = set(self.fields) if self.fields else None
        self.pp_pipe = self.source.pipeline(names, self.type, cfg, want,
                                            self.adapter_kind, provided)

        for fname, varnames in self.pp_pipe.fields.items():
            meta = FieldMeta('point', len(varnames), np.dtype(self.dtype))
            self._extra_fields[fname] = meta

    def _postproc(self, soln_t, ploc, *extra):
        if self.pp_pipe.plugins:
            return self.pp_pipe(self.soln, soln_t, ploc, *extra)
        else:
            return {}

    def _pre_proc_fields_soln(self, soln):
        return con_block_to_pri(self.elementscls, self.cfg, self.ndims,
                                soln, grads=self._gradients, resid=self._resid)

    def _pre_proc_fields_scal(self, soln):
        return soln

    def _post_proc_fields_soln(self, vsoln):
        # Prepare the fields
        fields = []
        for vnames in self._vtk_vars.values():
            ix = [self._soln_fields.index(vn) for vn in vnames]

            fields.append(vsoln[ix])

        return fields

    def _post_proc_fields_scal(self, vsoln):
        return [vsoln[self._soln_fields.index(k)] for k in self._vtk_vars]

    def _nsvpts(self, etype):
        div = self.etypes_div[etype]
        return subclass_where(BaseShape, name=etype).npts_from_order(div)

    def _get_npts_ncells_nnodes_lin(self, etype, neles):
        div = self.etypes_div[etype]

        # Get the number of points
        nsvpts = self._nsvpts(etype)

        # Get the number of subdivided nodes
        sdiv = get_subdiv(etype, div)
        ncells = len(sdiv.subcells)*neles
        nnodes = len(sdiv.subnodes)*neles

        return self._output.npts(etype, neles, nsvpts), ncells, nnodes

    def _get_npts_ncells_nnodes_ho(self, etype, neles):
        # Fallback to subdivision for pyramids
        if etype == 'pyr':
            counts = self._get_npts_ncells_nnodes_lin(etype, neles)
        else:
            nsvpts = self._nsvpts(etype)
            npts = self._output.npts(etype, neles, nsvpts)
            counts = npts, neles, neles*nsvpts

        return counts

    def _array_attrs(self):
        vvars = self._vtk_vars

        # Floating point data type and size
        dtype = 'Float32' if self.dtype == np.float32 else 'Float64'

        # Base array attributes
        attrs = [('', dtype, '3'), ('connectivity', 'Int64', ''),
                 ('offsets', 'Int64', ''), ('types', 'UInt8', '')]

        if self.output_curved:
            attrs.append(('Curved', 'UInt8', '1'))

        cell_fields, point_fields = self._extra_field_lists()

        # Extra fields as cell data
        for fname in cell_fields:
            adtype, _, acomps = self._field_info(fname)
            attrs.append((fname.replace('-', ' ').title(), adtype,
                          str(acomps)))

        for fname, varnames in vvars.items():
            attrs.append((fname.title(), dtype, str(len(varnames))))

        # Extra fields as point data
        for fname in point_fields:
            adtype, _, acomps = self._field_info(fname)
            attrs.append((fname.replace('-', ' ').title(), adtype,
                          str(acomps)))

        return attrs

    def _array_sizes(self, npts, ncells, nnodes):
        dsize = np.dtype(self.dtype).itemsize
        nb = npts*dsize

        sizes = [3*nb, 8*nnodes, 8*ncells, ncells]

        if self.output_curved:
            sizes.append(ncells)

        cell_fields, point_fields = self._extra_field_lists()

        # Extra cell field sizes
        for fname in cell_fields:
            _, asize, _ = self._field_info(fname)
            sizes.append(asize*ncells)

        sizes.extend(len(vn)*nb for vn in self._vtk_vars.values())

        # Extra point field sizes
        for fname in point_fields:
            _, asize, _ = self._field_info(fname)
            sizes.append(asize*npts)

        return sizes

    def _load_soln(self, *args, **kwargs):
        super()._load_soln(*args, **kwargs)

        # Determine the per-etype divisor
        divisor = self.divisor or self.cfg.getint('solver', 'order')
        self.etypes_div = defaultdict(lambda: divisor)
        self.etypes_div['pyr'] += self._output_cls.pyr_divisor_bump

        # Solutions need a separate processing pipeline to other data
        if self.dataprefix == 'soln':
            self._pre_proc_fields = self._pre_proc_fields_soln
            self._post_proc_fields = self._post_proc_fields_soln
            self._soln_fields = self.elementscls.privars(self.ndims, self.cfg)
            self._vtk_vars = self.elementscls.visvars(self.ndims, self.cfg)
            self.tcurr = self.stats.getfloat('solver-time-integrator', 'tcurr')

            # See if our solution contains gradient or residual data
            self._gradients = bool(self.soln.grad_data)
            self._resid = bool(self.soln.resid_data)
        # Otherwise we're dealing with simple scalar data (e.g., tavg)
        else:
            self._pre_proc_fields = self._pre_proc_fields_scal
            self._post_proc_fields = self._post_proc_fields_scal
            self._soln_fields = list(self.soln.fields)
            self._vtk_vars = {k: [k] for k in self._soln_fields}
            self.tcurr = None
            self._gradients = self._resid = False

        # Classify aux + register pp output fields
        self._build_extra_fields()

        # Restrict the extra fields and plugins to any requested subset
        if want := set(self.fields or []):
            self._subset_fields(want)

        # Synthesise corrected gradients for files without stored ones
        if self.dataprefix == 'soln' and not self._gradients:
            if (any(f.startswith('grad ') for f in want) or
                self.pp_pipe.needs_grads):
                self._synth_grads()

        # Stack any gradient and residual blocks needed for the output
        if self._gradients or self._resid:
            self._stack_blocks(want)

        # Give the data source a chance to augment the solution
        self.source.prepare(self.mesh, self.soln, self.pp_pipe.plugins)

        # Prune the output variables themselves
        if want:
            self._vtk_vars = {f: v for f, v in self._vtk_vars.items()
                              if f in want}

    def _subset_fields(self, want):
        # Validate the request against everything the file can provide
        known = self._vtk_vars.keys() | self._extra_fields.keys()
        if self._gradients or self.dataprefix == 'soln':
            known |= {f'grad {v}' for v in self._vtk_vars}
        if self._resid:
            known |= {f'resid {v}' for v in self._vtk_vars}

        if missing := want - known:
            raise RuntimeError('Invalid field specification: '
                               f'{", ".join(sorted(missing))}')

        self._extra_fields = {f: m for f, m in self._extra_fields.items()
                              if f in want}

    def _synth_grads(self):
        rec = FieldRecovery(self.mesh, self.elementscls, self.cfg)
        bcrows = ('con', list(range(len(self._soln_fields))))

        for et, g in rec.grad_corrected(self.soln.data, bcrows).items():
            self.soln.grad_data[et] = g.transpose(2, 0, 1, 3)

        self._gradients = True

    def _stack_blocks(self, want):
        pnames = list(self._soln_fields)

        # Stack a data block and derive its field and VTK names
        def stack_block(data, prefix, nmap):
            for et in list(self.soln.data):
                self.soln.data[et] = np.concatenate(
                    [self.soln.data[et], data(et)], axis=1
                )

            self._soln_fields.extend(nmap(pnames))
            for var, vfields in list(self._vtk_vars.items()):
                if not var.startswith(('grad ', 'resid ')):
                    self._vtk_vars[f'{prefix} {var}'] = nmap(vfields)

        def wanted(p):
            return not want or any(f.startswith(f'{p} ') for f in want)

        if self._gradients:
            # Postproc plugins consume the gradient block positionally
            if wanted('grad') or self.pp_pipe.needs_grads:
                def gdata(et):
                    g = self.soln.grad_data[et].transpose(1, 2, 0, 3)
                    return g.reshape(g.shape[0], -1, g.shape[3])

                stack_block(gdata, 'grad',
                            lambda fs: [f'{f}-{d}' for f in fs
                                        for d in range(self.ndims)])
            else:
                self._gradients = False

            self.soln.grad_data = {}

        if self._resid:
            if wanted('resid'):
                stack_block(lambda et: self.soln.resid_data[et], 'resid',
                            lambda fs: [f'resid-{f}' for f in fs])
            else:
                self._resid = False

            self.soln.resid_data = {}

    def list_fields(self, solnf):
        self._load_soln(solnf)

        fields = {f: len(v) for f, v in self._vtk_vars.items()}
        fields |= {f: m.ncomps for f, m in self._extra_fields.items()}

        return fields

    def process(self, solnf, outfname):
        # Clear per-solution memoize caches
        clear_memoize(self)

        # Load the solution
        self._load_soln(solnf)

        # Prepare element data
        self._prepared = {et: self._prepare_pts(et) for et, _ in self.einfo}
        self._output = self._output_cls(self)

        if Path(outfname).suffix == '.vtu':
            self._write_vtu(outfname)
        else:
            self._write_pvtu(outfname)

    def _point_field_data(self, etype):
        _, vsoln, _, _, pointf = self._prepared[etype]
        nsvpts, neles = vsoln.shape[0], vsoln.shape[2]

        fields = []
        for arr in self._post_proc_fields(vsoln.swapaxes(0, 1)):
            arr = arr[..., None] if arr.ndim == 2 else arr.transpose(1, 2, 0)
            fields.append((arr, self.dtype))

        for fname in self._extra_field_lists()[1]:
            arr = pointf[fname].reshape(nsvpts, neles, -1)
            fields.append((arr, self._extra_fields[fname].dtype))

        return fields

    def _local_counts(self):
        return [self._get_npts_ncells_nnodes(et, ne)
                for et, ne in self.einfo]

    def _write_piece(self, write_s, write_b):
        write_s('<?xml version="1.0" ?>\n<VTKFile '
                'byte_order="LittleEndian" type="UnstructuredGrid" '
                f'version="{self.vtkfile_version}" '
                'header_type="UInt64">\n<UnstructuredGrid>\n')
        self._write_piece_headers(write_s, self._local_counts())
        write_s('</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')
        for etype, _ in self.einfo:
            self._write_data(write_b, etype)
        write_s('\n</AppendedData>\n</VTKFile>')

    def _write_piece_headers(self, write_s, counts, off=0):
        for c in counts:
            off = self._write_serial_header(write_s, *c, off)
        return off

    def _write_vtu(self, fname):
        comm, rank, root = get_comm_rank_root()

        fh = mpi.File.Open(comm, fname, mpi.MODE_CREATE | mpi.MODE_WRONLY)
        write_s = lambda s: fh.Write(s.encode())

        # Gather per-rank array counts to the root rank
        gcounts = comm.gather(self._local_counts(), root=root)

        # If we have any header information then write it
        if rank == root:
            write_s('<?xml version="1.0" ?>\n<VTKFile '
                    'byte_order="LittleEndian" type="UnstructuredGrid" '
                    f'version="{self.vtkfile_version}" '
                    'header_type="UInt64">\n<UnstructuredGrid>\n')

            if self.tcurr is not None:
                self._write_time_value(write_s)

            # Running byte-offset for appended data
            soffs, off = [], 0

            # Write out the array headers for each rank's pieces
            for ecounts in gcounts:
                soffs.append(off)
                off = self._write_piece_headers(write_s, ecounts, off)

            write_s('</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')

            # Get the size of the header
            hsize = fh.Get_position()

            # Use this to displace the offsets
            soffs = [s + hsize for s in soffs]

            # Compute the total size of the file sans footer
            size = hsize + off
        else:
            size, soffs = None, None

        # Distribute the total size and starting offset information
        size = comm.bcast(size, root=root)
        soff = comm.scatter(soffs, root=root)

        # Allocate space in the file
        fh.Set_size(size)

        # Have the root rank also write out the footer
        if rank == root:
            fh.Seek(0, mpi.SEEK_END)
            write_s('\n</AppendedData>\n</VTKFile>')

        # Seek to our region of the file
        fh.Seek(soff, mpi.SEEK_SET)

        # Write out our ranks data
        for etype, _ in self.einfo:
            self._write_data(lambda b: fh.Write(b), etype)

        # Wait for all ranks to finish writing
        fh.Close()

    def _write_pvtu(self, fname):
        comm, rank, root = get_comm_rank_root()

        # Have each rank write out its own VTU file
        if self.einfo:
            with open(f'{fname[:-5]}_p{rank}.vtu', 'wb') as fh:
                self._write_piece(lambda s: fh.write(s.encode()),
                                  lambda b: fh.write(b))

        # Inform the root rank if we wrote a file or not
        fidx = comm.gather(bool(self.einfo), root=root)

        # Also have the root rank write out the PVTU file itself
        if rank == root:
            with open(fname, 'wb') as fh:
                write_s = lambda s: fh.write(s.encode())
                write_s('<?xml version="1.0" ?>\n<VTKFile '
                        'byte_order="LittleEndian" type="PUnstructuredGrid" '
                        f'version="{self.vtkfile_version}">\n'
                        '<PUnstructuredGrid>\n')

                if self.tcurr is not None:
                    self._write_time_value(write_s)

                # Header
                self._write_parallel_header(write_s)

                # Constituent pieces
                for r, w in enumerate(fidx):
                    if w:
                        bname = Path(f'{fname[:-5]}_p{r}.vtu').name
                        write_s(f'<Piece Source="{bname}"/>\n')

                write_s('</PUnstructuredGrid>\n</VTKFile>\n')

    def _write_darray(self, array, write, dtype):
        array = np.ascontiguousarray(array, dtype=dtype)

        write(np.uint64(array.nbytes))
        write(array)

    def _component_names(self, ncomps):
        cnames = {
            '2': ['X', 'Y'],
            '3': ['X', 'Y', 'Z'],
            '4': ['XX', 'XY', 'YX', 'YY'],
            '9': ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
        }

        if ncomps in cnames:
            return ' '.join(f'ComponentName{i}="{n}"'
                            for i, n in enumerate(cnames[ncomps]))
        else:
            return ''

    _vtk_dtypes = {
        np.int32: 'Int32', np.int64: 'Int64',
        np.uint8: 'UInt8', np.uint32: 'UInt32',
        np.float32: 'Float32', np.float64: 'Float64'
    }

    def _vtk_dtype(self, dtype):
        return self._vtk_dtypes[np.dtype(dtype).type]

    @memoize
    def _get_shape(self, etype, cfg):
        nspts = self.reader.f[f'eles/{etype}'].dtype['nodes'].shape[0]
        return subclass_where(BaseShape, name=etype)(nspts, cfg)

    def _extra_point_shapes(self, etype):
        dtype = self.soln.dtypes[etype]
        group = next(g for g in dtype.names if g != 'aux')
        shape = self._get_shape(etype, self.cfg)
        return {dtype[group][0].shape[-1:], (len(shape.linspts),)}

    def _extra_field_lists(self):
        cfields, pfields = [], []
        for name, meta in self._extra_fields.items():
            lst = pfields if meta.kind == 'point' else cfields
            lst.append(name)
        return cfields, pfields

    def _field_info(self, name):
        meta = self._extra_fields[name]
        asize = meta.dtype.itemsize * meta.ncomps
        return self._vtk_dtype(meta.dtype), asize, meta.ncomps

    def _write_serial_header(self, write_s, npts, ncells, nnodes, off):
        cell_fields, _ = self._extra_field_lists()
        ncelld = self.output_curved + len(cell_fields)

        write_s(f'<Piece NumberOfPoints="{npts}" '
                f'NumberOfCells="{ncells}">\n<Points>\n')

        # Write VTK DataArray headers
        attrs = zip(self._array_attrs(),
                    self._array_sizes(npts, ncells, nnodes))
        for i, ((n, t, c), s) in enumerate(attrs):
            write_s(f'<DataArray Name="{n}" type="{t}" '
                    f'NumberOfComponents="{c}" {self._component_names(c)} '
                    f'format="appended" offset="{off}"/>\n')

            off += 8 + s

            # Points => Cells => CellData => PointData transition
            if i == 0:
                write_s('</Points>\n<Cells>\n')
            if i == 3:
                write_s('</Cells>\n<CellData>\n')
            if i == 3 + ncelld:
                write_s('</CellData>\n<PointData>\n')

        # Close
        write_s('</PointData>\n</Piece>\n')

        # Return the current offset
        return off

    def _write_parallel_header(self, write_s):
        cell_fields, _ = self._extra_field_lists()
        ncelld = self.output_curved + len(cell_fields)
        write_s('<PPoints>\n')

        # Write VTK DataArray headers
        for i, (n, t, c) in enumerate(self._array_attrs()):
            write_s(f'<PDataArray Name="{n}" type="{t}" '
                    f'NumberOfComponents="{c}" {self._component_names(c)}/>\n')

            # Points => Cells => CellData => PointData transition
            if i == 0:
                write_s('</PPoints>\n<PCells>\n')
            if i == 3:
                write_s('</PCells>\n<PCellData>\n')
            if i == 3 + ncelld:
                write_s('</PCellData>\n<PPointData>\n')

        # Close
        write_s('</PPointData>\n')

    def _write_time_value(self, write_s):
        write_s('<FieldData>\n'
                '<DataArray Name="TimeValue" type="Float64" '
                'NumberOfComponents="1" NumberOfTuples="1" format="ascii">\n'
                f'{self.tcurr}\n'
                '</DataArray>\n</FieldData>\n')

    def _write_data(self, write, etype):
        vpts, vsoln, curved, cellf, _ = self._prepared[etype]
        nsvpts, neles = vsoln.shape[0], vsoln.shape[2]

        # Write element node locations
        out = self._output.points(etype, vpts)
        self._write_darray(out, write, self.dtype)

        # Perform the sub division
        sdiv = get_subdiv(etype, self.etypes_div[etype])
        if etype != 'pyr' and self.ho_output:
            nodes = np.arange(nsvpts)
            subcellsoff = nsvpts
            types = sdiv.vtk_ho_type
        else:
            nodes = sdiv.subnodes
            subcellsoff = sdiv.subcelloffs
            types = sdiv.vtk_subcelltypes

        # Prepare VTU cell arrays
        vtu_con = self._output.connectivity(etype, nodes, neles, nsvpts)

        # Generate offset into the connectivity array
        vtu_off = np.tile(subcellsoff, (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]

        # Tile VTU cell type numbers
        vtu_typ = np.tile(types, neles)

        # Write VTU node connectivity, connectivity offsets and cell types
        self._write_darray(vtu_con, write, np.int64)
        self._write_darray(vtu_off, write, np.int64)
        self._write_darray(vtu_typ, write, np.uint8)

        # VTU cell curvature information
        if self.output_curved:
            vtu_curved = np.repeat(curved, len(vtu_typ) // neles)
            self._write_darray(vtu_curved, write, np.uint8)

        # Extra cell fields (iterate in header order)
        cfields, _ = self._extra_field_lists()
        ncells_per_ele = len(vtu_typ) // neles
        for fname in cfields:
            data = cellf[fname]
            vtu_aux = data.reshape(neles, -1)
            vtu_aux = np.repeat(vtu_aux, ncells_per_ele, axis=0)
            self._write_darray(vtu_aux, write, self._extra_fields[fname].dtype)

        # Point fields
        for arr, dtype in self._output.point_fields(etype):
            self._write_darray(arr, write, dtype)
