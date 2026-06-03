import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.fields.runner import FieldRunner
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where
from pyfr.writers import BaseWriter
from pyfr.writers.vtk.shapes import get_vtk_shape


class BaseVTKWriter(BaseWriter):
    name = 'vtk'
    extn = ['.vtu', '.pvtu']

    # Type of export (volume/boundary/STL)
    type = None

    # If to output curvature data
    output_curved = False

    # Per-etype divisor bump for pyramids
    pyr_divisor_bump = 0

    def __init__(self, mesh, cfg, *, prec='single', order=None, divisor=None,
                 add_fields=[], remove_fields=[], field_cfg=None,
                 discontinuous=False):
        self.mesh = mesh
        self.cfg = cfg

        if not re.fullmatch(self.dimensions, str(mesh.ndims)):
            raise RuntimeError(f'{mesh.ndims}D grids not supported')

        self.dtype = np.dtype(prec).type

        self._field_names = add_fields
        self._remove_fields = set(remove_fields)
        self._field_cfg = field_cfg

        self._clean = not discontinuous

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

        # Per-etype divisor
        divisor_default = self.divisor or self.cfg.getint('solver', 'order')
        self.etypes_div = defaultdict(lambda: divisor_default)
        self.etypes_div['pyr'] += self.pyr_divisor_bump

        self._init_einfo()
        self._region = self._build_region()

        self.field_runner = FieldRunner(self._field_names, mesh.ndims,
                                        self._field_cfg or self.cfg, self.type)

        self.tcurr = None

    def emit(self, snap, outfname):
        if self._field_names and snap.prefix != 'soln':
            raise ValueError(
                f'Field providers are only supported for conservative-form '
                f'solution files (snap.prefix = {snap.prefix!r})')

        # MPI collective must be in lockstep across ranks
        snap.compute_grads()
        self._sample = self._region.sample(snap)
        self._post_sample(snap)
        self._sample.run(self.field_runner, public_only=True)

        if unknown := self._remove_fields - set(self._sample.fields):
            raise RuntimeError(
                f'--remove-fields names not in output pool: '
                f'{sorted(unknown)} '
                f'(available: {sorted(self._sample.fields)})')

        if snap.prefix == 'soln':
            self.tcurr = getattr(snap, 'tcurr', None)
        else:
            self.tcurr = None

        if Path(outfname).suffix == '.vtu':
            self._write_vtu(outfname)
        else:
            self._write_pvtu(outfname)

    def _init_einfo(self):
        raise NotImplementedError

    def _build_region(self):
        raise NotImplementedError

    def _post_sample(self, snap):
        pass

    def _emit_fields(self, kind):
        for name, info in self._sample.fields.items():
            if info.kind != kind:
                continue
            if name in self._remove_fields:
                continue
            yield name, info

    def _nsvpts(self, etype):
        div = self.etypes_div[etype]
        return subclass_where(BaseShape, name=etype).npts_from_order(div)

    def _get_npts_ncells_nnodes_lin(self, etype, neles):
        div = self.etypes_div[etype]

        # Get the number of subdivided nodes
        subdv = get_vtk_shape(etype, div)
        ncells = len(subdv.subcells)*neles
        nnodes = len(subdv.subnodes)*neles

        return self._region.npts(etype), ncells, nnodes

    def _get_npts_ncells_nnodes_ho(self, etype, neles):
        # Fallback to subdivision for pyramids
        if etype == 'pyr':
            return self._get_npts_ncells_nnodes_lin(etype, neles)

        nsvpts = self._nsvpts(etype)
        return self._region.npts(etype), neles, neles*nsvpts

    def _array_attrs(self):
        attrs = [('', self._vtk_dtype(self.dtype), '3'),
                 ('connectivity', 'Int64', ''),
                 ('offsets', 'Int64', ''),
                 ('types', 'UInt8', '')]

        if self.output_curved:
            attrs.append(('Curved', 'UInt8', '1'))

        for name, info in self._emit_fields('cell'):
            attrs.append((name.replace('-', ' ').title(),
                          self._vtk_dtype(info.dtype), str(info.ncomps)))

        for name, info in self._emit_fields('point'):
            attrs.append((name.replace('-', ' ').title(),
                          self._vtk_dtype(info.dtype), str(info.ncomps)))

        return attrs

    def _array_sizes(self, npts, ncells, nnodes):
        sizes = [3*npts*np.dtype(self.dtype).itemsize,
                 8*nnodes, 8*ncells, ncells]

        if self.output_curved:
            sizes.append(ncells)

        for name, info in self._emit_fields('cell'):
            sizes.append(info.dtype.itemsize*info.ncomps*ncells)

        for name, info in self._emit_fields('point'):
            sizes.append(info.dtype.itemsize*info.ncomps*npts)

        return sizes

    def _refpts_fn(self, shapecls, _shape=None):
        # Subdivided sample points, permuted to VTK HO node order when emitting
        # HO cells (non-pyr — pyrs always linearise).  Used by both the vis
        # SnapshotRegion and the SurfaceSnapshotRegion (face shapecls is quad
        # or tri, so the pyr guard is a no-op there).
        div = self.etypes_div[shapecls.name]
        svpts = shapecls.std_ele(div)
        if shapecls.name != 'pyr' and self.ho_output:
            svpts = svpts[get_vtk_shape(shapecls.name, div).nodemaps[len(svpts)]]
        return svpts

    def _point_field_data(self, etype):
        region = self._region
        sample = self._sample
        fields = []

        for name, info in self._emit_fields('point'):
            arr = sample.field_array(etype, info)
            if arr is None:
                continue

            if info.source in ('primitive', 'gradient'):
                if not region.clean:
                    arr = arr.swapaxes(0, 1)
            else:
                if region.clean:
                    arr = arr[:, None] if arr.ndim == 1 else arr.T
                else:
                    arr = (arr.swapaxes(0, 1)[..., None] if arr.ndim == 2
                           else arr.transpose(2, 1, 0))

            fields.append((np.ascontiguousarray(arr, dtype=info.dtype),
                           info.dtype))

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

    def _write_serial_header(self, write_s, npts, ncells, nnodes, off):
        ncelld = self.output_curved + sum(1 for _ in self._emit_fields('cell'))

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
        ncelld = self.output_curved + sum(1 for _ in self._emit_fields('cell'))
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
        region = self._region
        neles = dict(self.einfo)[etype]

        self._write_darray(region.points(etype,
                                         self._sample.ploc[etype]),
                                         write, self.dtype)

        # VTK-specific sub-cell layout tables.
        if etype != 'pyr' and self.ho_output:
            nsvpts = self._nsvpts(etype)
            nodes = np.arange(nsvpts)
            subcellsoff = nsvpts
            types = get_vtk_shape(etype, self.etypes_div[etype]).vtk_ho_type
        else:
            subdiv = get_vtk_shape(etype, self.etypes_div[etype])
            nodes = subdiv.subnodes
            subcellsoff = subdiv.subcelloffs
            types = subdiv.subcelltypes

        # Connectivity: region applies clean / raw layout to the writer's
        # per-element sub-node template.
        vtu_con = region.connectivity(etype, nodes)
        vtu_off = np.tile(subcellsoff, (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]
        vtu_typ = np.tile(types, neles)

        self._write_darray(vtu_con, write, np.int64)
        self._write_darray(vtu_off, write, np.int64)
        self._write_darray(vtu_typ, write, np.uint8)

        if self.output_curved:
            curved = region.cell_curved(etype)
            vtu_curved = np.repeat(curved, len(vtu_typ) // neles)
            self._write_darray(vtu_curved, write, np.uint8)

        ncells_per_ele = len(vtu_typ) // neles
        for name, info in self._emit_fields('cell'):
            data = self._sample.field_arrays.get((etype, name))
            if data is None:
                continue
            vtu_aux = data.reshape(neles, -1)
            vtu_aux = np.repeat(vtu_aux, ncells_per_ele, axis=0)
            self._write_darray(vtu_aux, write, info.dtype)

        for arr, dtype in self._point_field_data(etype):
            self._write_darray(arr, write, dtype)
