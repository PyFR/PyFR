import os
import uuid

import h5py
import numpy as np

from pyfr.mpiutil import Gatherer, get_comm_rank_root, mpi
from pyfr.quadrules import get_quadrule
from pyfr.util import file_path_gen, mv


class NativeWriter:
    def __init__(self, intg, basedir, basename, prefix, *, extn='.pyfrs'):
        comm, rank, root = get_comm_rank_root()

        self.cfg = intg.cfg

        # Tally up how many elements of each type our partition has
        self._ecounts = {etype: len(intg.system.mesh.eidxs.get(etype, []))
                         for etype in intg.system.mesh.etypes}

        # Data prefix
        self.prefix = prefix

        # Data type
        self.fpdtype = intg.backend.fpdtype

        # Append the relevant extension
        if not basename.endswith(extn):
            basename += extn

        # Output counter (incremented each time write() is called)
        self.fgen = file_path_gen(basedir, basename, intg.isrestart)

        # Temporary file name
        if rank == root:
            tname = os.path.join(basedir, f'pyfr-{uuid.uuid4()}{extn}')
        else:
            tname = None

        self.tname = comm.bcast(tname, root=root)

        # Parallel I/O
        if (h5py.get_config().mpi and
            'PYFR_FORCE_SERIAL_HDF5' not in os.environ):
            self._write = self._write_parallel
        # Serial I/O
        else:
            self._write = self._write_serial

    def set_shapes_eidxs(self, shapes, eidxs):
        comm, rank, root = get_comm_rank_root()

        # Prepare the element information
        self._einfo = {}
        for etype, ecount in self._ecounts.items():
            # See if any ranks want to write elements of this type
            eshape = comm.allgather(shapes.get(etype, None))
            if any(eshape):
                # Create a gatherer for this element type
                idxs = eidxs.get(etype, [])
                gatherer = Gatherer(comm, idxs)

                # Exchange counts and offsets
                noff = comm.allgather((gatherer.cnt, gatherer.off))
                noff = {i: j for i, j in enumerate(noff) if j}

                # Determine the final shape of the element array
                shape = (gatherer.tot, *next(es for es in eshape if es))

                # See if the element is being subset
                subset = comm.allreduce(len(idxs) != ecount, op=mpi.LOR)

                # Also get the associated nodal points
                rname = self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
                upts = get_quadrule(etype, rname, shape[2]).pts

                self._einfo[etype] = (gatherer, subset, shape, noff, upts)

    def write(self, data, tcurr, metadata=None):
        comm, rank, root = get_comm_rank_root()

        if metadata:
            if rank != root:
                raise ValueError('Metadata must be written by the root rank')

            # Convert all strings to arrays
            metadata = dict(metadata)
            for k, v in metadata.items():
                if isinstance(v, str):
                    metadata[k] = np.array(v.encode(), dtype='S')

        # Gather the solution data into contiguous arrays
        gdata = {}
        for etype, (gatherer, subset, shape, *_) in self._einfo.items():
            if etype in data:
                dset = data[etype]
            else:
                dset = np.empty((0, *shape[1:]), dtype=self.fpdtype)

            gdata[etype] = gatherer(dset)

        # Delegate to _write to do the actual outputting
        self._write(self.tname, gdata)

        # Determine the final output path
        path = self.fgen.send(tcurr)

        # Add in the metadata
        if rank == root:
            with h5py.File(self.tname, 'r+') as f:
                self._write_meta(f, metadata)

            # Move the file to its final location
            mv(self.tname, path)

        # Wait for everyone to finish
        comm.barrier()

        # Return the path
        return path

    def _write_meta(self, f, metadata):
        # Write the metadata
        for k, v in metadata.items():
            if isinstance(v, str):
                f[k] = np.array(v.encode(), dtype='S')
            else:
                f[k] = v

        # Add each elements nodal points as an attribute
        for etype, (*_, upts) in self._einfo.items():
            f[f'{self.prefix}/{etype}'].attrs['pts'] = upts

    def _write_parallel(self, path, data):
        comm, rank, root = get_comm_rank_root()
        prefix = self.prefix

        kwargs = {'driver': 'mpio', 'comm': comm, 'libver': 'latest'}
        with h5py.File(path, 'w', **kwargs) as f:
            # Collectively create all of the datasets
            for etype, (gatherer, subset, shape, *_) in self._einfo.items():
                f.create_dataset(f'{prefix}/{etype}', shape, self.fpdtype)
                f.create_dataset(f'{prefix}/{etype}_parts', shape[0:1],
                                 np.int32)

                if subset:
                    f.create_dataset(f'{prefix}/{etype}_idxs', shape[0:1],
                                     np.int64)

            # Write out our element data
            for etype, (gatherer, subset, *_) in self._einfo.items():
                region = slice(gatherer.off, gatherer.off + gatherer.cnt)
                f[f'{prefix}/{etype}'][region] = data[etype]
                f[f'{prefix}/{etype}_parts'][region] = gatherer.rsrc

                # If the element has been subset then write the index data
                if subset:
                    f[f'{prefix}/{etype}_idxs'][region] = gatherer.ridx

    def _write_serial(self, path, data):
        einfo = self._einfo
        comm, rank, root = get_comm_rank_root()

        if rank != root:
            for etype, (gatherer, subset, *_) in einfo.items():
                if not gatherer.cnt:
                    continue

                # Send the data to the root rank for writing
                comm.Send(np.ascontiguousarray(data[etype]), root)
                comm.Send(np.ascontiguousarray(gatherer.rsrc), root)

                # And, if needed, the associated indices
                if subset:
                    comm.Send(np.ascontiguousarray(gatherer.ridx), root)
        else:
            with h5py.File(path, 'w', libver='latest') as f:
                # Collect and write solution and index data
                for etype, ei in einfo.items():
                    gatherer, subset, shape, noff, upts = ei
                    dtype = self.fpdtype

                    # Allocate the solution and partition datasets
                    fdata = f.create_dataset(f'{self.prefix}/{etype}', shape,
                                             dtype=dtype)
                    fpart = f.create_dataset(f'{self.prefix}/{etype}_parts',
                                             gatherer.tot, dtype=np.int32)

                    # If needed allocate an index dataset
                    if subset:
                        fidx = f.create_dataset(f'{self.prefix}/{etype}_idxs',
                                                gatherer.tot, dtype=int)

                    # Receive the data from the remaining ranks
                    for i, (n, off) in noff.items():
                        if i != root:
                            rdata = np.empty((n, *shape[1:]), dtype=dtype)
                            rpart = np.empty(n, dtype=np.int32)
                            comm.Recv(rdata, i)
                            comm.Recv(rpart, i)
                        else:
                            rdata = data[etype]
                            rpart = gatherer.rsrc

                        # Write out the data
                        fdata[off:off + n] = rdata
                        fpart[off:off + n] = rpart

                        if subset:
                            if i != root:
                                ridx = np.empty(n, dtype=int)
                                comm.Recv(ridx, i)
                            else:
                                ridx = gatherer.ridx

                            # Write out the indices
                            fidx[off:off + n] = ridx
