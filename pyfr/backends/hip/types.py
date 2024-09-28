from collections import defaultdict
from functools import cached_property

import numpy as np

import pyfr.backends.base as base


class _HIPMatrixCommon:
    @cached_property
    def _as_parameter_(self):
        return self.data


class HIPMatrixBase(_HIPMatrixCommon, base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata
        self.data = int(self.basedata) + offset
        self.offset = offset

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        # Allocate an empty buffer
        buf = np.empty((self.nrow, self.leaddim), dtype=self.dtype)

        # Copy
        self.backend.hip.memcpy(buf, self.data, self.nbytes)

        # Unpack
        return self._unpack(buf)

    def _set(self, ary):
        buf = self._pack(ary)

        # Copy
        self.backend.hip.memcpy(self.data, buf, self.nbytes)


class HIPMatrixSlice(_HIPMatrixCommon, base.MatrixSlice):
    @cached_property
    def data(self):
        return int(self.basedata) + self.offset


class HIPMatrix(HIPMatrixBase, base.Matrix): pass
class HIPConstMatrix(HIPMatrixBase, base.ConstMatrix): pass
class HIPView(base.View): pass
class HIPXchgView(base.XchgView): pass


class HIPXchgMatrix(HIPMatrix, base.XchgMatrix):
    def __init__(self, backend, dtype, ioshape, initval, extent, aliases,
                 tags):
        # Call the standard matrix constructor
        super().__init__(backend, dtype, ioshape, initval, extent, aliases,
                         tags)

        # If MPI is HIP-aware then simply annotate our device buffer
        if backend.mpitype == 'hip-aware':
            class HostData:
                __array_interface__ = {
                    'version': 3,
                    'typestr': np.dtype(self.dtype).str,
                    'data': (self.data, False),
                    'shape': (self.nrow, self.ncol)
                }

            self.hdata = np.array(HostData(), copy=False)
        # Otherwise, allocate a buffer on the host for MPI to send/recv from
        else:
            shape = (self.nrow, self.ncol)
            self.hdata = backend.hip.pagelocked_empty(shape, dtype)


class HIPGraph(base.Graph):
    needs_pdeps = True

    def __init__(self, backend):
        super().__init__(backend)

        self.graph = backend.hip.create_graph()
        self.stale_kparams = {}
        self.mpi_events = []

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        if deps:
            event = self.backend.hip.create_event()
            self.graph.add_event_record(event, [self.knodes[d] for d in deps])

            self.mpi_events.append((event, req))

    def commit(self):
        super().commit()

        self.exc_graph = self.graph.instantiate()

    def run(self, stream):
        # Ensure our kernel parameters are up to date
        for node, params in self.stale_kparams.items():
            self.exc_graph.set_kernel_node_params(node, params)

        self.exc_graph.launch(stream)
        self.stale_kparams.clear()

        # Start all dependency-free MPI requests
        self._startall(self.mpi_root_reqs)

        # Start any remaining requests once their dependencies are satisfied
        for event, req in self.mpi_events:
            event.synchronize()
            req.Start()

        # Wait for all of the MPI requests to finish
        self._waitall(self.mpi_reqs)
