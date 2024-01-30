from collections import defaultdict
from ctypes import addressof, c_int, c_void_p, cast, sizeof

from functools import cached_property
import numpy as np

import pyfr.backends.base as base
from pyfr.backends.openmp.provider import (OpenMPBlockKernelArgs,
                                           OpenMPBlockRunArgs,
                                           OpenMPKRunArgs)
from pyfr.ctypesutil import make_array


class OpenMPMatrixBase(base.MatrixBase):
    def onalloc(self, basedata, offset):
        self.basedata = basedata.ctypes.data

        self.data = basedata[offset:offset + self.nbytes]
        self.data = self.data.view(self.dtype)
        self.data = self.data.reshape(self.nblocks, self.nrow, self.leaddim)

        self.offset = offset

        # Pointer to our ndarray (used by ctypes)
        self._as_parameter_ = self.data.ctypes.data

        # Process any initial value
        if self._initval is not None:
            self._set(self._initval)

        # Remove
        del self._initval

    def _get(self):
        return self._unpack(self.data)

    def _set(self, ary):
        self.data[:] = self._pack(ary)


class OpenMPMatrix(OpenMPMatrixBase, base.Matrix):
    @cached_property
    def hdata(self):
        return self.data


class OpenMPMatrixSlice(base.MatrixSlice):
    @cached_property
    def data(self):
        return self.parent.data[self.ba:self.bb, self.ra:self.rb, :]

    @cached_property
    def _as_parameter_(self):
        return self.data.ctypes.data


class OpenMPConstMatrix(OpenMPMatrixBase, base.ConstMatrix): pass
class OpenMPXchgMatrix(OpenMPMatrix, base.XchgMatrix): pass
class OpenMPXchgView(base.XchgView): pass
class OpenMPView(base.View): pass


class OpenMPGraph(base.Graph):
    def __init__(self, backend):
        super().__init__(backend)

        self.klist = []
        self.kgnodes = []
        self.kptrs = None
        self.bkargs = []
        self.b_runargs = []
        self.kg_runargs = []
        self.mpi_idxs = defaultdict(list)

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        if deps:
            ix = max(self.knodes[d] for d in deps)

            self.mpi_idxs[ix].append(req)

    def commit(self):
        super().commit()

        # Group kernels in runs separated by MPI requests
        self._runlist, i = [], 0

        for j in sorted(self.mpi_idxs):
            self._runlist_update(i, j, self.mpi_idxs[j])
            i = j

        if i != len(self.klist) - 1:
            self._runlist_update(i, len(self.klist), [])

    def group(self, knodes, subs=[]):
        self.kgnodes.append([k.kernel for kernel in knodes for k in kernel])

        # Loop over kernel groups per each element type in the domain
        for ks in zip(*knodes):
            self.bkargs.append([])
            bka = []
            for k in ks:
                kernel = k.kernel
                fun = cast(kernel.fun, c_void_p)
                args, argsz = addressof(kernel.kargs), sizeof(kernel.kargs)

                bka.append({'fun':fun, 'args':args, 'argsz':argsz})

            nsubs = 0
            argsubs = []
            allocsz = 0

            # Loop over entries in substitutions
            for elmgr in subs:
                nsubs += len(elmgr)
                for ka_pair in elmgr:
                    krnl, arg = ka_pair
                    for i, k in enumerate(ks):
                        if (krnl == k.kernel.fname):
                            offset = k.kernel.arg_off(arg)
                            ix = k.kernel.argn.index(arg)
                            blksz = k.kernel.argblks[f'arg{ix}']
                            argsubs.append([i, offset, allocsz])

                            bka[i]['argmask'] = ix

                allocsz += blksz

            c_argsubs = make_array(
                np.array(argsubs, dtype=np.int32).flatten(), type=c_int
            )

            self.bkargs.append([OpenMPBlockKernelArgs(**d) for d in bka])
            self.kptrs = make_array(self.bkargs[-1])

            self.b_runargs = OpenMPBlockRunArgs(
                nkerns=len(ks), nblocks=ks[0].kernel.nblocks,
                kernels=self.kptrs,
                nsubs=nsubs, allocsz=allocsz, subs=c_argsubs,
            )

            self.kg_runargs.append(OpenMPKRunArgs(
                ktype=OpenMPKRunArgs.KTYPE_BLOCK_GROUP, b=self.b_runargs
            ))

    def run(self):
        # Start all dependency-free MPI requests
        self._startall(self.mpi_root_reqs)

        for n, krunargs, reqs in self._runlist:
            self.backend.krunner(n, krunargs)

            self._startall(reqs)

        # Wait for all of the MPI requests to finish
        self._waitall(self.mpi_reqs)

    def _runlist_update(self, i, j, mpi_req):
        popidxs = []
        runargs_list = [k.runargs for k in self.klist[i:j]]

        for k_i, k in enumerate(self.klist[i:j]):
            for kgs in self.kgnodes:
                if k in kgs:
                    # Log kernel's index
                    popidxs.append(k_i)

        runargs_list = [ra for ix, ra in enumerate(runargs_list)
                        if ix not in popidxs]

        # Place grouped kernel to an appropriate location in the runlist
        if popidxs:
            runargs_list[popidxs[0]:popidxs[0]] = self.kg_runargs

        count = j - i - len(popidxs) + len(self.kg_runargs)
        krunargs = make_array(runargs_list)
        self._runlist.append((count, krunargs, mpi_req))
