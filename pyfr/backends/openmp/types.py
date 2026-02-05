from collections import defaultdict
from ctypes import c_int
from functools import cached_property

import pyfr.backends.base as base
from pyfr.backends.openmp.provider import (OpenMPBlockKernelArgs,
                                           OpenMPRegularRunArgs,
                                           OpenMPKRunArgs)
from pyfr.ctypesutil import make_array
from pyfr.mpiutil import mpi


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
    needs_pdeps = False

    def __init__(self, backend):
        super().__init__(backend)

        self.klist = []
        self.kskip = set()
        self.kins = defaultdict(list)

    def _get_kranges(self):
        kranges, i = {}, 0

        for k, j in self.knodes.items():
            kranges[k] = range(i, j)
            i = j

        return kranges

    def _get_nblocks(self, idxs):
        return max(self.klist[i].runargs.b.nblocks for i in idxs)

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        rra = OpenMPRegularRunArgs(fun=mpi.funcs.Start, args=mpi.addrof(req))
        kra = OpenMPKRunArgs(ktype=OpenMPKRunArgs.KTYPE_REGULAR, r=rra)

        self.kins[len(self.klist)].append(kra)

    def _group_splits(self, kerns, kranges):
        nblocks = self._get_nblocks([ix for k in kerns for ix in kranges[k]])
        blocksz = self.backend.csubsz
        gkerns, splits = {}, set()

        # Process the kernels and identify any split points
        for k in kerns:
            # Extract the block arguments for the kernels
            runargs = {j: self.klist[j].runargs.b for j in kranges[k]}

            # Handle compound (split) kernels
            if k.compound:
                ksplits = [0] + [s // blocksz for s in k.splits] + [nblocks]
                splits.update(ksplits[1:-1])

                for i, (j, ra) in enumerate(runargs.items()):
                    gkerns[j] = (*ksplits[i:i + 2], ra.kernels[0])
            else:
                for i, (j, ra) in enumerate(runargs.items()):
                    gkerns[j] = (0, ra.nblocks, ra.kernels[0])

        # Process the split information
        splits = sorted(splits)
        splits = [(i, j - i) for i, j in zip([0] + splits, splits + [nblocks])]

        return gkerns, splits

    def _group_subs(self, subs, kranges):
        allocsz = 0
        argsubs, argmasks = defaultdict(list), defaultdict(int)

        for s in subs:
            for k, aname in s:
                for j in kranges[k]:
                    aidx = self.klist[j].arg_idx(aname)
                    aoff = self.klist[j].arg_off(aidx)
                    absz = self.klist[j].arg_blocksz(aidx)
                    suboff = self.klist[j].subs_off(aidx)

                    argsubs[j].append((aoff, allocsz + suboff))
                    argmasks[j] |= 1 << aidx

            allocsz += absz

        return allocsz, argsubs, argmasks

    def group(self, kerns, subs=[]):
        super().group(kerns, subs)

        kranges = self._get_kranges()

        # Handle split kernels
        gkerns, splits = self._group_splits(kerns, kranges)

        # Handle argument substitutions
        allocsz, argsubs, argmasks = self._group_subs(subs, kranges)

        # Construct the groupings
        groups = []
        for off, n in splits:
            bkernels, bsubs = [], []
            for j, (start, end, bka) in gkerns.items():
                if start <= off and end >= off + n:
                    for aoff, nbytes in argsubs[j]:
                        bsubs.extend([len(bkernels), aoff, nbytes])

                    bkernels.append(OpenMPBlockKernelArgs(
                        fun=bka.fun, args=bka.args, argsz=bka.argsz,
                        argmask=argmasks[j], offset=off - start
                    ))

            rargs = OpenMPKRunArgs(ktype=OpenMPKRunArgs.KTYPE_BLOCK_GROUP)
            rargs.b.nblocks = n
            rargs.b.allocsz = allocsz
            rargs.b.nkerns = len(bkernels)
            rargs.b.nsubs = len(bsubs) // 3
            rargs.b.kernels = make_array(bkernels)
            rargs.b.subs = make_array(bsubs, type=c_int)

            groups.append(rargs)

        # Arrange for the groupings to be inserted into the final run list
        gdeps = [dep for k in kerns
                 for dep in self.kdeps[k] if dep not in kerns]
        lk = max((self.knodes[dep] for dep in gdeps), default=-1)
        gid = min(self.knodes[k] for k in kerns if self.knodes[k] > lk) - 1

        # Sanity check that other dependencies haven't been violated
        for k in self.kdeps:
            if k not in kerns:
                for dep in self.kdeps[k]:
                    if dep in kerns and self.knodes[k] < gid:
                        raise RuntimeError('Graph grouping dependency error')

        self.kins[gid].extend(groups)

        # Finally, prevent grouped being added to the final run list
        for k in kerns:
            self.kskip.update(kranges[k])

    def commit(self):
        super().commit()

        rlist = []

        for i, k in enumerate(self.klist):
            if i in self.kins:
                rlist.extend(self.kins[i])

            if i not in self.kskip:
                rlist.append(k.runargs)

        self._runlist = make_array(rlist)

    def run(self):
        # Start all dependency-free MPI requests
        self._startall(self.mpi_root_reqs)

        # Run the kernels (including MPI_Start requests)
        self.backend.krunner(len(self._runlist), self._runlist)

        # Wait for all of the MPI requests to finish
        self._waitall(self.mpi_reqs)
