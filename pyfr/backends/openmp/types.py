from collections import defaultdict
from ctypes import c_int
from functools import cached_property
from graphlib import TopologicalSorter

import pyfr.backends.base as base
from pyfr.backends.openmp.provider import OpenMPBlockKernelArgs, OpenMPKRunArgs
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

# Need a hashable wrapper for a list of kernels for the topological sort
class OpenMPGroupNode:
    def __init__(self, groups):
        self.groups = groups

class OpenMPGraph(base.Graph):
    needs_pdeps = False

    def __init__(self, backend):
        super().__init__(backend)

        self.klist = []

        self.mpi_idxs = defaultdict(list)

        self.dep_graph = {}

    def _get_kranges(self):
        kranges, i = {}, 0

        for k, j in self.knodes.items():
            kranges[k] = range(i, j)
            i = j

        return kranges

    def _get_nblocks(self, idxs):
        return max(self.klist[i].runargs.b.nblocks for i in idxs)

    def _make_runlist(self, start, stop):
        # Helper function to handle meta kernels
        def expand_kernels(obj):
            if hasattr(obj, 'kernels'):
                tmp = []
                for k in obj.kernels:
                    tmp.extend(expand_kernels(k))
                return tmp
            else:
                return [obj.kernel.runargs]
        
        rlist = []

        for i in range(start, stop):
            if isinstance(self.run_order[i], OpenMPGroupNode):
                rlist.extend(self.run_order[i].groups)
            else:
                rlist.extend(expand_kernels(self.run_order[i]))

        return make_array(rlist)

    def add(self, kern, deps=[], pdeps=[]):
        super().add(kern, deps, pdeps)

        self.dep_graph[kern] = list(deps)

    def add_mpi_req(self, req, deps=[]):
        super().add_mpi_req(req, deps)

        if deps:
            ix = max(self.knodes[d] for d in deps)

            self.mpi_idxs[ix].append(req)

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
        
        # Need hashable object for topological sort later
        group_node = OpenMPGroupNode(groups)
        # Get the dependencies of the group and remove grouped kernels from dependency graph
        group_deps = set()
        for k in kerns:
            group_deps.update([dep for dep in self.dep_graph[k] if dep not in kerns])
            del self.dep_graph[k]
        # Iterate over all nodes in dependency graph and replace dependencies on grouped kernels
        # with a dependency on the group
        for node in self.dep_graph:
            if not set(self.dep_graph[node]).isdisjoint(kerns):
                self.dep_graph[node][:] = [dep for dep in self.dep_graph[node] if dep not in kerns]
                self.dep_graph[node].append(group_node)
        # Iterate over all MPI requests and do the same
        for deps in self.mpi_req_deps:
            if not set(deps).isdisjoint(kerns):
                deps[:] = [dep for dep in deps if dep not in kerns]
                deps.append(group_node)
        self.dep_graph[group_node] = list(group_deps)
            

    def commit(self):
        super().commit()

        # Do topological sort on kernels/groups to get run order
        ts = TopologicalSorter(self.dep_graph)
        self.run_order = tuple(ts.static_order())

        # Get MPI request idxs
        self.mpi_idxs = defaultdict(list)
        for req, deps in zip(self.mpi_reqs, self.mpi_req_deps):
            if deps:
                ix = max([i + 1 for i, k in enumerate(self.run_order) if k in deps])
                self.mpi_idxs[ix].append(req)

        # Group kernels in runs separated by MPI requests
        self._runlist, i = [], 0

        for j in sorted(self.mpi_idxs):
            krunargs = self._make_runlist(i, j)
            self._runlist.append((krunargs, self.mpi_idxs[j]))
            i = j

        if self.run_order[i:]:
            krunargs = self._make_runlist(i, len(self.run_order))
            self._runlist.append((krunargs, []))

    def run(self):
        # Start all dependency-free MPI requests
        self._startall(self.mpi_root_reqs)

        for krunargs, reqs in self._runlist:
            if krunargs:
                self.backend.krunner(len(krunargs), krunargs)

            self._startall(reqs)

        # Wait for all of the MPI requests to finish
        self._waitall(self.mpi_reqs)
