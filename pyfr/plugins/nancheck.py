import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, WriterMixin, region_data
from pyfr.writers.native import NativeWriter


class NaNCheckPlugin(WriterMixin, BaseSolnPlugin):
    name = 'nancheck'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.dump = self.cfg.getbool(cfgsect, 'region-dump', False)
        if self.dump:
            self.rgn_len = self.cfg.getfloat(cfgsect, 'region-len')

            # Base output directory and file name
            basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
            basename = self.cfg.get(self.cfgsect, 'basename')

            # Construct the solution writer
            self._writer = NativeWriter(intg, basedir, basename, 'soln')

            # Output field names
            self.fields = intg.system.elementscls.convarmap[self.ndims]

            # Output data type
            self.fpdtype = intg.backend.fpdtype

            self.centroid = c = {}
            for etype, eles in intg.system.ele_map.items():
                c[etype] = np.mean(eles.ploc_at_np('upts'), axis=0).T

        self.nsteps = self.cfg.getint(self.cfgsect, 'nsteps')

    def _nan_regions(self, intg):
        comm, rank, root = get_comm_rank_root()

        rgn_parts = []
        # For each element type, get nan region centroid and make box
        for s, etype in zip(intg.soln, self.centroid):
            i = np.unique(np.argwhere(np.isnan(s))[:,-1])

            if len(i):
                x = np.mean(self.centroid[etype][i], axis=0)
                x0, x1 = x - 0.5*self.rgn_len, x + 0.5*self.rgn_len

                def a2s(y): return np.array2string(y, separator=',')
                rgn_parts.append(f'box({a2s(x0)}, {a2s(x1)})')

        # Construct a single region for all ranks
        rgn_str = ' + '.join(r for r in rgn_parts)
        rgn_str = comm.allgather(rgn_str)
        rgn_glob = ' + '.join(r for r in rgn_str if r)

        # Get region indices
        ridxs = region_data(self.cfg, self.cfgsect, intg.system.mesh,
                            intg.rallocs, rgn_glob)

        # Generate the appropriate metadata arrays
        ele_regions, ele_region_data = [], {}
        for etype, eidxs in ridxs.items():
            doff = intg.system.ele_types.index(etype)
            ele_regions.append((doff, etype, eidxs))

            if not isinstance(eidxs, slice):
                ele_region_data[f'{etype}_idxs'] = eidxs

        return ele_regions, ele_region_data

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            comm, rank, root = get_comm_rank_root()
            isnan = np.array(any(np.isnan(np.sum(s)) for s in intg.soln))

            comm.allreduce(isnan, isnan, op=mpi.LOR)

            if self.dump and isnan:
                ele_regions, ele_region_data = self._nan_regions(intg)

                intg.collect_stats(self.stats)

                # If we are the root rank then prepare the metadata
                if rank == root:
                    metadata = dict(intg.cfgmeta,
                                    stats=self.stats.tostr(),
                                    mesh_uuid=intg.mesh_uuid)
                else:
                    metadata = None

                # Fetch and (if necessary) subset the solution
                data = dict(ele_region_data)
                for idx, etype, rgn in ele_regions:
                    data[etype] = intg.soln[idx][..., rgn].astype(self.fpdtype)

                # Write out the file
                self._writer.write(data, intg.tcurr, metadata)

                comm.Barrier()

            if isnan and rank == root:
                raise RuntimeError(f'NaNs detected at t = {intg.tcurr}')
