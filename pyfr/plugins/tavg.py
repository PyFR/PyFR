# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class TavgPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')

        # Expressions pre-processing
        self._prepare_exprs()

        # Output data type
        fpdtype = self.cfg.get(cfgsect, 'precision', 'single')
        if fpdtype == 'single':
            self.fpdtype = np.float32
        elif fpdtype == 'double':
            self.fpdtype = np.float64
        else:
            raise ValueError('Invalid floating point data type')

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the file writer
        self._writer = NativeWriter(intg, basedir, basename, 'tavg')

        # Gradient pre-processing
        self._init_gradients(intg)

        # Time averaging parameters
        self.tstart = self.cfg.getfloat(cfgsect, 'tstart', 0.0)
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Mark ourselves as not currently averaging
        self._started = False

    def _prepare_exprs(self):
        cfg, cfgsect = self.cfg, self.cfgsect
        c = self.cfg.items_as('constants', float)
        self.anames, self.aexprs = [], []
        self.outfields, self.fexprs = [], []

        # Iterate over accumulation expressions first
        for k in cfg.items(cfgsect):
            if k.startswith('avg-'):
                self.anames.append(k[4:])
                self.aexprs.append(cfg.getexpr(cfgsect, k, subs=c))
                self.outfields.append(k)

        # Followed by any functional expressions
        for k in cfg.items(cfgsect):
            if k.startswith('fun-avg-'):
                self.fexprs.append(cfg.getexpr(cfgsect, k, subs=c))
                self.outfields.append(k)

    def _init_gradients(self, intg):
        # Determine what gradients, if any, are required
        gradpnames = set()
        for ex in self.aexprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                           for pname in gradpnames]

    def _init_accumex(self, intg):
        self.prevt = self.tout_last = intg.tcurr
        self.prevex = self._eval_acc_exprs(intg)
        self.accex = [np.zeros_like(p, dtype=np.float64) for p in self.prevex]

        # Extra state for continuous accumulation
        if self.mode == 'continuous':
            self.caccex = [np.zeros_like(a) for a in self.accex]
            self.tstart_actual = intg.tcurr

    def _eval_acc_exprs(self, intg):
        exprs = []

        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        # Iterate over each element type in the simulation
        for idx, etype, rgn in self._ele_regions:
            soln = intg.soln[idx][..., rgn].swapaxes(0, 1)

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))

            # Prepare any required gradients
            if self._gradpinfo:
                # Compute the gradients
                grad_soln = np.rollaxis(intg.grad_soln[idx], 2)[..., rgn]

                # Transform from conservative to primitive gradients
                pgrads = self.elementscls.grad_con_to_pri(soln, grad_soln,
                                                          self.cfg)

                # Add them to the substitutions dictionary
                for pname, idx in self._gradpinfo:
                    for dim, grad in zip('xyz', pgrads[idx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            # Evaluate the expressions
            exprs.append([npeval(v, subs) for v in self.aexprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def _eval_fun_exprs(self, intg, accex):
        exprs = []

        # Iterate over each element type our averaging region
        for avals in accex:
            # Prepare the substitution dictionary
            subs = dict(zip(self.anames, avals.swapaxes(0, 1)))

            exprs.append([npeval(v, subs) for v in self.fexprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def __call__(self, intg):
        # If we are not supposed to be averaging yet then return
        if intg.tcurr < self.tstart:
            return

        # If necessary, run the start-up routines
        if not self._started:
            self._init_accumex(intg)
            self._started = True

        # See if we are due to write and/or accumulate this step
        dowrite = intg.tcurr - self.tout_last >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
            currex = self._eval_acc_exprs(intg)

            # Accumulate them; always do this even when just writing
            for a, p, c in zip(self.accex, self.prevex, currex):
                a += 0.5*(intg.tcurr - self.prevt)*(p + c)

            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            if dowrite:
                comm, rank, root = get_comm_rank_root()

                if self.mode == 'windowed':
                    accex = self.accex
                    tstart = self.tout_last
                else:
                    for a, c in zip(self.accex, self.caccex):
                        c += a

                    accex = self.caccex
                    tstart = self.tstart_actual

                # Normalise the accumulated expressions
                tavg = [a / (intg.tcurr - tstart) for a in accex]

                # Evaluate any functional expressions
                if self.fexprs:
                    funex = self._eval_fun_exprs(intg, tavg)
                    tavg = [np.hstack([a, f]) for a, f in zip(tavg, funex)]

                # Form the output records to be written to disk
                data = dict(self._ele_region_data)
                for (idx, etype, rgn), d in zip(self._ele_regions, tavg):
                    data[etype] = d.astype(self.fpdtype)

                # If we are the root rank then prepare the metadata
                if rank == root:
                    stats = Inifile()
                    stats.set('data', 'prefix', 'tavg')
                    stats.set('data', 'fields', ','.join(self.outfields))
                    stats.set('tavg', 'tstart', tstart)
                    stats.set('tavg', 'tend', intg.tcurr)
                    intg.collect_stats(stats)

                    metadata = dict(intg.cfgmeta,
                                    stats=stats.tostr(),
                                    mesh_uuid=intg.mesh_uuid)
                else:
                    metadata = None

                # Write to disk
                solnfname = self._writer.write(data, intg.tcurr, metadata)

                # If a post-action has been registered then invoke it
                self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                        soln=solnfname, t=intg.tcurr)

                # Reset the accumulators
                for a in self.accex:
                    a.fill(0)

                self.tout_last = intg.tcurr
