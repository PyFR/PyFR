# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin, PostactionMixin, RegionMixin
from pyfr.nputil import npeval


class TavgPlugin(PostactionMixin, RegionMixin, BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Averaging mode
        self.mode = self.cfg.get(cfgsect, 'mode', 'windowed')
        if self.mode not in {'continuous', 'windowed'}:
            raise ValueError('Invalid averaging mode')

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Expressions pre-processing
        self._prepare_exprs()

        # Output data type
        fpdtype = self.cfg.get(cfgsect, 'precision', 'single')

        # Construct the file writer
        self._writer = self._init_writer_for_region(intg, len(self.outfields),
                                                    'tavg', fpdtype=fpdtype)
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
        self._gradpnames = gradpnames = set()
        for ex in self.aexprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        # If gradients are required then form the relevant operators
        if gradpnames:
            self._gradop, self._rcpjact = [], []

            for i, rgn in self._ele_regions:
                eles = intg.system.ele_map[intg.system.ele_types[i]]

                self._gradop.append(eles.basis.m4)

                # Get the smats at the solution points
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                # Get |J|^-1 at the solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # Product to give J^-T at the solution points
                self._rcpjact.append(smat[..., rgn]*rcpdjac[..., rgn])

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
        for i, (j, rgn) in enumerate(self._ele_regions):
            soln = intg.soln[j][..., rgn]

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln.swapaxes(0, 1),
                                                 self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))

            # Compute any required gradients
            if self._gradpnames:
                # Gradient operator and J^-T matrix
                gradop, rcpjact = self._gradop[i], self._rcpjact[i]
                nupts = gradop.shape[1]

                for pname in self._gradpnames:
                    psoln = subs[pname]

                    # Compute the transformed gradient
                    tgradpn = gradop @ psoln
                    tgradpn = tgradpn.reshape(self.ndims, nupts, -1)

                    # Untransform this to get the physical gradient
                    gradpn = np.einsum('ijkl,jkl->ikl', rcpjact, tgradpn)
                    gradpn = gradpn.reshape(self.ndims, nupts, -1)

                    for dim, grad in zip('xyz', gradpn):
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
                if self.mode == 'windowed':
                    accex = self.accex
                    tstart = self.tout_last
                else:
                    for a, c in zip(self.accex, self.caccex):
                        c += a

                    accex = self.caccex
                    tstart = self.tstart_actual

                # Normalise the accumulated expressions
                data = [a / (intg.tcurr - tstart) for a in accex]

                # Evaluate any functional expressions
                if self.fexprs:
                    funex = self._eval_fun_exprs(intg, data)
                    data = [np.hstack([a, f]) for a, f in zip(data, funex)]

                # Prepare the stats record
                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields', ','.join(self.outfields))
                stats.set('tavg', 'tstart', tstart)
                stats.set('tavg', 'tend', intg.tcurr)
                intg.collect_stats(stats)

                # Prepare the metadata
                metadata = dict(intg.cfgmeta,
                                stats=stats.tostr(),
                                mesh_uuid=intg.mesh_uuid)

                # Add in any required region data and write to disk
                data = self._add_region_data(data)
                solnfname = self._writer.write(data, metadata, intg.tcurr)

                # If a post-action has been registered then invoke it
                self._invoke_postaction(mesh=intg.system.mesh.fname,
                                        soln=solnfname, t=intg.tcurr)

                # Reset the accumulators
                for a in self.accex:
                    a.fill(0)

                self.tout_last = intg.tcurr
