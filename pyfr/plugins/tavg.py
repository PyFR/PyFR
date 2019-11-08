# -*- coding: utf-8 -*-

import re

import numpy as np

from pyfr.inifile import Inifile
from pyfr.plugins.base import BasePlugin
from pyfr.nputil import npeval
from pyfr.writers.native import NativeWriter


class TavgPlugin(BasePlugin):
    name = 'tavg'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Expressions to time average
        c = self.cfg.items_as('constants', float)
        self.exprs = [(k, self.cfg.getexpr(cfgsect, k, subs=c))
                      for k in self.cfg.items(cfgsect)
                      if k.startswith('avg-')]

        # Gradient pre-processing
        self._init_gradients(intg)

        # Save a reference to the physical solution point locations
        self.plocs = intg.system.ele_ploc_upts

        # Output file directory, base name, and writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        self._writer = NativeWriter(intg, len(self.exprs), basedir, basename,
                                    prefix='tavg')

        # Time averaging parameters
        self.dtout = self.cfg.getfloat(cfgsect, 'dt-out')
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.tout_last = intg.tcurr

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dtout)

        # Time averaging state
        self.prevt = intg.tcurr
        self.prevex = self._eval_exprs(intg)
        self.accmex = [np.zeros_like(p) for p in self.prevex]

    def _init_gradients(self, intg):
        # Determine what gradients, if any, are required
        self._gradpnames = gradpnames = set()
        for k, ex in self.exprs:
            gradpnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', ex))

        # If gradients are required then form the relevant operators
        if gradpnames:
            self._gradop, self._rcpjact = [], []

            for eles in intg.system.ele_map.values():
                self._gradop.append(eles.basis.m4)

                # Get the smats at the solution points
                smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)

                # Get |J|^-1 at the solution points
                rcpdjac = eles.rcpdjac_at_np('upts')

                # Product to give J^-T at the solution points
                self._rcpjact.append(smat*rcpdjac)

    def _eval_exprs(self, intg):
        exprs = []

        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        # Iterate over each element type in the simulation
        for i, (soln, ploc) in enumerate(zip(intg.soln, self.plocs)):
            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln.swapaxes(0, 1),
                                                 self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns))
            subs.update(zip('xyz', ploc.swapaxes(0, 1)))

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
                        subs['grad_{0}_{1}'.format(pname, dim)] = grad

            # Evaluate the expressions
            exprs.append([npeval(v, subs) for k, v in self.exprs])

        # Stack up the expressions for each element type and return
        return [np.dstack(exs).swapaxes(1, 2) for exs in exprs]

    def __call__(self, intg):
        tdiff = intg.tcurr - self.tout_last
        dowrite = tdiff >= self.dtout - self.tol
        doaccum = intg.nacptsteps % self.nsteps == 0

        if dowrite or doaccum:
            # Evaluate the time averaging expressions
            currex = self._eval_exprs(intg)

            # Accumulate them; always do this even when just writing
            for a, p, c in zip(self.accmex, self.prevex, currex):
                a += 0.5*(intg.tcurr - self.prevt)*(p + c)

            # Save the time and solution
            self.prevt = intg.tcurr
            self.prevex = currex

            if dowrite:
                # Normalise
                accmex = [a / tdiff for a in self.accmex]

                stats = Inifile()
                stats.set('data', 'prefix', 'tavg')
                stats.set('data', 'fields',
                          ','.join(k for k, v in self.exprs))
                stats.set('tavg', 'tstart', self.tout_last)
                stats.set('tavg', 'tend', intg.tcurr)
                intg.collect_stats(stats)

                metadata = dict(intg.cfgmeta,
                                stats=stats.tostr(),
                                mesh_uuid=intg.mesh_uuid)

                self._writer.write(accmex, metadata, intg.tcurr)

                self.tout_last = intg.tcurr
                self.accmex = [np.zeros_like(a) for a in accmex]
