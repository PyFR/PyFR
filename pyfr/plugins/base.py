# -*- coding: utf-8 -*-

import os
import shlex

from pytools import prefork

from pyfr.mpiutil import get_comm_rank_root


def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if os.path.getsize(fname) == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf


class BasePlugin(object):
    name = None
    systems = None
    formulations = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars

        # Tolerance for time comparisons
        self.tol = 5*intg.dtmin

        # Initalise our post-action (if any)
        self.postact = None
        self.postactaid = None
        self.postactmode = None

        if self.cfg.hasopt(cfgsect, 'post-action'):
            self.postact = self.cfg.getpath(cfgsect, 'post-action')
            self.postactmode = self.cfg.get(cfgsect, 'post-action-mode',
                                            'blocking')

            if self.postactmode not in {'blocking', 'non-blocking'}:
                raise ValueError('Invalid post action mode')

        # Check that we support this particular system
        if not ('*' in self.systems or intg.system.name in self.systems):
            raise RuntimeError('System {0} not supported by plugin {1}'
                               .format(intg.system.name, self.name))

        # Check that we support this particular integrator formulation
        if intg.formulation not in self.formulations:
            raise RuntimeError('Formulation {0} not supported by plugin {1}'
                               .format(intg.formulation, self.name))

    def __del__(self):
        if self.postactaid is not None:
            prefork.wait(self.postactaid)

    def _invoke_postaction(self, **kwargs):
        comm, rank, root = get_comm_rank_root()

        # If we have a post-action and are the root rank then fire it
        if rank == root and self.postact:
            # If a post-action is currently running then wait for it
            if self.postactaid is not None:
                prefork.wait(self.postactaid)

            # Prepare the command line
            cmdline = shlex.split(self.postact.format(**kwargs))

            # Invoke
            if self.postactmode == 'blocking':
                prefork.call(cmdline)
            else:
                self.postactaid = prefork.call_async(cmdline)

    def __call__(self, intg):
        pass
