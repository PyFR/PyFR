from functools import cached_property

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import init_mpi
from pyfr.plugins.base import BaseCLIPlugin
from pyfr.plugins.common import cli_external
from pyfr.plugins.soln.ascent import AscentRenderer, face_shape_ops, pri_block
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.stats import tavg_exprs
from pyfr.util import subclass_where


class _CLIAdapter:
    def __init__(self, mesh, soln, acfg, cfgsect):
        from pyfr.solvers.base import BaseSystem

        self.mesh = mesh
        self._soln = soln
        self.scfg = soln.config
        self.acfg = acfg
        self.cfgsect = cfgsect
        self.dtype = np.float32

        # Overlay the ascent config onto the solution config for postproc
        self.ppcfg = ppcfg = Inifile(self.scfg.tostr())
        for sect in acfg.sections():
            for k, v in acfg.items(sect).items():
                ppcfg.set(sect, k, v)

        sname = self.scfg.get('solver', 'system')
        self.elementscls = subclass_where(BaseSystem, name=sname).elementscls

    @property
    def tcurr(self):
        return self._soln.stats.getfloat('solver-time-integrator', 'tcurr')

    @property
    def cycle(self):
        stats = self._soln.stats
        return stats.getint('solver-time-integrator', 'nacptsteps', 0)

    @cached_property
    def is_tavg(self):
        return self._soln.stats.get('data', 'prefix') == 'tavg'

    @cached_property
    def _tavg_indices(self):
        # Primitive (or grad_X_Y) name -> index in soln.fields for tavg
        section = self._soln.stats.get('tavg', 'cfg-section')
        te = tavg_exprs(self._soln.config, section, self.mesh.ndims,
                        self.elementscls)

        fields = self._soln.fields
        mapping = {}
        for name, expr in te.avgs.items():
            if (sym := expr.strip('() \t')).isidentifier():
                mapping[sym] = fields.index(f'avg-{name}')

        # Deduplicated moments resolve through their alias
        for cname, sname in te.aliases.items():
            mapping[cname] = fields.index(f'avg-{sname}')

        return mapping

    @property
    def has_grads(self):
        if self.is_tavg:
            return all(f'grad_{v}_{d}' in self._tavg_indices
                       for v in 'uvw'[:self.mesh.ndims]
                       for d in 'xyz'[:self.mesh.ndims])
        else:
            return 'grad' in self._soln.blocks

    def block(self, etype, eidxs, grads):
        data, nd = self._soln.data[etype], self.mesh.ndims

        # Tavg grads live in _soln.data and are pulled via _tavg_indices
        if self.is_tavg:
            names, rows = self._tavg_rows()
            return names, data[:, rows][..., eidxs].swapaxes(0, 1)
        else:
            nv = len(self._soln.fields)
            nb = nv*(nd + 1) if grads else nv
            # Promote to double precision ahead of the conversion
            block = data[:, :nb][..., eidxs].swapaxes(0, 1).astype(np.float64)

            return pri_block(self.elementscls, self.scfg, nd, block, grads)

    def _tavg_rows(self):
        privars = self.elementscls.privars(self.mesh.ndims, self.scfg)
        idx = self._tavg_indices

        if missing := [pn for pn in privars if pn not in idx]:
            raise KeyError(f'Tavg missing primitives: {missing}')

        return list(idx), list(idx.values())

    def soln_op_vpts(self, etype, divisor):
        meshf = self.mesh.spts[etype]

        shapecls = subclass_where(BaseShape, name=etype)
        shape = shapecls(len(meshf), self.scfg)

        svpts = shapecls.std_ele(divisor)
        mesh_op = shape.sbasis.nodal_basis_at(svpts)
        soln_op = shape.ubasis.nodal_basis_at(svpts)

        vpts = mesh_op @ meshf.reshape(len(meshf), -1)
        vpts = vpts.reshape(-1, *meshf.shape[1:])

        return soln_op, vpts.swapaxes(1, 2)

    def face_soln_op_vpts(self, etype, fidx, divisor):
        nspts = len(self.mesh.spts[etype])
        return face_shape_ops(etype, fidx, divisor, nspts, self.scfg)


class AscentCLIPlugin(BaseCLIPlugin):
    name = 'ascent'

    @classmethod
    def add_cli(cls, parser):
        sp = parser.add_subparsers()

        # Render command
        ap_render = sp.add_parser('render', help='ascent render --help')
        ap_render.set_defaults(process=cls.render_cli)
        ap_render.add_argument('mesh', help='mesh file')
        ap_render.add_argument('solns', nargs='*', help='solution files')
        ap_render.add_argument('cfg', help='ascent config file')
        ap_render.add_argument('--cfgsect', help='ascent config file section')

    @cli_external
    def render_cli(self, args):
        # Initialise MPI
        init_mpi()

        reader = NativeReader(args.mesh)
        acfg = Inifile.load(args.cfg)

        # Default to the first section with any scenes defined
        if (acfgsect := args.cfgsect) is None:
            for s in acfg.sections():
                if acfg.items(s, prefix='scene-'):
                    acfgsect = s
                    break
            else:
                raise ValueError('No section with scenes found; use '
                                 '--cfgsect')

        # Current Ascent render and associated config
        renderer, rcfg = None, None

        # Iterate over the solutions
        for s in args.solns:
            # Open the solution and create an Ascent adapter
            mesh, soln = reader.load_subset_mesh_soln(s)
            adapter = _CLIAdapter(mesh, soln, acfg, acfgsect)

            # See if we need to create a new Ascent renderer
            if not renderer or rcfg != soln.config:
                renderer = AscentRenderer(adapter, isrestart=True)
                rcfg = soln.config

            # Perform the rendering
            renderer.render(adapter)
