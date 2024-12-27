from argparse import FileType
from ctypes import (RTLD_GLOBAL, c_char_p, c_double, c_int, c_int32, c_int64,
                    c_void_p)
import re

import numpy as np

from pyfr.ctypesutil import LibWrapper
from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.nputil import npeval
from pyfr.plugins.base import (BaseCLIPlugin, BaseSolnPlugin, cli_external,
                               region_data)
from pyfr.readers.native import NativeReader
from pyfr.shapes import BaseShape
from pyfr.util import file_path_gen, subclass_where
from pyfr.writers.vtk import get_subdiv


class AscentError(Exception): pass
class ConduitError(Exception): pass


class ConduitWrappers(LibWrapper):
    _libname = 'conduit'
    _errtype = c_void_p
    _mode = RTLD_GLOBAL

    # Functions
    _functions = [
        (c_int, 'conduit_datatype_sizeof_index_t'),
        (c_void_p, 'conduit_node_append', c_void_p),
        (c_void_p, 'conduit_node_create', c_void_p),
        (None, 'conduit_node_destroy', c_void_p),
        (None, 'conduit_node_set_path_char8_str', c_void_p, c_char_p,
         c_char_p),
        (None, 'conduit_node_set_path_float32_ptr', c_void_p, c_char_p,
         c_void_p, c_int64),
        (None, 'conduit_node_set_path_float64', c_void_p, c_char_p,
         c_double),
        (None, 'conduit_node_set_path_float64_ptr', c_void_p, c_char_p,
         c_void_p, c_int64),
        (None, 'conduit_node_set_path_int32', c_void_p, c_char_p, c_int32),
        (None, 'conduit_node_set_path_int64', c_void_p, c_char_p, c_int64),
        (None, 'conduit_node_set_path_int64_ptr', c_void_p, c_char_p,
         c_void_p, c_int64),
        (None, 'conduit_node_set_path_node', c_void_p, c_char_p, c_void_p)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.conduit_datatype_sizeof_index_t() != 8:
            raise RuntimeError('Conduit must be compiled with 64-bit index '
                               'types')

    def _errcheck(self, status, fn, args):
        if not status:
            raise ConduitError

        return status


class ConduitNode:
    def __init__(self, lib, ptr=None, child=False):
        self.lib = lib
        self.child = child
        self._as_parameter_ = ptr or self.lib.conduit_node_create(None)

    def __del__(self):
        if not self.child:
            self.lib.conduit_node_destroy(self)

    def __setitem__(self, key, value):
        key = key.encode()
        match value:
            case str():
                self.lib.conduit_node_set_path_char8_str(self, key,
                                                         value.encode())
            case ConduitNode():
                self.lib.conduit_node_set_path_node(self, key, value)
            case int():
                if value.bit_length() <= 32:
                    self.lib.conduit_node_set_path_int32(self, key, value)
                else:
                    self.lib.conduit_node_set_path_int64(self, key, value)
            case float():
                self.lib.conduit_node_set_path_float64(self, key, value)
            case (np.ndarray() | np.generic()):
                value = np.ascontiguousarray(value)
                fn = getattr(self.lib,
                             f'conduit_node_set_path_{value.dtype}_ptr')
                fn(self, key, value.ctypes.data, value.size)
            case list():
                value = np.array(value, dtype=float)
                self.lib.conduit_node_set_path_float64_ptr(self, key,
                                                           value.ctypes.data,
                                                           value.size)
            case _:
                ValueError('ConduitNode: __setitem__ type not supported')

    def append(self):
        ptr = self.lib.conduit_node_append(self)
        return ConduitNode(self.lib, ptr, child=True)


class AscentWrappers(LibWrapper):
    _libname = 'ascent_mpi'

    # Functions
    _functions = [
        (None, 'ascent_close', c_void_p),
        (c_void_p, 'ascent_create', c_void_p),
        (None, 'ascent_execute', c_void_p, c_void_p),
        (None, 'ascent_open', c_void_p, c_void_p),
        (None, 'ascent_publish', c_void_p, c_void_p)
    ]


class _IntegratorAdapter:
    def __init__(self, intg, cfgsect):
        self.intg = intg
        self.acfg = self.scfg = intg.cfg
        self.cfgsect = cfgsect

    @property
    def ndims(self):
        return self.intg.system.ndims

    @property
    def etypes(self):
        return self.intg.system.ele_types

    @property
    def mesh_uuid(self):
        return self.intg.mesh_uuid

    @property
    def elementscls(self):
        return self.intg.system.elementscls

    def eidx(self, etype):
        return self.intg.system.ele_types.index(etype)

    @property
    def dtype(self):
        return self.intg.system.backend.fpdtype

    @property
    def region_data(self):
        return region_data(self.acfg, self.cfgsect, self.intg.system.mesh)

    def soln_op_vpts(self, ename, divisor):
        eles = self.intg.system.ele_map[ename]
        shapecls = subclass_where(BaseShape, name=ename)
        shape = shapecls(eles.nspts, self.scfg)

        svpts = shape.std_ele(divisor)
        soln_op = shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

        return soln_op, eles.ploc_at_np(svpts)

    @property
    def tcurr(self):
        return self.intg.tcurr

    @property
    def soln(self):
        return self.intg.soln

    @property
    def grad_soln(self):
        return self.intg.grad_soln


class _CLIAdapter:
    def __init__(self, mesh, soln, acfg, acfgsect):
        self.acfg = acfg
        self.cfgsect = acfgsect

        self._mesh = mesh
        self._soln = soln

        self.scfg = soln['config']

    @property
    def ndims(self):
        return self._mesh.ndims

    @property
    def etypes(self):
        return list(self._mesh.eidxs)

    @property
    def mesh_uuid(self):
        return self._mesh.uuid

    @property
    def elementscls(self):
        from pyfr.solvers.base import BaseSystem

        sname = self.scfg.get('solver', 'system')
        return subclass_where(BaseSystem, name=sname).elementscls

    @property
    def dtype(self):
        return np.float32

    @property
    def region_data(self):
        return region_data(self.acfg, self.cfgsect, self._mesh)

    def soln_op_vpts(self, ename, divisor):
        meshf = self._mesh.spts[ename]

        shapecls = subclass_where(BaseShape, name=ename)
        shape = shapecls(len(meshf), self.scfg)

        svpts = shapecls.std_ele(divisor)
        mesh_op = shape.sbasis.nodal_basis_at(svpts)
        soln_op = shape.ubasis.nodal_basis_at(svpts)

        vpts = mesh_op @ meshf.reshape(len(meshf), -1)
        vpts = vpts.reshape(-1, *meshf.shape[1:])

        return soln_op, vpts.swapaxes(1, 2)

    @property
    def tcurr(self):
        stats = self._soln['stats']
        return stats.getfloat('solver-time-integrator', 'tcurr')

    @property
    def soln(self):
        return [self._soln[etype] for etype in self.etypes]

    @property
    def grad_soln(self):
        raise NotImplementedError('Gradients are not supported in CLI mode')


class _AscentRenderer:
    # Element name mapping for conduit
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    # Ascent filters that implicitly require velocity expression
    v_filter = ['qcriterion', 'vorticity']

    def __init__(self, adapter, isrestart):
        # Set order for subdivision
        sorder = adapter.scfg.getint('solver', 'order')
        divisor = adapter.acfg.getint(adapter.cfgsect, 'division', sorder)

        # Load Conduit
        self.conduit = ConduitWrappers()

        # Setup outputting options
        self.basedir = adapter.acfg.getpath(adapter.cfgsect, 'basedir', '.',
                                            abs=True)
        self.isrestart = isrestart
        self._image_paths = []

        # Expressions to plot and configs
        self._exprs = []
        self._fields_write = set()
        self._fields_read = set()
        self._init_fields(adapter, adapter.cfgsect)
        self._init_scenes(adapter, adapter.cfgsect)
        self._init_pipelines(adapter, adapter.cfgsect)

        if not self._fields_read.issubset(self._fields_write):
            raise AscentError('Not all fields used are defined')

        # Gradient pre-processing
        self._init_gradients(adapter)

        # Generate a Conduit node for the mesh
        self.mesh_n = ConduitNode(self.conduit)

        # Build the Conduit blueprint mesh for the regions
        self._ele_regions_lin = []
        for etype, eidxs in adapter.region_data.items():
            # Build the conduit blueprint mesh for the regions
            self._build_blueprint(adapter, etype, eidxs, divisor)

        # Initalise Ascent and the open an instance
        self._init_ascent(adapter)

    def __del__(self):
        if getattr(self, 'ascent_ptr', None):
            self.lib.ascent_close(self.ascent_ptr)

    def _build_blueprint(self, adapter, etype, rgn, divisor):
        comm, rank, root = get_comm_rank_root()

        mesh_n = self.mesh_n
        d_str = f'domain_{rank}_{etype}'
        e_str = f'{d_str}/topologies/mesh/elements'

        eidx = adapter.etypes.index(etype)
        soln_op, xd = adapter.soln_op_vpts(etype, divisor)
        self._ele_regions_lin.append((d_str, eidx, rgn, soln_op))

        xd = xd[..., rgn].transpose(1, 2, 0)
        ndims, neles, nsvpts = xd.shape

        mesh_n[f'{d_str}/state/domain_id'] = rank
        mesh_n[f'{d_str}/state/config/keyword'] = 'Config'
        mesh_n[f'{d_str}/state/config/data'] = adapter.scfg.tostr()
        mesh_n[f'{d_str}/state/mesh_uuid/keyword'] = 'Mesh_UUID'
        mesh_n[f'{d_str}/state/mesh_uuid/data'] = adapter.mesh_uuid

        mesh_n[f'{d_str}/coordsets/coords/type'] = 'explicit'
        mesh_n[f'{d_str}/topologies/mesh/coordset'] = 'coords'
        mesh_n[f'{d_str}/topologies/mesh/type'] = 'unstructured'

        for l, x in zip('xyz', xd.reshape(adapter.ndims, -1)):
            mesh_n[f'{d_str}/coordsets/coords/values/{l}'] = x

        # Subdivide the element
        subdiv = get_subdiv(etype, divisor)
        snodes = subdiv.subnodes

        sconn = np.tile(snodes, (neles, 1))
        sconn += (np.arange(neles)*nsvpts)[:, None]
        mesh_n[f'{e_str}/connectivity'] = sconn

        # Handle elements which subdivide into more than one type of element
        if len(scells := set(subdiv.subcells)) > 1:
            mesh_n[f'{e_str}/shape'] = 'mixed'

            for sc in scells:
                an = self.bp_emap[sc]
                mesh_n[f'{e_str}/shape_map/{an}'] = subdiv.vtk_types[sc]

            scell_t = subdiv.subcelltypes
            mesh_n[f'{e_str}/shapes'] = np.tile(scell_t, neles)

            scell_s = subdiv.subcells
            scell_s = [subdiv.vtk_nodes[sc] for sc in scell_s]
            mesh_n[f'{e_str}/sizes'] = np.tile(scell_s, neles)

            scell_o = np.tile(subdiv.subcelloffs, (neles, 1))
            scell_o += (np.arange(neles)*len(snodes))[:, None]
            scell_o = np.concatenate(([0], scell_o.flat[:-1]))
            mesh_n[f'{e_str}/offsets'] = scell_o
        else:
            mesh_n[f'{e_str}/shape'] = self.bp_emap[etype]

        for field, path, expr in self._exprs:
            mesh_n[f'{d_str}/fields/{field}/association'] = 'vertex'
            mesh_n[f'{d_str}/fields/{field}/volume_dependent'] = 'false'
            mesh_n[f'{d_str}/fields/{field}/topology'] = 'mesh'

    def _init_ascent(self, adapter):
        comm, rank, root = get_comm_rank_root()

        self.lib = lib = AscentWrappers()
        self.ascent_ptr = lib.ascent_create(None)

        self.ascent_config = ConduitNode(self.conduit)
        self.ascent_config['mpi_comm'] = comm.py2f()
        self.ascent_config['runtime/type'] = 'ascent'
        vtkm_backend = adapter.acfg.get(adapter.cfgsect, 'vtkm-backend',
                                        'serial')
        self.ascent_config['runtime/vtkm/backend'] = vtkm_backend

        lib.ascent_open(self.ascent_ptr, self.ascent_config)

        # Pre configure scenes and pipelines
        self.actions = ConduitNode(self.conduit)
        self._add_scene = self.actions.append()
        self._add_scene['action'] = 'add_scenes'
        self._add_scene['scenes'] = self.scenes

        self._add_pipeline = self.actions.append()
        self._add_pipeline['action'] = 'add_pipelines'
        self._add_pipeline['pipelines'] = self.pipelines

    def _init_fields(self, adapter, cfgsect):
        cons = adapter.scfg.items_as('constants', float)

        for k in adapter.acfg.items(cfgsect, prefix='field-'):
            field = k.removeprefix('field-')

            if field in self._fields_write:
                raise KeyError(f"Field '{field}' already exists")
            self._fields_write.add(field)

            exprs = adapter.acfg.getexpr(cfgsect, k, subs=cons)
            self._exprs.append((field, f'fields/{field}/values', exprs))

    def _init_gradients(self, adapter):
        # Determine what gradients, if any, are required
        g_pnames = set()
        for field, path, expr in self._exprs:
            g_pnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', expr))

        privars = adapter.elementscls.privars(adapter.ndims, adapter.scfg)
        self._gradpinfo = [(pname, privars.index(pname)) for pname in g_pnames]

    def _init_pipelines(self, adapter, cfgsect):
        pipel_cfg = {}
        self.pipelines = pl = ConduitNode(self.conduit)

        for k in adapter.acfg.items(cfgsect, prefix='pipeline-'):
            pn = k.removeprefix('pipeline-')
            cfg = adapter.acfg.getliteral(cfgsect, k)
            pipel_cfg[pn] = cfg = [cfg] if isinstance(cfg, dict) else cfg

            for j, filter in enumerate(cfg):
                params = ConduitNode(self.conduit)

                pl[f'pl_{pn}/f{j}/type'] = ptype = filter.pop('type')
                for kf, vf in filter.items():
                    if kf == 'output-name':
                        if vf in self._fields_write:
                            raise KeyError(f"Output name '{vf}' already used")
                        self._fields_write.add(vf)
                    elif kf == 'field':
                        self._fields_read.add(vf)

                    params[kf.replace('_', '/').replace('-', '_')] = vf

                if ptype in self.v_filter:
                    self._fields_read.add('velocity')

                pl[f'pl_{pn}/f{j}/params'] = params

    def _init_scenes(self, adapter, cfgsect):
        self._scene_cfg = scene_cfg = {}
        self.scenes = ConduitNode(self.conduit)

        for k in adapter.acfg.items(cfgsect, prefix='scene-'):
            sn = k.removeprefix('scene-')
            scene_cfg[sn] = cfg = adapter.acfg.getliteral(cfgsect, k)

            if cfg['type'] != 'mesh':
                self._fields_read.add(cfg['field'])

            for kc, vc in cfg.items():
                if kc == 'pipeline':
                    self.scenes[f's_{sn}/plots/p1/pipeline'] = f'pl_{vc}'
                elif kc.startswith('render-'):
                    rname = kc.removeprefix('render-')
                    self._render_options(f's_{sn}/renders/r_{rname}', vc)
                else:
                    key = kc.replace('_', '/').replace('-', '_')
                    self.scenes[f's_{sn}/plots/p1/{key}'] = vc

            # If no render options then throw error
            if not any(kc.startswith('render-') for kc in cfg):
                raise KeyError(f"No render config given for scene '{sn}'")

    def _eval_exprs(self, adapter):
        elementscls = adapter.elementscls

        # Get the primitive variable names
        pnames = elementscls.privars(adapter.ndims, adapter.scfg)

        # Obtain the solution
        soln = adapter.soln

        # Compute the gradients
        if self._gradpinfo:
            grad_soln = adapter.grad_soln

        # Iterate over each element type in our region
        for d_str, idx, rgn, soln_op in self._ele_regions_lin:
            self.mesh_n[f'{d_str}/state/time/keyword'] = 'Time'
            self.mesh_n[f'{d_str}/state/time/data'] = str(adapter.tcurr)

            # Subset and transpose the solution
            csolns = soln[idx][..., rgn].swapaxes(0, 1)

            # Interpolate the solution to the subdivided points
            csolns = soln_op @ csolns

            # Convert from conservative to primitive variables
            psolns = elementscls.con_to_pri(csolns, adapter.scfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns), t=adapter.tcurr)

            # Prepare any required gradients
            if self._gradpinfo:
                grads = np.moveaxis(grad_soln[idx], 2, 0)[..., rgn]

                # Interpolate the gradients to the points
                grad_soln = soln_op @ grad_soln

                # Transform from conservative to primitive gradients
                pgrads = elementscls.grad_con_to_pri(soln, grads, adapter.scfg)

                # Add them to the substitutions dictionary
                for pname, idx in self._gradpinfo:
                    for dim, grad in zip('xyz', pgrads[idx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            for field, path, expr in self._exprs:
                fun = npeval(expr, subs)
                if isinstance(fun, tuple):
                    for f, x in zip(fun, 'xyz'):
                        self.mesh_n[f'{d_str}/{path}/{x}'] = f.T
                else:
                    self.mesh_n[f'{d_str}/{path}'] = fun.T

    def _render_options(self, path, opts):
        for k, v in opts.items():
            if k != 'image-name':
                # Replace - with _ for vectors and sections
                key = k.replace('_', '/').replace('-', '_')
                self.scenes[f'{path}/{key}'] = v

        gen = file_path_gen(self.basedir, opts['image-name'], self.isrestart)
        self._image_paths.append((f'scenes/{path}/image_name', gen))

    def render(self, adapter):
        comm, rank, root = get_comm_rank_root()

        # Set file names
        for path, gen in self._image_paths:
            self._add_scene[path] = gen.send(adapter.tcurr)

        # Set field expressions
        self._eval_exprs(adapter)

        self.lib.ascent_publish(self.ascent_ptr, self.mesh_n)
        self.lib.ascent_execute(self.ascent_ptr, self.actions)

        comm.barrier()


class AscentPlugin(BaseSolnPlugin):
    name = 'ascent'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self._renderer = _AscentRenderer(_IntegratorAdapter(intg, cfgsect),
                                         intg.isrestart)

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            self._renderer.render(_IntegratorAdapter(intg, self.cfgsect))


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
        ap_render.add_argument('cfg', type=FileType('r'),
                               help='ascent config file')
        ap_render.add_argument('--cfgsect', help='ascent config file section')

    @cli_external
    def render_cli(self, args):
        # Initialise MPI
        init_mpi()

        reader = NativeReader(args.mesh, construct_con=False)
        acfg = Inifile.load(args.cfg)
        acfgsect = args.cfgsect or acfg.sections()[0]

        # Current Ascent render and associated config
        renderer, rcfg = None, None

        # Iterate over the solutions
        for s in args.solns:
            # Open the solution and create an Ascent adapter
            mesh, soln = reader.load_subset_mesh_soln(s, prefix='soln')
            adapter = _CLIAdapter(mesh, soln, acfg, acfgsect)

            # See if we need to create a new Ascent renderer
            if not renderer or rcfg != soln['config']:
                renderer = _AscentRenderer(adapter, isrestart=True)
                rcfg = soln['config']

            # Perform the rendering
            renderer.render(adapter)
