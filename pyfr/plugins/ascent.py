from collections import defaultdict
from ctypes import RTLD_GLOBAL, c_char_p, c_double, c_int, c_int64, c_void_p
import re

import numpy as np

from pyfr.ctypesutil import LibWrapper
from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.base import BaseSolnPlugin, region_data
from pyfr.shapes import BaseShape
from pyfr.util import file_path_gen, subclass_where
from pyfr.writers.vtk import BaseShapeSubDiv


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


class AscentPlugin(BaseSolnPlugin):
    name = 'ascent'
    systems = ['*']
    formulations = ['dual', 'std']

    # Element name mapping for conduit
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    # Ascent filters that implicitly require velocity expression
    v_filter = ['qcriterion', 'vorticity']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Get datatype
        self.dtype = intg.system.backend.fpdtype

        # Underlying system elements class
        self.elementscls = intg.system.elementscls

        # Set order for division
        sorder = self.cfg.getint('solver', 'order')
        dorder = self.cfg.getint(cfgsect, 'division', sorder)
        divisors = defaultdict(lambda: dorder, pyr=0)

        # Load conduit library
        self.conduit = ConduitWrappers()

        # Setup outputting options
        self.basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        self.isrestart = intg.isrestart
        self._image_paths = []

        # Expressions to plot and configs
        self._exprs = []
        self._fields_write = set()
        self._fields_read = set()
        self._init_fields(cfgsect)
        self._init_scenes(cfgsect)
        self._init_pipelines(cfgsect)

        if not self._fields_read.issubset(self._fields_write):
            raise AscentError('Not all fields used are defined')

        # Gradient pre-processing
        self._init_gradients()

        # Generate conduit blueprint of mesh
        self.mesh_n = ConduitNode(self.conduit)

        # Parse the region
        ridxs = region_data(self.cfg, cfgsect, intg.system.mesh, intg.rallocs)

        # Build the conduit blueprint mesh for the regions
        self._ele_regions_lin = []
        for etype, eidxs in ridxs.items():
            eoff = intg.system.ele_types.index(etype)

            # Build the conduit blueprint mesh for the regions
            self._build_blueprint(intg, eoff, etype, eidxs, divisors)

        # Initalise ascent and the open an instance
        self._init_ascent()

    def __del__(self):
        if getattr(self, 'ascent_ptr', None):
            self.lib.ascent_close(self.ascent_ptr)

    def _build_blueprint(self, intg, idx, ename, rgn, divisors):
        mesh_n = self.mesh_n
        d_str = f'domain_{intg.rallocs.prank}_{ename}'

        eles = intg.system.ele_map[ename]
        shapecls = subclass_where(BaseShape, name=ename)
        shape = shapecls(eles.nspts, self.cfg)

        svpts = shape.std_ele(divisors[ename])
        nsvpts = len(svpts)

        soln_op = shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)
        self._ele_regions_lin.append((d_str, idx, rgn, soln_op))

        mesh_n[f'{d_str}/state/domain_id'] = intg.rallocs.prank
        mesh_n[f'{d_str}/coordsets/coords/type'] = 'explicit'
        mesh_n[f'{d_str}/topologies/mesh/coordset'] = 'coords'
        mesh_n[f'{d_str}/topologies/mesh/type'] = 'unstructured'
        mesh_n[f'{d_str}/topologies/mesh/elements/shape'] = self.bp_emap[ename]

        xd = eles.ploc_at_np(svpts)[..., rgn].transpose(1, 2, 0)
        for l, x in zip('xyz', xd.reshape(eles.ndims, -1)):
            mesh_n[f'{d_str}/coordsets/coords/values/{l}'] = x

        subdvcls = subclass_where(BaseShapeSubDiv, name=ename)
        sconn = subdvcls.subnodes(divisors[ename])
        subdvcon = np.hstack([sconn + j*nsvpts for j in range(xd.shape[1])])
        mesh_n[f'{d_str}/topologies/mesh/elements/connectivity'] = subdvcon

        for field, path, expr in self._exprs:
            mesh_n[f'{d_str}/fields/{field}/association'] = 'vertex'
            mesh_n[f'{d_str}/fields/{field}/volume_dependent'] = 'false'
            mesh_n[f'{d_str}/fields/{field}/topology'] = 'mesh'

    def _eval_exprs(self, intg):
        # Get the primitive variable names
        pnames = self.elementscls.privarmap[self.ndims]

        # Compute the gradients
        if self._gradpinfo:
            grad_soln = intg.grad_soln

        # Iterate over each element type in our region
        for d_str, idx, rgn, soln_op in self._ele_regions_lin:
            # Subset and transpose the solution
            soln = intg.soln[idx][..., rgn].swapaxes(0, 1)

            # Interpolate the solution to the subdivided points
            soln = soln_op @ soln

            # Convert from conservative to primitive variables
            psolns = self.elementscls.con_to_pri(soln, self.cfg)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns), t=intg.tcurr)

            # Prepare any required gradients
            if self._gradpinfo:
                grads = np.moveaxis(grad_soln[idx], 2, 0)[..., rgn]

                # Interpolate the gradients to the points
                grad_soln = soln_op @ grad_soln

                # Transform from conservative to primitive gradients
                pgrads = self.elementscls.grad_con_to_pri(soln, grads,
                                                          self.cfg)

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

    def _init_ascent(self):
        comm, rank, root = get_comm_rank_root()

        self.lib = lib = AscentWrappers()
        self.ascent_ptr = lib.ascent_create(None)

        self.ascent_config = ConduitNode(self.conduit)
        self.ascent_config['mpi_comm'] = comm.py2f()
        self.ascent_config['runtime/type'] = 'ascent'
        vtkm_backend = self.cfg.get(self.cfgsect, 'vtkm-backend', 'serial')
        self.ascent_config['runtine/vtkm/backend'] = vtkm_backend

        lib.ascent_open(self.ascent_ptr, self.ascent_config)

        # Pre configure scenes and pipelines
        self.actions = ConduitNode(self.conduit)
        self._add_scene = self.actions.append()
        self._add_scene['action'] = 'add_scenes'
        self._add_scene['scenes'] = self.scenes

        self._add_pipeline = self.actions.append()
        self._add_pipeline['action'] = 'add_pipelines'
        self._add_pipeline['pipelines'] = self.pipelines

    def _init_fields(self, cfgsect):
        cons = self.cfg.items_as('constants', float)

        for k in self.cfg.items(cfgsect, prefix='field-'):
            field = k.removeprefix('field-')

            if field in self._fields_write:
                raise KeyError(f"Field '{field}' already exists")
            self._fields_write.add(field)

            exprs = self.cfg.getexpr(cfgsect, k, subs=cons)
            self._exprs.append((field, f'fields/{field}/values', exprs))

    def _init_gradients(self):
        # Determine what gradients, if any, are required
        g_pnames = set()
        for field, path, expr in self._exprs:
            g_pnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', expr))

        privarmap = self.elementscls.privarmap[self.ndims]
        self._gradpinfo = [(pname, privarmap.index(pname))
                           for pname in g_pnames]

    def _init_pipelines(self, cfgsect):
        pipel_cfg = {}
        self.pipelines = pl = ConduitNode(self.conduit)

        for k in self.cfg.items(cfgsect, prefix='pipeline-'):
            pn = k.removeprefix('pipeline-')
            cfg = self.cfg.getliteral(cfgsect, k)
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

    def _init_scenes(self, cfgsect):
        self._scene_cfg = scene_cfg = {}
        self.scenes = ConduitNode(self.conduit)

        for k in self.cfg.items(cfgsect, prefix='scene-'):
            sn = k.removeprefix('scene-')
            scene_cfg[sn] = cfg = self.cfg.getliteral(cfgsect, k)

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

    def _render_options(self, path, opts):
        for k, v in opts.items():
            if k != 'image-name':
                # Replace for vectors and sections
                key = k.replace('_', '/').replace('-', '_')
                self.scenes[f'{path}/{key}'] = v

        gen = file_path_gen(self.basedir, opts['image-name'], self.isrestart)
        self._image_paths.append((f'scenes/{path}/image_name', gen))

    def __call__(self, intg):
        if intg.nacptsteps % self.nsteps == 0:
            comm, rank, root = get_comm_rank_root()

            # Set file names
            for path, gen in self._image_paths:
                self._add_scene[path] = gen.send(intg.tcurr)

            # Set field expressions
            self._eval_exprs(intg)

            self.lib.ascent_publish(self.ascent_ptr, self.mesh_n)
            self.lib.ascent_execute(self.ascent_ptr, self.actions)

            comm.barrier()
