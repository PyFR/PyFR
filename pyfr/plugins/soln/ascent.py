from collections import defaultdict
from ctypes import (RTLD_GLOBAL, c_char_p, c_double, c_int, c_int32, c_int64,
                    c_void_p)
import re

import numpy as np

from pyfr.ctypesutil import LibWrapper
from pyfr.exprs import npeval
from pyfr.fields import CleanToGrid
from pyfr.inifile import process_expr
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.common import region_data
from pyfr.plugins.postproc.adapters import (BoundaryPostProcData, FaceInfo,
                                            PostProcData)
from pyfr.plugins.postproc import get_source
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.shapes import BaseShape, proj_pts
from pyfr.subdiv import get_subdiv
from pyfr.util import (file_path_gen, first, paren_depths,
                       subclass_where)


def con_psolns_pgrads(elementscls, scfg, csolns, cgrads):
    psolns = elementscls.con_to_pri(csolns, scfg)

    if cgrads is None:
        return psolns, None
    else:
        return psolns, elementscls.grad_con_to_pri(csolns, cgrads, scfg)


def face_shape_ops(etype, fidx, divisor, nspts, scfg):
    # Interp ops for sub-divided sample pts on face fidx of etype
    shapecls = subclass_where(BaseShape, name=etype)
    itype, proj, _ = shapecls.faces[fidx]
    ishapecls = subclass_where(BaseShape, name=itype)
    fsvpts = proj_pts(proj, ishapecls.std_ele(divisor))

    shape = shapecls(nspts, scfg)
    mesh_op = shape.sbasis.nodal_basis_at(fsvpts)
    soln_op = shape.ubasis.nodal_basis_at(fsvpts)

    return itype, mesh_op, soln_op, fsvpts


def split_components(expr):
    # Split on top-level commas; respect bracket/paren/brace nesting
    parts, buf = [], []
    for c, d in paren_depths(expr):
        if c == ',' and d == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(c)

    parts.append(''.join(buf).strip())

    return parts


def _build_cleaner(mesh, divisor, cnodemap, svptsmap):
    divmap = dict.fromkeys(cnodemap, divisor)
    shared = np.fromiter(mesh.shared_nodes.by_node, dtype=int)
    return CleanToGrid(cnodemap, divmap, svptsmap, shared)


def _bp_key(k):
    return k.replace('_', '/').replace('-', '_')


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
                raise ValueError('ConduitNode: __setitem__ type not supported')

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
    has_grads = True

    def __init__(self, intg, acfg, cfgsect):
        self.intg = intg
        self.mesh = intg.system.mesh
        self.scfg = intg.cfg
        self.acfg = acfg
        self.cfgsect = cfgsect
        self.dtype = intg.system.backend.fpdtype
        self.elementscls = intg.system.elementscls

    @property
    def tcurr(self):
        return self.intg.tcurr

    @property
    def cycle(self):
        return self.intg.nacptsteps

    @property
    def soln(self):
        return dict(zip(self.intg.system.ele_types, self.intg.soln))

    @property
    def grad_soln(self):
        return dict(zip(self.intg.system.ele_types, self.intg.grad_soln))

    def psolns_pgrads(self, csolns, cgrads):
        return con_psolns_pgrads(self.elementscls, self.scfg, csolns, cgrads)

    def soln_op_vpts(self, etype, divisor):
        eles = self.intg.system.ele_map[etype]
        shapecls = subclass_where(BaseShape, name=etype)
        shape = shapecls(eles.nspts, self.scfg)

        svpts = shape.std_ele(divisor)
        soln_op = shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)

        return soln_op, eles.ploc_at_np(svpts)

    def face_soln_op_vpts(self, etype, fidx, divisor):
        nspts = self.intg.system.ele_map[etype].nspts
        ops = face_shape_ops(etype, fidx, divisor, nspts, self.scfg)
        itype, mop, sop, svpts = ops
        return itype, mop, sop.astype(self.dtype), svpts


class _VolumeAscentOutput:
    kind = 'volume'

    def __init__(self, renderer, sname='volume', clean=False):
        self.renderer = renderer
        self.sname = sname
        self.rdata = renderer.rdata
        self.soln_ops = {}
        self._setup_clean(clean)

    def _setup_clean(self, clean):
        if clean:
            self.cleaner = self._make_cleaner()
            self.publish_fields = self._publish_clean
        else:
            self.cleaner = None
            self.publish_fields = self._publish_direct

    def _make_cleaner(self):
        mesh, divisor = self.renderer.mesh, self.renderer.divisor
        cnodemap, svptsmap = {}, {}
        for etype, eidxs in self.rdata.items():
            shapecls = subclass_where(BaseShape, name=etype)
            spts_nodes = mesh.spts_nodes[etype]
            cidxs = shapecls.corner_pts_idxs(spts_nodes.shape[1])
            cnodemap[etype] = spts_nodes[eidxs][:, cidxs]
            svptsmap[etype] = shapecls.std_ele(divisor)

        return _build_cleaner(mesh, divisor, cnodemap, svptsmap)

    def domain_keys(self):
        return list(self.rdata)

    def _points(self, key, xd):
        if self.cleaner is None:
            return xd.swapaxes(0, 1).reshape(-1, xd.shape[-1]).T
        else:
            return self.cleaner.select(key, xd).T

    def _connectivity(self, key, snodes, neles, nsvpts):
        if self.cleaner is None:
            con = np.tile(snodes, (neles, 1))
            con += (np.arange(neles)*nsvpts)[:, None]
            return con
        else:
            return self.cleaner.layouts[key][0][:, snodes]

    def build_blueprint(self, renderer, domid, etype):
        dom = f'domain_{domid}'

        soln_op, xd = renderer.adapter.soln_op_vpts(etype, renderer.divisor)
        xd = xd[..., self.rdata[etype]].transpose(0, 2, 1)
        nsvpts, neles, _ = xd.shape

        self.soln_ops[etype] = soln_op

        snodes = get_subdiv(etype, renderer.divisor).subnodes
        conn = self._connectivity(etype, snodes, neles, nsvpts)

        renderer._write_domain(dom, self.sname, domid, etype, neles,
                               self._points(etype, xd), conn)
        return dom

    def csolns_cgrads(self, soln, grad_soln, etype):
        soln_op = self.soln_ops[etype]
        rgn = self.rdata[etype]

        # Subset and transpose the solution, then interpolate to subdiv points
        csolns = soln_op @ soln[etype][..., rgn].swapaxes(0, 1)
        if grad_soln is not None:
            # Interpolate the gradients to the points
            cg = np.rollaxis(grad_soln[etype], 2)[..., rgn]
            return csolns, soln_op @ cg
        else:
            return csolns, None

    def run_postproc(self, plugins, etype, psolns, pgrads):
        data = PostProcData(self.renderer.scfg, psolns, grad_pris=pgrads)
        for pp in plugins:
            pp.run(data)

        return data.fields

    def _emit(self, mesh_n, dom, fname, arr):
        # Scalars publish at values; vectors split into values/x, /y, /z
        path = f'{dom}/fields/{fname}/values'
        if len(comps := arr.T) == 1:
            mesh_n[path] = comps[0]
        else:
            for x, sl in zip('xyz', comps):
                mesh_n[f'{path}/{x}'] = sl

    def _publish_direct(self, mesh_n, fields):
        for key, items in fields.items():
            dom = self.renderer.dinfo[self.sname, key]
            for fname, arr in items:
                self._emit(mesh_n, dom, fname, arr)

    def _publish_clean(self, mesh_n, fields):
        comm, _, _ = get_comm_rank_root()
        dtype = self.renderer.dtype

        # Re-key by published (possibly namespaced) field name
        byfield = defaultdict(dict)
        for key, items in fields.items():
            for fname, arr in items:
                byfield[fname][key] = arr

        # Sorted per-source list so idle ranks hit every collective
        for field in sorted(self.renderer._source_fields[self.sname]):
            fname = self.renderer._field_name(self.sname, field)
            arrs = byfield.get(fname, {})

            ncomp = max((a.shape[-1] for a in arrs.values()), default=1)
            ncomp = comm.allreduce(ncomp, op=mpi.MAX)

            efields = {key: arr.astype(dtype, copy=False)
                       for key, arr in arrs.items()}

            avg = self.cleaner.average(efields, ncomp, dtype)
            for key, vals in avg.items():
                dom = self.renderer.dinfo[self.sname, key]
                self._emit(mesh_n, dom, fname, vals)


class _BoundaryAscentOutput(_VolumeAscentOutput):
    kind = 'boundary'

    def __init__(self, renderer, sname, region, clean=False):
        self.renderer = renderer
        self.sname = sname

        # Boundary surface sources take the form bc/<name>
        if not region.startswith('bc/'):
            raise ValueError(f'Invalid surface source: {region}')

        conn = renderer.mesh.bcon_for(region.removeprefix('bc/'))

        # Per-itype patches, each a flat (eidxs, etype, mop, sop, fidx, svpts)
        self.patches = patches = defaultdict(list)
        if conn is not None:
            for etype, fidx, eidxs in conn.items():
                itype, mop, sop, svpts = renderer.adapter.face_soln_op_vpts(
                    etype, fidx, renderer.divisor
                )
                patches[itype].append((eidxs, etype, mop, sop, fidx, svpts))

        self._setup_clean(clean)

    def _make_cleaner(self):
        mesh, divisor = self.renderer.mesh, self.renderer.divisor
        cnodemap, svptsmap = {}, {}
        for itype, patches in self.patches.items():
            ishapecls = subclass_where(BaseShape, name=itype)
            pieces = []
            for eidxs, etype, _, _, fidx, _ in patches:
                shapecls = subclass_where(BaseShape, name=etype)
                spts_nodes = mesh.spts_nodes[etype]
                cidxs = shapecls.face_corner_pts_idxs(fidx,
                                                      spts_nodes.shape[1])
                pieces.append(spts_nodes[np.ix_(eidxs, cidxs)])
            cnodemap[itype] = np.concatenate(pieces)
            svptsmap[itype] = ishapecls.std_ele(divisor)

        return _build_cleaner(mesh, divisor, cnodemap, svptsmap)

    def domain_keys(self):
        return list(self.patches)

    def build_blueprint(self, renderer, domid, itype):
        dom = f'domain_{domid}'
        spts = renderer.mesh.spts

        # Per-patch face coords concatenated to (nsvpts, neles, ndim)
        xparts = []
        for eidxs, etype, mop, _, _, _ in self.patches[itype]:
            sp = spts[etype][:, eidxs]
            vp = (mop @ sp.reshape(len(sp), -1)).reshape(-1, *sp.shape[1:])
            xparts.append(vp)

        xd = np.concatenate(xparts, axis=1)
        nsvpts, neles, _ = xd.shape

        snodes = get_subdiv(itype, renderer.divisor).subnodes
        conn = self._connectivity(itype, snodes, neles, nsvpts)

        renderer._write_domain(dom, self.sname, domid, itype, neles,
                               self._points(itype, xd), conn)
        return dom

    def csolns_cgrads(self, soln, grad_soln, itype):
        css, cgs = [], []
        for eidxs, etype, _, sop, _, _ in self.patches[itype]:
            css.append(sop @ soln[etype][..., eidxs].swapaxes(0, 1))
            if grad_soln is not None:
                cg = np.rollaxis(grad_soln[etype], 2)[..., eidxs]
                cgs.append(sop @ cg)

        csolns = np.concatenate(css, axis=2)
        cgrads = np.concatenate(cgs, axis=3) if cgs else None
        return csolns, cgrads

    def run_postproc(self, plugins, itype, psolns, pgrads):
        r = self.renderer
        spts, elementscls = r.mesh.spts, r.adapter.elementscls

        # Slice per-patch views, run postprocs, merge by field across patches
        merged = defaultdict(list)
        offset = 0
        for eidxs, etype, mop, _, fidx, svpts in self.patches[itype]:
            sl = slice(offset, offset + len(eidxs))
            ppris = [p[..., sl] for p in psolns]
            if pgrads is None:
                ppgrads = None
            else:
                ppgrads = [g[..., sl] for g in pgrads]
            offset = sl.stop

            # Physical locations of the face visualisation points
            psp = spts[etype][:, eidxs]
            vp = mop @ psp.reshape(len(psp), -1)
            ploc = vp.reshape(-1, *psp.shape[1:]).transpose(2, 0, 1)

            norm = subclass_where(BaseShape, name=etype).faces[fidx][2]
            finfo = FaceInfo(etype, fidx, svpts, norm)
            data = BoundaryPostProcData(r.scfg, ppris, ploc, ppgrads,
                                        elementscls, psp, finfo)
            for pp in plugins:
                pp.run(data)

            for fname, arr in data.fields.items():
                merged[fname].append(arr)

        return {fname: np.concatenate(parts, axis=1)
                for fname, parts in merged.items()}


class AscentRenderer:
    # Element name mapping for conduit
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    def __init__(self, adapter, isrestart):
        comm, _, _ = get_comm_rank_root()

        self.adapter = adapter
        self.mesh = adapter.mesh
        self.acfg = acfg = adapter.acfg
        self.cfgsect = cfgsect = adapter.cfgsect

        self.scfg = adapter.scfg
        self.elementscls = adapter.elementscls
        self.dtype = adapter.dtype

        # Set order for subdivision
        sorder = self.scfg.getint('solver', 'order')
        self.divisor = acfg.getint(cfgsect, 'division', sorder)

        self.clean = acfg.getbool(cfgsect, 'clean', True)

        # Named surface sources (one per surface-{name} = <region> entry)
        self.surfaces = {k.removeprefix('surface-'): acfg.get(cfgsect, k)
                         for k in acfg.items(cfgsect, prefix='surface-')}

        # Volume source: implicit if no surfaces; opt-in alongside surfaces
        self.want_volume = (not self.surfaces or
                            acfg.getbool(cfgsect, 'volume', False))

        # Load Conduit
        self.conduit = ConduitWrappers()

        # Setup outputting options
        self.basedir = acfg.getpath(cfgsect, 'basedir', '.', abs=True)
        self.isrestart = isrestart
        self._image_paths = []

        # Region data (per-etype element indices) only needed for volume
        if self.want_volume:
            self.rdata = region_data(acfg, cfgsect, self.mesh)

        # Per-source output strategies, keyed by source name (topology)
        self.sources = {}
        if self.want_volume:
            self.sources['volume'] = _VolumeAscentOutput(self,
                                                         clean=self.clean)
        for sname, sregion in self.surfaces.items():
            self.sources[sname] = _BoundaryAscentOutput(self, sname, sregion,
                                                        clean=self.clean)

        # Expressions to plot and configs (scenes need self.sources)
        self._exprs = []
        self._user_fields = set()
        self._source_fields = defaultdict(set)
        self._fields_write = set()
        self._fields_read = set()
        self._init_fields()
        self._init_postproc()
        self._init_scenes()
        self._init_pipelines()

        if not self._fields_read.issubset(self._fields_write):
            missing = self._fields_read - self._fields_write
            raise AscentError(f'Fields used but not defined: {missing}')

        # Gradient pre-processing
        self._init_gradients()

        # Generate a Conduit node for the mesh
        self.mesh_n = ConduitNode(self.conduit)

        # One Conduit domain per (source, key); topology = sname
        self.domains = [(src, key) for src in self.sources.values()
                        for key in src.domain_keys()]
        doff = comm.exscan(len(self.domains)) or 0

        # Build the Conduit blueprint mesh for the regions
        self.dinfo = {}
        for i, (src, key) in enumerate(self.domains):
            dom = src.build_blueprint(self, doff + i, key)
            self.dinfo[src.sname, key] = dom

        # Initialise Ascent and open an instance
        self._init_ascent()

    def __del__(self):
        if getattr(self, 'ascent_ptr', None):
            self.lib.ascent_close(self.ascent_ptr)

    def _write_state_meta(self, mesh_n, dom, domid):
        mesh_n[f'{dom}/state/domain_id'] = domid
        mesh_n[f'{dom}/state/config/keyword'] = 'Config'
        mesh_n[f'{dom}/state/config/data'] = self.scfg.tostr()
        mesh_n[f'{dom}/state/mesh_uuid/keyword'] = 'Mesh_UUID'
        mesh_n[f'{dom}/state/mesh_uuid/data'] = self.mesh.uuid

    def _field_name(self, sname, field):
        # Ensure user fields are namespaced per source
        if field in self._user_fields:
            return f'{sname}_{field}'
        else:
            return field

    def _write_field_meta(self, mesh_n, dom, sname):
        for field in self._source_fields[sname]:
            fname = self._field_name(sname, field)
            mesh_n[f'{dom}/fields/{fname}/association'] = 'vertex'
            mesh_n[f'{dom}/fields/{fname}/volume_dependent'] = 'false'
            mesh_n[f'{dom}/fields/{fname}/topology'] = sname

    def _write_domain(self, dom, sname, domid, etype, neles, xyz, conn):
        mesh_n = self.mesh_n
        cs = f'{sname}_coords'
        elem = f'{dom}/topologies/{sname}/elements'

        self._write_state_meta(mesh_n, dom, domid)

        mesh_n[f'{dom}/coordsets/{cs}/type'] = 'explicit'
        mesh_n[f'{dom}/topologies/{sname}/coordset'] = cs
        mesh_n[f'{dom}/topologies/{sname}/type'] = 'unstructured'

        for l, x in zip('xyz', xyz):
            mesh_n[f'{dom}/coordsets/{cs}/values/{l}'] = x

        self._write_field_meta(mesh_n, dom, sname)

        sdiv = get_subdiv(etype, self.divisor)
        snodes = sdiv.subnodes

        mesh_n[f'{elem}/connectivity'] = conn

        # Subdivide; handle elements split into multiple subcell types
        if len(scells := set(sdiv.subcells)) > 1:
            mesh_n[f'{elem}/shape'] = 'mixed'

            for sc in scells:
                an = self.bp_emap[sc]
                mesh_n[f'{elem}/shape_map/{an}'] = sdiv.vtk_types[sc]

            scell_t = sdiv.vtk_subcelltypes
            mesh_n[f'{elem}/shapes'] = np.tile(scell_t, neles)

            scell_s = sdiv.subcells
            scell_s = [sdiv.cell_nodes[sc] for sc in scell_s]
            mesh_n[f'{elem}/sizes'] = np.tile(scell_s, neles)

            scell_o = np.tile(sdiv.subcelloffs, (neles, 1))
            scell_o += (np.arange(neles)*len(snodes))[:, None]
            scell_o = np.concatenate(([0], scell_o.flat[:-1]))
            mesh_n[f'{elem}/offsets'] = scell_o
        else:
            mesh_n[f'{elem}/shape'] = self.bp_emap[etype]

    def _init_ascent(self):
        comm, _, _ = get_comm_rank_root()

        self.lib = lib = AscentWrappers()
        self.ascent_ptr = lib.ascent_create(None)

        self.ascent_config = ConduitNode(self.conduit)
        self.ascent_config['mpi_comm'] = comm.py2f()
        self.ascent_config['runtime/type'] = 'ascent'
        backend = self.acfg.get(self.cfgsect, 'viskores-backend', 'serial')
        self.ascent_config['runtime/viskores/backend'] = backend

        # Disable autoload of ascent_actions.{yaml,json}
        self.ascent_config['actions_file'] = ''

        # Open an Ascent instance
        lib.ascent_open(self.ascent_ptr, self.ascent_config)

        # Pre configure scenes and pipelines
        self.actions = ConduitNode(self.conduit)
        self._add_scene = self.actions.append()
        self._add_scene['action'] = 'add_scenes'
        self._add_scene['scenes'] = self.scenes

        self._add_pipeline = self.actions.append()
        self._add_pipeline['action'] = 'add_pipelines'
        self._add_pipeline['pipelines'] = self.pipelines

    def _register_user_field(self, sname, field):
        # Mark field as user-namespaced and reserve its slot on this source
        self._user_fields.add(field)
        fname = self._field_name(sname, field)
        if fname in self._fields_write:
            raise KeyError(f'Field {fname!r} already exists')
        self._fields_write.add(fname)
        self._source_fields[sname].add(field)

    def _init_fields(self):
        cons = self.scfg.items_as('constants', float)

        for k in self.acfg.items(self.cfgsect, prefix='field-'):
            field = k.removeprefix('field-')

            # Each source publishes its own namespaced copy
            for sname in self.sources:
                self._register_user_field(sname, field)

            raw = self.acfg.get(self.cfgsect, k)
            comps = [process_expr(c, cons) for c in split_components(raw)]
            self._exprs.append((field, comps))

    def _init_postproc(self):
        # Parse postproc-{name} = <sources>; one plugin list per source
        groups = defaultdict(list)
        for k in self.acfg.items(self.cfgsect, prefix='postproc-'):
            name = k.removeprefix('postproc-')
            for s in self.acfg.get(self.cfgsect, k).split(','):
                sname = s.strip()
                if sname not in self.sources:
                    raise AscentError(f'Postproc {name!r}: unknown source '
                                      f'{sname!r}')
                groups[sname].append(name)

        self._postproc_plugins = {}
        dsrc = get_source('soln', self.scfg, None, self.mesh.ndims)
        for sname, names in groups.items():
            plugins = dsrc.quantities(names, self.sources[sname].kind)
            self._postproc_plugins[sname] = plugins

            for pp in plugins:
                for fname in pp.fields:
                    self._register_user_field(sname, fname)

    def _init_gradients(self):
        privars = self.elementscls.privars(self.mesh.ndims, self.scfg)
        pp_plugins = self._postproc_plugins.values()

        # First, see what gradient terms are used by expressions
        g_pnames = set()
        for _, comps in self._exprs:
            for c in comps:
                g_pnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', c))

        # Then, see if any post-processing plugins require gradients
        if any(pp.needs_grads for ppl in pp_plugins for pp in ppl):
            g_pnames.update(privars)

        if g_pnames and not self.adapter.has_grads:
            raise AscentError('Gradients required but not available')

        self._gradpinfo = [(pname, privars.index(pname)) for pname in g_pnames]

    def _init_pipelines(self):
        self.pipelines = pl = ConduitNode(self.conduit)

        for k in self.acfg.items(self.cfgsect, prefix='pipeline-'):
            pn = k.removeprefix('pipeline-')
            cfg = self.acfg.getliteral(self.cfgsect, k)
            cfg = [cfg] if isinstance(cfg, dict) else cfg

            for j, filt in enumerate(cfg):
                params = ConduitNode(self.conduit)

                pl[f'pl_{pn}/f{j}/type'] = filt.pop('type')
                for kf, vf in filt.items():
                    if kf == 'output-name':
                        if vf in self._fields_write:
                            raise KeyError(f'Output name {vf!r} already used')
                        self._fields_write.add(vf)
                    elif kf == 'field':
                        self._fields_read.add(vf)

                    params[_bp_key(kf)] = vf

                pl[f'pl_{pn}/f{j}/params'] = params

    def _init_scenes(self):
        self.scenes = ConduitNode(self.conduit)

        for k in self.acfg.items(self.cfgsect, prefix='scene-'):
            sn = k.removeprefix('scene-')
            cfg = self.acfg.getliteral(self.cfgsect, k)

            plots = cfg.get('plots')
            if not isinstance(plots, list) or not plots:
                raise AscentError(f'Scene {sn!r} must define plots = [...]')

            for j, plot in enumerate(plots):
                self._init_plot(sn, f'p{j}', plot)

            for kc, vc in cfg.items():
                if kc.startswith('render-'):
                    rname = kc.removeprefix('render-')
                    self._render_options(f's_{sn}/renders/r_{rname}', vc)

            # Synthesise a single default render from scene-level keys
            keys = ('image-name', 'image-prefix')
            defaults = {k: cfg[k] for k in keys if k in cfg}
            has_render = any(kc.startswith('render-') for kc in cfg)
            if defaults and not has_render:
                self._render_options(f's_{sn}/renders/r_default', defaults)
            elif not has_render:
                raise KeyError(f'No render config given for scene {sn!r}')

    def _init_plot(self, sn, pname, plot):
        # Default to the only source when there is no ambiguity
        if (src := plot.get('source')) is None:
            if len(self.sources) > 1:
                raise AscentError(f'Plot {pname!r} of scene {sn!r} needs '
                                  'source= when multiple sources exist')
            src = first(self.sources)

        # Plot field references are auto-namespaced to the plot's source
        if (field := plot.get('field')) is not None:
            field = self._field_name(src, field)
            self._fields_read.add(field)

        for kc, vc in plot.items():
            if kc.startswith('render-') or kc == 'source':
                continue
            if kc == 'pipeline':
                self.scenes[f's_{sn}/plots/{pname}/pipeline'] = f'pl_{vc}'
            elif kc == 'field':
                self.scenes[f's_{sn}/plots/{pname}/field'] = field
            else:
                self.scenes[f's_{sn}/plots/{pname}/{_bp_key(kc)}'] = vc

    def _evaluate_exprs(self, adapter):
        elementscls = self.elementscls

        # Get the primitive variable names
        pnames = elementscls.privars(self.mesh.ndims, self.scfg)

        tcurr = adapter.tcurr
        cycle = adapter.cycle

        # Obtain the solution (and gradients if needed)
        soln = adapter.soln
        grad_soln = adapter.grad_soln if self._gradpinfo else None

        # out[sname] = {key: [(field, arr)]} for per-source publish
        out = defaultdict(dict)

        # Iterate over each (source, key) pair in our blueprint
        for (sname, key), dom in self.dinfo.items():
            self.mesh_n[f'{dom}/state/time/keyword'] = 'Time'
            self.mesh_n[f'{dom}/state/time/data'] = str(tcurr)
            self.mesh_n[f'{dom}/state/cycle'] = cycle

            source = self.sources[sname]
            csolns, cgrads = source.csolns_cgrads(soln, grad_soln, key)

            # Adapter chooses conservative->primitive vs name-mapped (tavg)
            psolns, pgrads = adapter.psolns_pgrads(csolns, cgrads)

            # Prepare the substitutions dictionary
            subs = dict(zip(pnames, psolns), t=tcurr)

            # Prepare any required gradients; None slots are skipped
            if self._gradpinfo and pgrads is not None:
                for pname, pidx in self._gradpinfo:
                    if pgrads[pidx] is None:
                        continue
                    for dim, grad in zip('xyz', pgrads[pidx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            items = []

            # Field expressions
            for field, comps in self._exprs:
                arr = np.stack([npeval(c, subs) for c in comps], axis=-1)
                items.append((self._field_name(sname, field), arr))

            # Postproc plugins for this source/key
            if plugins := self._postproc_plugins.get(sname):
                pp_fields = source.run_postproc(plugins, key, psolns, pgrads)
                for field, arr in pp_fields.items():
                    items.append((self._field_name(sname, field),
                                  np.atleast_3d(arr)))

            out[sname][key] = items

        return out

    def _render_options(self, path, opts):
        for k, v in opts.items():
            if k in ('image-name', 'image-prefix'):
                continue

            self.scenes[f'{path}/{_bp_key(k)}'] = v

        if (name := opts.get('image-name')) is not None:
            gen = file_path_gen(self.basedir, name, self.isrestart,
                                extn='.png')
            self._image_paths.append((f'scenes/{path}/image_name', gen))
        elif (prefix := opts.get('image-prefix')) is not None:
            self.scenes[f'{path}/image_prefix'] = f'{self.basedir}/{prefix}'
        else:
            raise KeyError(f'Render at {path!r} needs image-name or '
                           'image-prefix')

    def render(self, adapter):
        comm, _, _ = get_comm_rank_root()

        # Set file names
        for path, gen in self._image_paths:
            self._add_scene[path] = str(gen.send(adapter.tcurr))

        # Set field expressions; publish per source
        fields = self._evaluate_exprs(adapter)
        for sname, source in self.sources.items():
            source.publish_fields(self.mesh_n, fields.get(sname, {}))

        self.lib.ascent_publish(self.ascent_ptr, self.mesh_n)
        self.lib.ascent_execute(self.ascent_ptr, self.actions)

        comm.barrier()


class AscentPlugin(BaseSolnPlugin):
    name = 'ascent'
    systems = '.*'
    dimensions = '2|3'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        adapter = _IntegratorAdapter(intg, intg.cfg, cfgsect)
        self._renderer = AscentRenderer(adapter, intg.isrestart)

    def __call__(self, intg):
        self._renderer.render(_IntegratorAdapter(intg, intg.cfg, self.cfgsect))
