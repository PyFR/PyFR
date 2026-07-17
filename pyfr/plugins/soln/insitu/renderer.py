from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import process_expr
from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.common import region_data
from pyfr.plugins.postproc.runner import PostProcRunner
from pyfr.plugins.soln.insitu.conduit import ConduitNode, ConduitWrappers
from pyfr.plugins.soln.insitu.base import split_components
from pyfr.plugins.soln.insitu.outputs import BoundaryOutput, VolumeOutput
from pyfr.writers.vtk.shapes import get_vtk_shape


class InSituError(Exception): pass


class InSituRenderer:
    # Conduit blueprint element name mapping
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    # Subclasses can narrow this to their own error type
    error_cls = InSituError

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

        # Load Conduit (subclass may override for host-specific fallbacks)
        self.conduit = self._load_conduit()

        # Setup outputting options
        self.isrestart = isrestart

        # Region data (per-etype element indices) only needed for volume
        if self.want_volume:
            self.rdata = region_data(acfg, cfgsect, self.mesh)

        # Per-source output strategies, keyed by source name (topology)
        self.sources = {}
        if self.want_volume:
            self.sources['volume'] = VolumeOutput(self, clean=self.clean)
        for sname, sregion in self.surfaces.items():
            self.sources[sname] = BoundaryOutput(self, sname, sregion,
                                                 clean=self.clean)

        # Expressions and field/postproc bookkeeping
        self._exprs = []
        self._user_fields = set()
        self._source_fields = defaultdict(set)
        self._fields_write = set()
        self._fields_read = set()
        self._init_fields()
        self._init_postproc()

        # Host-specific publishing config (Ascent scenes/pipelines, etc.)
        self._init_host_publish()

        if not self._fields_read.issubset(self._fields_write):
            missing = self._fields_read - self._fields_write
            raise self.error_cls(f'Fields used but not defined: {missing}')

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

        # Host-specific instance init (open the library, etc.)
        self._init_host()

        self.adapter = None

    # --- Host hooks ----------------------------------------------------

    def _load_conduit(self):
        return ConduitWrappers()

    def _init_host_publish(self):
        # Override to wire in scenes/pipelines/etc.; may add to
        # self._fields_read and self._fields_write.
        pass

    def _init_host(self):
        # Override to open the host instance using self.mesh_n
        pass

    def _domain_path(self, sname, domid):
        # Path prefix for the domain inside mesh_n.  Override on hosts that
        # nest domains (eg. Catalyst's per-source channels).
        return f'domain_{domid}'

    def _write_step_state(self, dom, tcurr, cycle):
        # Per-domain time/cycle state.  Ascent uses keyword/data sub-paths;
        # override if the host expects a different schema.
        self.mesh_n[f'{dom}/state/time/keyword'] = 'Time'
        self.mesh_n[f'{dom}/state/time/data'] = str(tcurr)
        self.mesh_n[f'{dom}/state/cycle'] = cycle

    def _emit_field(self, mesh_n, dom, fname, arr):
        # Scalars publish at values; vectors split into values/x, /y, /z.
        # Catalyst overrides this to emit interleaved (AoS) arrays.
        path = f'{dom}/fields/{fname}/values'
        if len(comps := arr.T) == 1:
            mesh_n[path] = comps[0]
        else:
            for x, sl in zip('xyz', comps):
                mesh_n[f'{path}/{x}'] = sl

    def _emit_coords(self, mesh_n, dom, cs, xyz):
        # Split coords; Catalyst overrides to emit interleaved (AoS).
        for l, x in zip('xyz', xyz):
            mesh_n[f'{dom}/coordsets/{cs}/values/{l}'] = x

    # --- Generic methods ----------------------------------------------

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

        self._emit_coords(mesh_n, dom, cs, xyz)

        self._write_field_meta(mesh_n, dom, sname)

        subdiv = get_vtk_shape(etype, self.divisor)
        snodes = subdiv.subnodes

        mesh_n[f'{elem}/connectivity'] = conn

        # Subdivide; handle elements split into multiple subcell types
        if len(scells := set(subdiv.subcells)) > 1:
            mesh_n[f'{elem}/shape'] = 'mixed'

            for sc in scells:
                an = self.bp_emap[sc]
                mesh_n[f'{elem}/shape_map/{an}'] = subdiv.vtk_types[sc]

            scell_t = subdiv.subcelltypes
            mesh_n[f'{elem}/shapes'] = np.tile(scell_t, neles)

            scell_s = subdiv.subcells
            scell_s = [subdiv.vtk_nodes[sc] for sc in scell_s]
            mesh_n[f'{elem}/sizes'] = np.tile(scell_s, neles)

            scell_o = np.tile(subdiv.subcelloffs, (neles, 1))
            scell_o += (np.arange(neles)*len(snodes))[:, None]
            scell_o = np.concatenate(([0], scell_o.flat[:-1]))
            mesh_n[f'{elem}/offsets'] = scell_o
        else:
            mesh_n[f'{elem}/shape'] = self.bp_emap[etype]

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
        # Parse postproc-{name} = <sources>; one runner per source
        groups = defaultdict(list)
        for k in self.acfg.items(self.cfgsect, prefix='postproc-'):
            name = k.removeprefix('postproc-')
            for s in self.acfg.get(self.cfgsect, k).split(','):
                sname = s.strip()
                if sname not in self.sources:
                    raise self.error_cls(f'Postproc {name!r}: unknown '
                                         f'source {sname!r}')
                groups[sname].append(name)

        self._postproc_runners = {}
        for sname, names in groups.items():
            runner = PostProcRunner(names, self.mesh.ndims, self.scfg,
                                    export_type=self.sources[sname].kind)
            self._postproc_runners[sname] = runner

            for fname in runner.fields(public_only=True):
                self._register_user_field(sname, fname)

    def _init_gradients(self):
        # Determine what gradients, if any, are required
        g_pnames = set()
        for _, comps in self._exprs:
            for c in comps:
                g_pnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', c))

        privars = self.elementscls.privars(self.mesh.ndims, self.scfg)

        # Postproc plugins index pgrads positionally; request them all
        if any(r.needs_grads for r in self._postproc_runners.values()):
            g_pnames.update(privars)

        if g_pnames and not self.adapter.has_grads:
            raise self.error_cls('Gradients required but not available')

        self._gradpinfo = [(pname, privars.index(pname)) for pname in g_pnames]

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
            self._write_step_state(dom, tcurr, cycle)

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
            if runner := self._postproc_runners.get(sname):
                pp_fields = source.run_postproc(runner, key, psolns, pgrads)
                for field, arr in pp_fields.items():
                    items.append((self._field_name(sname, field),
                                  np.atleast_3d(arr)))

            out[sname][key] = items

        return out
