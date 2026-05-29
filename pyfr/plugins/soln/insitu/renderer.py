from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import process_expr
from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.common import region_data
from pyfr.plugins.fields.runner import FieldRunner
from pyfr.plugins.soln.insitu.base import split_components
from pyfr.plugins.soln.insitu.conduit import ConduitNode, ConduitWrappers
from pyfr.writers.vtk.shapes import get_vtk_shape


class InSituError(Exception): pass


class InSituRenderer:
    # In-situ renderer: serialise a Snapshot region into a Conduit blueprint
    # node, run providers + expression evaluation per step, and publish the
    # results to the host (Catalyst or Ascent).
    #
    # Lifetime model — Freddie's rule (plugins don't retain intg):
    #   * __init__ takes a transient `snap` (IntgSnapshot for soln plugins,
    #     FileSnapshot for CLI).  We capture static metadata (mesh, config,
    #     elementscls, dtype, has_grads) into self.X and DO NOT keep the
    #     snap reference.  Regions also drop their snap reference after
    #     building geometry.
    #   * render(snap) is called per step (or per file in CLI batch mode)
    #     with a fresh snap.  For each region we build a per-call
    #     SnapshotSample(region, snap); the sample owns pris/grad_pris/fields
    #     and is dropped at end of render().

    # Conduit blueprint element name mapping
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    error_cls = InSituError

    def __init__(self, snap, acfg, cfgsect, isrestart):
        comm, _, _ = get_comm_rank_root()

        # Static metadata captured from snap; snap reference NOT retained
        self.mesh = snap.mesh
        self.scfg = snap.config
        self.elementscls = snap.elementscls
        self.dtype = snap.dtype
        self._snap_has_grads = snap.has_grads

        self.acfg = acfg
        self.cfgsect = cfgsect
        self.isrestart = isrestart

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

        # Build regions per source.  Each region captures its own static
        # geometry from snap and drops the snap reference internally.
        self.regions = {}
        self._source_kinds = {}
        if self.want_volume:
            rdata = region_data(acfg, cfgsect, self.mesh)
            self.regions['volume'] = snap.vis(
                spec=list(rdata), divisor=self.divisor, clean=self.clean)
            self._source_kinds['volume'] = 'volume'
        for sname, sregion in self.surfaces.items():
            self.regions[sname] = snap.surface(
                sregion, divisor=self.divisor, clean=self.clean)
            self._source_kinds[sname] = 'boundary'

        # Expressions and derived-field bookkeeping
        self._exprs = []
        self._user_fields = set()
        self._source_fields = defaultdict(set)
        self._fields_write = set()
        self._fields_read = set()
        self._init_fields()
        self._init_field_runners()

        # Host-specific publishing config (Ascent scenes/pipelines, etc.)
        self._init_host_publish()

        if not self._fields_read.issubset(self._fields_write):
            missing = self._fields_read - self._fields_write
            raise self.error_cls(f'Fields used but not defined: {missing}')

        # Gradient pre-processing
        self._init_gradients()

        # Generate a Conduit node for the mesh
        self.mesh_n = ConduitNode(self.conduit)

        # One Conduit domain per (source, etype/itype) pair
        self.domains = [(sname, et)
                        for sname, r in self.regions.items() for et in r.etypes]
        doff = comm.exscan(len(self.domains)) or 0

        # Build the Conduit blueprint mesh for the regions (geometry-only)
        self.dinfo = {}
        for i, (sname, etype) in enumerate(self.domains):
            dom = self._domain_path(sname, doff + i)
            self._build_blueprint(dom, sname, doff + i, etype)
            self.dinfo[sname, etype] = dom

        # Host-specific instance init (open the library, etc.)
        self._init_host()

    # --- Host hooks ----------------------------------------------------

    def _load_conduit(self):
        return ConduitWrappers()

    def _init_host_publish(self):
        # Override to wire in scenes/pipelines/etc.
        pass

    def _init_host(self):
        # Override to open the host instance using self.mesh_n
        pass

    def _domain_path(self, sname, domid):
        return f'domain_{domid}'

    def _write_step_state(self, dom, tcurr, cycle):
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
        if field in self._user_fields:
            return f'{sname}_{field}'
        return field

    def _write_field_meta(self, mesh_n, dom, sname):
        for field in self._source_fields[sname]:
            fname = self._field_name(sname, field)
            mesh_n[f'{dom}/fields/{fname}/association'] = 'vertex'
            mesh_n[f'{dom}/fields/{fname}/volume_dependent'] = 'false'
            mesh_n[f'{dom}/fields/{fname}/topology'] = sname

    def _flatten_coords(self, ploc, clean):
        # Convert region.ploc[etype] (or sample.ploc[etype]) to per-axis
        # arrays for _emit_coords.  clean -> (ndims, n_kept).  Raw ->
        # (ndims, nsvpts, neles); needs an element-major flatten so cell c's
        # verts land at flat indices [c*nsvpts, ..., (c+1)*nsvpts-1].
        if clean:
            return ploc
        return ploc.transpose(0, 2, 1).reshape(ploc.shape[0], -1)

    def _build_connectivity(self, etype, ploc, region):
        # Use cleaner.layouts when clean (deduplicated nodal connectivity);
        # otherwise tile the subnodes pattern per element.
        subdiv = get_vtk_shape(etype, self.divisor)
        snodes = subdiv.subnodes
        if region.clean:
            conn = region.cleaner.layouts[etype][0][:, snodes]
        else:
            neles = ploc.shape[-1]
            nsvpts = ploc.shape[-2]
            conn = np.tile(snodes, (neles, 1))
            conn += (np.arange(neles)*nsvpts)[:, None]
        return conn, subdiv, snodes

    def _build_blueprint(self, dom, sname, domid, etype):
        region = self.regions[sname]
        mesh_n = self.mesh_n
        cs = f'{sname}_coords'
        elem = f'{dom}/topologies/{sname}/elements'

        self._write_state_meta(mesh_n, dom, domid)

        mesh_n[f'{dom}/coordsets/{cs}/type'] = 'explicit'
        mesh_n[f'{dom}/topologies/{sname}/coordset'] = cs
        mesh_n[f'{dom}/topologies/{sname}/type'] = 'unstructured'

        # Coordset from region.ploc (static body-frame geometry).  If a
        # transformer mutates sample.ploc per step, the publish() path uses
        # sample.ploc instead — but the blueprint built here uses the region's
        # static coords as a one-time baseline (Catalyst/Ascent may rebuild
        # coords per step via _emit_coords).
        ploc = region.ploc[etype]
        self._emit_coords(mesh_n, dom, cs, self._flatten_coords(ploc,
                                                                region.clean))

        self._write_field_meta(mesh_n, dom, sname)

        # Connectivity from region (clean: cleaner.layouts; raw: tile)
        conn, subdiv, snodes = self._build_connectivity(etype, ploc, region)
        neles = conn.shape[0]

        mesh_n[f'{elem}/connectivity'] = conn

        # Subdivide; handle elements split into multiple subcell types
        if len(scells := set(subdiv.subcells)) > 1:
            mesh_n[f'{elem}/shape'] = 'mixed'

            for sc in scells:
                an = self.bp_emap[sc]
                mesh_n[f'{elem}/shape_map/{an}'] = subdiv.vtk_types[sc]

            scell_t = subdiv.subcelltypes
            mesh_n[f'{elem}/shapes'] = np.tile(scell_t, neles)

            scell_s = [subdiv.vtk_nodes[sc] for sc in subdiv.subcells]
            mesh_n[f'{elem}/sizes'] = np.tile(scell_s, neles)

            scell_o = np.tile(subdiv.subcelloffs, (neles, 1))
            scell_o += (np.arange(neles)*len(snodes))[:, None]
            scell_o = np.concatenate(([0], scell_o.flat[:-1]))
            mesh_n[f'{elem}/offsets'] = scell_o
        else:
            mesh_n[f'{elem}/shape'] = self.bp_emap[etype]

    def _register_user_field(self, sname, field):
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
            for sname in self.regions:
                self._register_user_field(sname, field)

            raw = self.acfg.get(self.cfgsect, k)
            comps = [process_expr(c, cons) for c in split_components(raw)]
            self._exprs.append((field, comps))

    def _init_field_runners(self):
        # Parse postproc-{name} = <sources>; one FieldRunner per source.  The
        # cfg key prefix `postproc-` is user-facing (back-compat).
        groups = defaultdict(list)
        for k in self.acfg.items(self.cfgsect, prefix='postproc-'):
            name = k.removeprefix('postproc-')
            for s in self.acfg.get(self.cfgsect, k).split(','):
                sname = s.strip()
                if sname not in self.regions:
                    raise self.error_cls(f'Field provider {name!r}: unknown '
                                         f'source {sname!r}')
                groups[sname].append(name)

        self._field_runners = {}
        for sname, names in groups.items():
            export_type = self._source_kinds[sname]
            runner = FieldRunner(names, self.mesh.ndims, self.scfg,
                                 export_type=export_type)
            self._field_runners[sname] = runner

            for fname in runner.fields(public_only=True):
                self._register_user_field(sname, fname)

    def _init_gradients(self):
        # Determine what gradients, if any, are required
        g_pnames = set()
        for _, comps in self._exprs:
            for c in comps:
                g_pnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', c))

        privars = self.elementscls.privars(self.mesh.ndims, self.scfg)

        # Field providers index pgrads positionally; request them all
        if any(r.needs_grads for r in self._field_runners.values()):
            g_pnames.update(privars)

        if g_pnames and not self._snap_has_grads:
            raise self.error_cls('Gradients required but not available')

        self._gradpinfo = [(pname, privars.index(pname)) for pname in g_pnames]

    def _evaluate_exprs(self, snap):
        # Per-step entry point.  Build a SnapshotSample per region from the
        # transient `snap`, run providers on each sample, then evaluate user
        # expressions over sample.pris/grad_pris.  Samples drop at end of
        # render() — no state is retained between calls.
        elementscls = self.elementscls
        pnames = elementscls.privars(self.mesh.ndims, self.scfg)

        tcurr = snap.tcurr
        cycle = snap.cycle

        # Build per-region samples (this is where all the cleaner.average and
        # MPI collectives fire — every rank, deterministic order).
        samples = {sname: region.sample(snap)
                   for sname, region in self.regions.items()}

        # Run field providers on each sample; results land in sample.fields
        for sname, runner in self._field_runners.items():
            runner.run_on_sample(samples[sname], public_only=True)

        # out[sname][etype] = [(field, arr), ...] for per-source publish
        out = defaultdict(dict)

        for (sname, etype), dom in self.dinfo.items():
            self._write_step_state(dom, tcurr, cycle)

            sample = samples[sname]
            psolns = sample.pris[etype]
            pgrads = (sample.grad_pris[etype] if self._gradpinfo else None)

            # Substitutions for expressions (primitives + grads + time)
            subs = dict(zip(pnames, psolns), t=tcurr)
            if self._gradpinfo and pgrads is not None:
                for pname, pidx in self._gradpinfo:
                    if pgrads[pidx] is None:
                        continue
                    for dim, grad in zip('xyz', pgrads[pidx]):
                        subs[f'grad_{pname}_{dim}'] = grad

            items = []

            # User field expressions
            for field, comps in self._exprs:
                arr = np.stack([npeval(c, subs) for c in comps], axis=-1)
                items.append((self._field_name(sname, field), arr))

            # Derived-field outputs from this step's runner
            if runner := self._field_runners.get(sname):
                for fname in runner.fields(public_only=True):
                    if (etype, fname) in sample.fields:
                        arr = sample.fields[(etype, fname)]
                        items.append((self._field_name(sname, fname),
                                      np.atleast_3d(arr)))

            out[sname][etype] = items

        return out

    def publish(self, fields):
        # Push the per-step field arrays into the Conduit node.  Each item
        # is (fname, arr) and arr's shape depends on clean/raw — _emit_field
        # handles the layout.
        for sname, by_etype in fields.items():
            for etype, items in by_etype.items():
                dom = self.dinfo[sname, etype]
                for fname, arr in items:
                    self._emit_field(self.mesh_n, dom, fname, arr)
