from collections import defaultdict
import re

import numpy as np

from pyfr.inifile import process_expr
from pyfr.mpiutil import get_comm_rank_root
from pyfr.nputil import npeval
from pyfr.plugins.common import region_data
from pyfr.plugins.fields.runner import FieldRunner
from pyfr.plugins.soln.insitu.conduit import ConduitNode, ConduitWrappers
from pyfr.snapshot.region import (SurfaceSnapshotRegion, VolumeSnapshotRegion)
from pyfr.util import paren_depths
from pyfr.writers.vtk.shapes import get_vtk_shape


def split_components(expr):
    # Split on top-level commas; respect bracket/paren/brace nesting.
    # Used for parsing `field-X = expr1, expr2, ...` user directives.
    parts, buf = [], []
    for c, d in paren_depths(expr):
        if c == ',' and d == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(c)

    parts.append(''.join(buf).strip())

    return parts


class InSituRenderer:
    # Conduit blueprint element name mapping
    bp_emap = {'hex': 'hex', 'pri': 'wedge', 'pyr': 'pyramid', 'quad': 'quad',
               'tet': 'tet', 'tri': 'tri'}

    def __init__(self, mesh, scfg, cfgsect, isrestart, *, acfg=None):
        comm, _, _ = get_comm_rank_root()

        self.mesh = mesh
        self.scfg = scfg
        self.acfg = acfg if acfg is not None else scfg

        self.cfgsect = cfgsect
        self.isrestart = isrestart

        sorder = scfg.getint('solver', 'order')
        self.divisor = self.acfg.getint(cfgsect, 'division', sorder)
        self.clean = self.acfg.getbool(cfgsect, 'clean', True)

        # Named surface sources (one per surface-{name} = <region> entry)
        self.surfaces = {k.removeprefix('surface-'): self.acfg.get(cfgsect, k)
                         for k in self.acfg.items(cfgsect, prefix='surface-')}

        # Volume source: implicit if no surfaces; opt-in alongside surfaces
        self.want_volume = (not self.surfaces or
                            self.acfg.getbool(cfgsect, 'volume', False))

        # Load Conduit (subclass may override for host-specific fallbacks)
        self.conduit = self._load_conduit()

        # Build regions per source
        self.regions = {}
        self._source_kinds = {}
        if self.want_volume:
            spec = region_data(self.acfg, cfgsect, self.mesh)
            div = self.divisor
            refpts_fn = lambda sc, sh: sc.std_ele(div)
            self.regions['volume'] = VolumeSnapshotRegion(
                self.mesh, scfg, spec, refpts_fn,
                divisor=div, clean=self.clean
            )
            self._source_kinds['volume'] = 'volume'
        for sname, sregion in self.surfaces.items():
            self.regions[sname] = SurfaceSnapshotRegion(
                self.mesh, scfg, [sregion], self.divisor,
                clean=self.clean
            )
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
            raise ValueError(f'Fields used but not defined: {missing}')

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

    # Host hooks

    def _load_conduit(self):
        return ConduitWrappers()

    def _init_host_publish(self):
        # Override to wire in scenes/pipelines/etc.
        pass

    def _domain_path(self, sname, domid):
        return f'domain_{domid}'

    def _write_step_state(self, dom, tcurr, cycle):
        self.mesh_n[f'{dom}/state/time/keyword'] = 'Time'
        self.mesh_n[f'{dom}/state/time/data'] = str(tcurr)
        self.mesh_n[f'{dom}/state/cycle'] = cycle

    def _emit_field(self, mesh_n, dom, fname, arr):
        # Scalars publish at values; vectors split into values/x, /y, /z
        path = f'{dom}/fields/{fname}/values'
        if len(comps := arr.T) == 1:
            mesh_n[path] = comps[0]
        else:
            for x, sl in zip('xyz', comps):
                mesh_n[f'{path}/{x}'] = sl

    def _emit_coords(self, mesh_n, dom, cs, xyz):
        for l, x in zip('xyz', xyz):
            mesh_n[f'{dom}/coordsets/{cs}/values/{l}'] = x

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

    def _flatten_coords(self, ploc):
        if ploc.ndim == 2:
            return ploc
        return ploc.transpose(0, 2, 1).reshape(ploc.shape[0], -1)

    def _build_connectivity(self, etype, region):
        subdiv = get_vtk_shape(etype, self.divisor)
        snodes = subdiv.subnodes
        conn = region.connectivity(etype, snodes)
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

        ploc = region.ploc[etype]
        self._emit_coords(mesh_n, dom, cs, self._flatten_coords(ploc))

        self._write_field_meta(mesh_n, dom, sname)

        conn, subdiv, snodes = self._build_connectivity(etype, region)
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
            for sname in self.regions:
                self._register_user_field(sname, field)

            raw = self.acfg.get(self.cfgsect, k)
            comps = [process_expr(c, cons) for c in split_components(raw)]
            self._exprs.append((field, comps))

    def _init_field_runners(self):
        # Parse add-field-{name} = <sources>; one runner per source
        groups = defaultdict(list)
        for k in self.acfg.items(self.cfgsect, prefix='add-field-'):
            name = k.removeprefix('add-field-')
            for s in self.acfg.get(self.cfgsect, k).split(','):
                sname = s.strip()
                if sname not in self.regions:
                    raise ValueError(f'Field provider {name!r}: unknown '
                                     f'source {sname!r}')
                groups[sname].append(name)

        self._field_runners = runners = {}
        for sname, names in groups.items():
            export_type = self._source_kinds[sname]
            runners[sname] = FieldRunner(names, self.mesh.ndims, self.scfg,
                                         export_type=export_type)

            for fname in runners[sname].fields(public_only=True):
                self._register_user_field(sname, fname)

    def _init_gradients(self):
        self._g_pnames_user = pnames = set()
        for _, comps in self._exprs:
            for c in comps:
                pnames.update(re.findall(r'\bgrad_(.+?)_[xyz]\b', c))

        runners = self._field_runners.values()
        self._provider_needs_grads = any(r.needs_grads for r in runners)
        self._gradpinfo = None

    def _resolve_gradpinfo(self, snap):
        if self._gradpinfo is not None:
            return
        privars = snap.pris_names
        g_pnames = set(self._g_pnames_user)
        needs_grads = bool(g_pnames) or self._provider_needs_grads
        if needs_grads and not snap.has_grads:
            raise ValueError('Gradients required but not available')
        if self._provider_needs_grads:
            g_pnames.update(privars)
        self._gradpinfo = [(p, privars.index(p)) for p in g_pnames]

    def _evaluate_exprs(self, snap):
        self._resolve_gradpinfo(snap)

        # Get the primitive variable names
        pnames = snap.pris_names
        tcurr = snap.tcurr
        cycle = snap.cycle

        samples = {sname: region.sample(snap)
                   for sname, region in self.regions.items()}

        # Run field providers on each sample; results land in sample.field_arrays
        for sname, runner in self._field_runners.items():
            runner.run_on_sample(samples[sname], public_only=True)

        # out[sname] = {etype: [(field, arr)]} for per-source publish
        out = defaultdict(dict)

        # Iterate over each (source, etype) pair in our blueprint
        for (sname, etype), dom in self.dinfo.items():
            self._write_step_state(dom, tcurr, cycle)

            sample = samples[sname]

            cs = f'{sname}_coords'
            self._emit_coords(self.mesh_n, dom, cs,
                              self._flatten_coords(sample.ploc[etype]))

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

            # Postproc plugins for this source/etype
            if runner := self._field_runners.get(sname):
                for fname in runner.fields(public_only=True):
                    if (etype, fname) in sample.field_arrays:
                        arr = sample.field_arrays[(etype, fname)]
                        items.append((self._field_name(sname, fname),
                                      np.atleast_3d(arr)))

            out[sname][etype] = items

        return out

    def publish(self, fields):
        for sname, by_etype in fields.items():
            for etype, items in by_etype.items():
                dom = self.dinfo[sname, etype]
                for fname, arr in items:
                    self._emit_field(self.mesh_n, dom, fname, arr)
