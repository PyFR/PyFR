from dataclasses import dataclass, field as dc_field
from functools import cached_property

import numpy as np

from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


def _interp(op, arr):
    # op (m, k); arr (k, ...) -> (m, ...).  Shared interpolation primitive used
    # by all region kinds — lifted here so region.py doesn't need to import
    # private helpers from sibling files.
    return (op @ arr.reshape(arr.shape[0], -1)).reshape(op.shape[0],
                                                        *arr.shape[1:])


@dataclass(frozen=True)
class FieldInfo:
    # Metadata describing one emittable field.  Snap owns the base registry
    # (primitives, gradients, aux); samples extend it with provider outputs.
    # Consumers (writers, samplers, future sinks) iterate this rather than
    # rebuilding their own field-shape conventions.
    name: str                                # public name: 'rho', 'velocity',
                                             # 'grad rho', 'artvisc-vtx', 'mach'
    kind: str                                # 'point' or 'cell'
    ncomps: int                              # 1 = scalar; ndims = vector;
                                             # ndims² = tensor
    dtype: object = None                     # np.dtype; None = snap.dtype
    source: str = 'unknown'                  # 'primitive' / 'gradient' /
                                             # 'aux' / 'provider'
    components: tuple = dc_field(default_factory=tuple)
                                             # ('u','v','w') for velocity;
                                             # ('Ma',) for mach; etc.


class Snapshot:
    # A coherent (mesh + solution + config + state) view of a dataset.
    # Concrete subclasses (selected by `name` via get_snapshot_cls):
    #   IntgSnapshot   name='intg'  — wraps a running integrator
    #   SolnSnapshot   name='soln'  — wraps a .pyfrs with data/prefix='soln'
    #   StatsSnapshot  name='stats' — wraps tavg/residual etc. (stored form;
    #                                  no con_to_pri transform, no providers)
    #
    # Static surface every concrete class exposes as plain attributes:
    #   mesh, config, elementscls, ele_types, dtype, has_grads
    #
    # Per-step surface (live: passthrough to intg; file: set once on load):
    #   tcurr, cycle, state
    #
    # Solution access methods:
    #   soln(etype)      -> (nupts, nvars, neles)        conservative or stored
    #   grad_soln(etype) -> (ndims, nupts, nvars, neles) or None
    #
    # Form-dependent transform hooks (region uses these polymorphically):
    #   to_pris(interp_data, cfg)       -> list of (nrefpts, neles) arrays
    #   to_grad_pris(d, cg, cfg)        -> list of (ndims, nrefpts, neles) | None
    #   pris_names                      -> list of component names indexing pris
    #
    # Capability flag (set on the class):
    #   supports_providers — True for soln-form, False for stored-form

    # Subclasses set this for subclass_where lookup
    name = None

    # Default capability: only soln-form classes opt in
    supports_providers = False

    @property
    def ndims(self):
        return self.mesh.ndims

    # --- region factories (geometry-only; build a sample from the region) ---
    #
    # `spec` accepts three forms:
    #   '*'                            — all etypes, all elements
    #   ['hex', 'tet', ...]            — listed etypes, all elements
    #   {'hex': eidxs_array, ...}      — listed etypes, subset by eidxs
    # The third form is what `region_data()` (in pyfr.plugins.common) produces
    # from a geometric region expression (`box(...)`, `sphere(...)`, etc.),
    # enabling geometric subsetting in vis/render/sample.

    def region(self, spec='*', refpts_fn=None):
        # Generic: a region sampled at reference points chosen by refpts_fn.
        # Default refpts_fn returns the element's native solution points.
        from pyfr.snapshot.region import VolumeSnapshotRegion
        rfn = refpts_fn or (lambda sc, sh: sh.upts)
        return VolumeSnapshotRegion(self, self._resolve_spec(spec), rfn)

    def vis(self, spec='*', divisor=None, *, clean=True):
        # A vis-derived region: sample points are the subdivision of each
        # element at `divisor`.  clean=True (default) deduplicates coincident
        # sub-points and averages primitives/gradients at shared positions.
        from pyfr.snapshot.region import VolumeSnapshotRegion
        div = divisor if divisor is not None else self.config.getint('solver',
                                                                     'order')
        refpts_fn = lambda sc, sh: sc.std_ele(div)
        return VolumeSnapshotRegion(self, self._resolve_spec(spec), refpts_fn,
                                    divisor=div, clean=clean)

    def _resolve_spec(self, spec):
        # Normalise spec to {etype: eidxs|None}.  '*' expands to all etypes;
        # list[str] becomes {et: None}; dict pass-through.
        if spec == '*':
            return {et: None for et in self.ele_types}
        if isinstance(spec, dict):
            return spec
        return {et: None for et in spec}

    def surface(self, name, divisor=None, refpts_fn=None, *, clean=True):
        # A boundary region: faces on the named boundary, plus geometric
        # normals + wall distance.  `name` may be a single bcname or a list
        # (faces from each are concatenated by itype).
        from pyfr.snapshot.region import SurfaceSnapshotRegion
        div = divisor if divisor is not None else self.config.getint('solver',
                                                                     'order')
        return SurfaceSnapshotRegion(self, name, div, refpts_fn, clean=clean)

    def at_points(self, ppts):
        # A region sampled at arbitrary physical points (PointLocator finds the
        # owning element + reference coords; PointSampler evaluates soln/grad
        # there).  MPI-aware: PointSampler gathers on root.
        from pyfr.snapshot.region import PointsSnapshotRegion
        return PointsSnapshotRegion(self, ppts)

    # Subclasses implement aux(etype) -> dict[name, ndarray].  Layout: the
    # array's first axis is `neles`; remaining shape varies per field
    # (scalar at upts: (neles, nupts); vector at upts: (neles, nupts, ncomp);
    # cell scalar: (neles,)).
    def aux(self, etype):
        return {}

    # Metadata-only counterpart of aux(): returns {name: (per_ele_shape,
    # dtype)} without invoking any getters.  Used by snap.fields to classify
    # aux fields by shape — calling aux() there would fire all backend
    # getters every render step purely to read arr.shape.  Subclasses
    # override; default empty dict.
    def aux_info(self, etype):
        return {}

    # Form-dependent transform hooks.  Subclasses override.
    @property
    def pris_names(self):
        # Default: privars (soln-form snaps).  StatsSnapshot overrides to
        # return stored_fields verbatim.
        return list(self.elementscls.privars(self.ndims, self.config))

    def to_pris(self, interp_data, cfg):
        # Map interpolated stored data → list of per-component primitive
        # arrays.  Soln-form: con_to_pri.  Stored-form: identity (the file
        # already holds the values in primitive layout).
        raise NotImplementedError

    def to_grad_pris(self, interp_data, grad_interp, cfg):
        # Map (interpolated soln, interpolated cons-form grads) → list of
        # primitive gradients per axis.  Stored-form returns None.
        raise NotImplementedError

    def compute_grads(self):
        # Trigger the gradient computation needed to make grad_soln(et) safe
        # to index per-etype.  Idempotent — re-calls hit the integrator's
        # internal cache.  File-based snaps no-op (data already loaded).
        #
        # Why this exists: VolumeSnapshotRegion / SurfaceSnapshotRegion iterate
        # `self.etypes` to fetch grads.  A region built from a geometric
        # subset (`region = box(...)`) can leave some ranks with an empty
        # etypes list — those ranks then skip the per-etype grad_soln calls
        # entirely.  The underlying compute is a COLLECTIVE MPI operation
        # (face-coupled gradient exchange), so a partial set of participating
        # ranks deadlocks.  Consumers (renderer, writer) call snap.compute_
        # grads() uniformly across all ranks before any per-region sample
        # build to keep MPI in lock-step.
        pass

    @cached_property
    def fields(self):
        # The full registry of emittable fields exposed by this snap:
        # primitives + (soln-form only) gradients + aux per etype.  Sample
        # objects inherit this at construction time; FieldRunner extends with
        # provider FieldInfos after they run.
        #
        # Subclasses may override to change the registry layout (e.g. stats
        # files emit flat scalars).  Default implementation here is the soln-
        # form layout: vector-grouped via visvars + optional grads.
        out = {}
        dtype = np.dtype(self.dtype)

        for name, varnames in self.elementscls.visvars(self.ndims,
                                                       self.config).items():
            out[name] = FieldInfo(name=name, kind='point',
                                  ncomps=len(varnames), dtype=dtype,
                                  source='primitive',
                                  components=tuple(varnames))

        if self.has_grads:
            for name, info in list(out.items()):
                gname = f'grad {name}'
                gcomps = tuple(f'{c}-{d}' for c in info.components
                               for d in range(self.ndims))
                out[gname] = FieldInfo(name=gname, kind='point',
                                       ncomps=len(gcomps), dtype=dtype,
                                       source='gradient',
                                       components=gcomps)

        self._append_aux_fields(out)
        return out

    def _append_aux_fields(self, out):
        # Aux fields classified by shape (point at upts/verts, cell otherwise).
        # Shared between soln-form and stored-form registries.
        if self.ele_types:
            et0 = self.ele_types[0]
            shapecls = subclass_where(BaseShape, name=et0)
            sh = shapecls(self.mesh.spts[et0].shape[0], self.config)
            pshapes = {(sh.nupts,), (len(sh.linspts),)}
        else:
            pshapes = set()

        for et in self.ele_types:
            for name, (per_ele, dtype) in self.aux_info(et).items():
                if name in out:
                    continue
                if per_ele in pshapes:
                    info = FieldInfo(name=name, kind='point', ncomps=1,
                                     dtype=dtype, source='aux',
                                     components=(name,))
                elif len(per_ele) > 1 and per_ele[:-1] in pshapes:
                    info = FieldInfo(name=name, kind='point',
                                     ncomps=per_ele[-1], dtype=dtype,
                                     source='aux',
                                     components=tuple(f'{name}-{d}'
                                                      for d in
                                                      range(per_ele[-1])))
                else:
                    ncomps = int(np.prod(per_ele) or 1)
                    info = FieldInfo(name=name, kind='cell', ncomps=ncomps,
                                     dtype=dtype, source='aux',
                                     components=(name,))
                out[name] = info
