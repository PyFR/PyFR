from pyfr.snapshot.base import Snapshot


class IntgSnapshot(Snapshot):
    # Wraps a running integrator.  Built transiently — the plugin's
    # __call__(intg) creates a fresh IntgSnapshot per step and drops it.  No
    # intg reference is retained by anything else (renderer/region hold their
    # own static metadata; the snap is passed into render(snap) per step).
    name = 'intg'
    supports_providers = True

    def __init__(self, *, intg, export_fields=None):
        sys = intg.system

        # Static surface
        self.mesh = sys.mesh
        self.config = intg.cfg
        self.elementscls = sys.elementscls
        self.ele_types = list(sys.ele_types)
        self.dtype = sys.backend.fpdtype
        self.has_grads = True

        # Per-step surface (read fresh from intg each access)
        self._intg = intg

        # Pre-captured export_fields (per-etype list of ExportableField); the
        # plugin captures these in its __init__ while ele_map is still alive
        # (system.commit() runs after plugin construction and frees ele_map).
        # Defaults to empty — snap.aux() returns {} for snaps without a
        # capturing plugin.
        self._export_fields = export_fields or {}

    @property
    def tcurr(self):
        return self._intg.tcurr

    @property
    def cycle(self):
        return self._intg.nacptsteps

    @property
    def state(self):
        # Live serialisable plugin/kinematic records, keyed by sprefix
        return {p.sprefix: p._serialise_data()
                for p in self._intg.plugins
                if getattr(p, 'sprefix', None)
                and getattr(p, '_serialise_data', None) is not None}

    def soln(self, etype):
        return self._intg.soln[self.ele_types.index(etype)]

    def grad_soln(self, etype):
        # intg.grad_soln is already (ndims, nupts, nvars, neles) per etype,
        # matching SolnSnapshot.grad_soln — return directly.
        return self._intg.grad_soln[self.ele_types.index(etype)]

    def aux(self, etype):
        # Live aux fields: each export_field's getter returns the full
        # (neles, *shape) array on host.  Reads from the pre-captured registry
        # (passed in by the plugin at __init__ time, when ele_map was alive).
        # NB: this fires backend kernels / host pulls.  Use aux_info() if you
        # only need shape/dtype.
        return {ef.name: ef.getter()
                for ef in self._export_fields.get(etype, ())}

    def aux_info(self, etype):
        # Metadata-only — reads ExportableField.shape + dtype on the captured
        # registry.  No getter calls, no MPI, no host pull.  Used by snap.
        # fields to classify aux without touching backend state every render.
        return {ef.name: (ef.shape, ef.dtype or self.dtype)
                for ef in self._export_fields.get(etype, ())}

    # Soln-form transforms — conservative input → primitive output.
    def to_pris(self, interp_data, cfg):
        return list(self.elementscls.con_to_pri(interp_data, cfg))

    def to_grad_pris(self, interp_data, grad_interp, cfg):
        return list(self.elementscls.grad_con_to_pri(interp_data, grad_interp,
                                                     cfg))

    def compute_grads(self):
        # Delegate to the integrator's idempotent compute_grads() wrapper —
        # this fires the MPI face-coupled gradient exchange (cached via
        # intg._grads_current).  We deliberately call the bare compute method
        # (not intg.grad_soln) to avoid the eager [e.get() for e in ...] host
        # pull; that pull happens lazily per-rank when snap.grad_soln(et) is
        # indexed inside region._compute_sample.  Ranks with empty etypes
        # skip the index — no host pull, no MPI risk.
        if self.has_grads:
            self._intg.compute_grads()
