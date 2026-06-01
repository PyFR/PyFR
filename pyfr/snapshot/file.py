import numpy as np

from pyfr.readers.native import NativeReader
from pyfr.snapshot.base import Snapshot
from pyfr.util import subclass_where


class FileSnapshot(Snapshot):
    # Abstract base for file-backed snapshots — wraps a NativeSoln (mesh + soln
    # pair).  Concrete subclasses (SolnSnapshot / StatsSnapshot) supply form-
    # specific transforms + field-registry layouts.  Two factory classmethods
    # are the canonical entry points; direct construction also works when the
    # caller knows the concrete subclass (e.g. writers that already have a
    # populated dataprefix).
    #
    # Form-specific overrides (to_pris, to_grad_pris, fields) raise on this
    # abstract base — instantiate via the classmethods below or a concrete
    # subclass.
    def __init__(self, *, mesh, soln, meshf=None, pname=None):
        # Inline cycle breaker: pyfr.solvers.base transitively pulls in
        # pyfr.integrators → pyfr.plugins → pyfr.snapshot.  Hoisting would
        # cycle during the snapshot package's own initialisation.
        from pyfr.solvers.base import BaseSystem

        # File paths optional — kept for surface()'s lazy bcon re-read when
        # the caller can point at the source .pyfrm.
        self._meshf, self._pname = meshf, pname
        self._soln = soln

        cfg = soln.config
        stats = soln.stats
        prec = cfg.get('backend', 'precision', 'single')
        syscls = subclass_where(BaseSystem,
                                name=cfg.get('solver', 'system'))

        self.mesh = mesh
        self.config = cfg
        self.stats = stats
        self.elementscls = syscls.elementscls
        self.ele_types = list(soln.data)
        self.dtype = np.float32 if prec == 'single' else np.float64
        self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        self.cycle = stats.getint('solver-time-integrator', 'nacptsteps', 0)
        self.has_grads = bool(soln.grad_data)
        self.state = soln.state
        # Stored field names as they appear in the file — convars for soln-
        # prefix files, tavg/residual names for those.
        self.stored_fields = list(soln.fields)

    @classmethod
    def from_file(cls, meshf, solnf, pname=None, construct_con=False):
        # Open a .pyfrm + .pyfrs pair from disk and dispatch to the right
        # concrete subclass based on the soln's data/prefix.  The single
        # entry point where prefix → class mapping lives.
        reader = NativeReader(meshf, pname, construct_con=construct_con)
        mesh, soln = reader.load_subset_mesh_soln(solnf)
        return cls.from_loaded(mesh, soln, meshf=meshf, pname=pname)

    @classmethod
    def from_loaded(cls, mesh, soln, *, meshf=None, pname=None):
        # Wrap an already-loaded mesh + soln pair, dispatching on data/prefix.
        # Used by consumers that opened the .pyfrs themselves (writers with
        # their own NativeReader for raw-HDF5 access).
        prefix = soln.stats.get('data', 'prefix')
        target = SolnSnapshot if prefix == 'soln' else StatsSnapshot
        return target(mesh=mesh, soln=soln, meshf=meshf, pname=pname)

    def soln(self, etype):
        return self._soln.data[etype]

    def grad_soln(self, etype):
        return self._soln.grad_data.get(etype)

    def aux(self, etype):
        return self._soln.aux.get(etype, {})

    def aux_info(self, etype):
        # File-based aux is already in memory; reading .shape / .dtype is
        # free — no getters, no MPI.  Returned shapes drop the leading neles
        # axis to match the IntgSnapshot convention (per-element shape only).
        return {name: (arr.shape[1:], arr.dtype)
                for name, arr in self._soln.aux.get(etype, {}).items()}

    def surface(self, name, divisor=None, refpts_fn=None, *, clean=True):
        # Boundary connectivity isn't built for cheap volume access; build it
        # lazily on first surface request — only possible when meshf was
        # supplied at construction.
        if not self.mesh.bcon and self._meshf:
            self.mesh = NativeReader(self._meshf, self._pname,
                                     construct_con=True).mesh

        return super().surface(name, divisor, refpts_fn, clean=clean)


# Forward refs for the classmethod dispatch — defined in sibling modules.
# Import at module bottom (not top) to avoid a circular import chain through
# soln.py / stats.py, which depend on FileSnapshot above.
from pyfr.snapshot.soln import SolnSnapshot  # noqa: E402
from pyfr.snapshot.stats import StatsSnapshot  # noqa: E402
