from pyfr.util import subclass_where


def get_shock_capturing(backend, cfg, system):
    scname = cfg.get('solver', 'shock-capturing', 'none')
    sccls = subclass_where(BaseShockCapturing, scheme=scname)

    # Refuse schemes which need solution gradients the system lacks
    if sccls.needs_grad_soln and not system.elementscls.has_grad_soln:
        raise ValueError(f'{scname} requires solution gradients')

    # Let the scheme substitute another scheme for itself
    sccls = sccls.resolve(cfg, system)

    return sccls(backend, cfg, system)


class BaseShockCapturing:
    scheme = None

    # If the scheme requires the gradient of the solution
    needs_grad_soln = False

    # If the scheme works with error estimating controllers
    supports_errest = True

    # If the scheme works with implicit time stepping
    supports_implicit = True

    # Extra kernel providers and MPI request providers for the system
    kern_parts = {}
    mpi_parts = []

    def __init__(self, backend=None, cfg=None, system=None):
        pass

    @classmethod
    def resolve(cls, cfg, system):
        # Scheme class to instantiate for this configuration and system
        return cls

    @property
    def tplargs(self):
        # Template arguments for the flux kernels
        return {'shock_capturing': self.scheme}

    def flux_args(self, eles, rgn, pts):
        return {}

    def comm_flux_args(self, inters):
        return {}

    def add_to_graph_pre_recv(self, g, k, m):
        pass

    def add_to_graph_post_recv(self, g, k, deps):
        pass

    def preproc_graphs(self, be, k, m, deps):
        return ()

    def postproc(self, be, k):
        pass


class NullShockCapturing(BaseShockCapturing):
    scheme = 'none'


class ShockCapturingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # By default disable shock capturing
        self.shock_capturing = NullShockCapturing()

    def set_shock_capturing(self, sc):
        if not isinstance(self.shock_capturing, NullShockCapturing):
            raise RuntimeError('Shock capturing scheme already set')

        self.shock_capturing = sc
