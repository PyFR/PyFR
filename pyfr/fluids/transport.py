class BaseTransport:
    name = None

    # Construction may do arbitrary preprocessing (curve fits, mixture
    # rules) against the fluid's parsed data; do not store the fluid
    def __init__(self, cfg, fluid):
        pass

    def register(self, fluid):
        if self._register_mu(fluid):
            self._register_kappa(fluid)

    def _register_mu(self, fluid):
        return False

    # Prandtl-number closure in cpT units; fluids with a real conductivity
    # model come with their own transport class which overrides this
    def _register_kappa(self, fluid):
        c = fluid.c

        if 'Pr' in c:
            kct = c['gamma'] / c['Pr']
            fluid._register_quantity('kappa', ('mu',), lambda u: f'mu*{kct}')


class ConstantTransport(BaseTransport):
    name = 'constant'

    def _register_mu(self, fluid):
        if 'mu' not in fluid.c:
            return False

        mu = fluid.c['mu']
        fluid._register_quantity('mu', (), lambda u: f'{mu}')

        return True


class SutherlandTransport(BaseTransport):
    name = 'sutherland'

    def _register_mu(self, fluid):
        c = fluid.c
        mu, cpTref, cpTs = c['mu'], c['cpTref'], c['cpTs']

        fluid._register_quantity('Trat', ('cpT',),
                                 lambda u: f'{1/cpTref}*cpT')
        fluid._register_quantity('mu', ('cpT', 'Trat'),
                                 lambda u: f'{mu*(cpTref + cpTs)}*Trat'
                                           f'*sqrt(Trat) / (cpT + {cpTs})')

        return True
