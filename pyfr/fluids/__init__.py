from pyfr.fluids.base import BaseFluid
from pyfr.fluids.cpg import CPGFluid
from pyfr.util import subclass_where


def get_fluid(cfg, ndims):
    name = cfg.get('solver', 'eos', 'cpg')

    return subclass_where(BaseFluid, name=name)(cfg, ndims)
