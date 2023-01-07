from pyfr.plugins.ascent import AscentPlugin
from pyfr.plugins.base import BasePlugin
from pyfr.plugins.dtstats import DtStatsPlugin
from pyfr.plugins.fluidforce import FluidForcePlugin
from pyfr.plugins.integrate import IntegratePlugin
from pyfr.plugins.nancheck import NaNCheckPlugin
from pyfr.plugins.pseudostats import PseudoStatsPlugin
from pyfr.plugins.residual import ResidualPlugin
from pyfr.plugins.sampler import SamplerPlugin
from pyfr.util import subclass_where

from pyfr.plugins.pseudodtstats import PseudodtStatsPlugin

 # Order of plugin execution controlled here for writing pyfrs
from pyfr.plugins.rewind         import RewindPlugin
from pyfr.plugins.writer         import WriterPlugin
from pyfr.plugins.tavg           import TavgPlugin
from pyfr.plugins.pseudodtwriter import PseudodtWriterPlugin

 # Order of plugin execution NOT controlled here for writing csv
from pyfr.plugins.optimisationstats    import OptimisationStatsPlugin
from pyfr.plugins.bayesianoptimisation import BayesianOptimisationPlugin
from pyfr.plugins.modifyconfiguration  import ModifyConfigPlugin

def get_plugin(name, *args, **kwargs):
    return subclass_where(BasePlugin, name=name)(*args, **kwargs)
