from pyfr.plugins.ascent import AscentPlugin
from pyfr.plugins.base import (BaseCLIPlugin, BasePlugin, BaseSolnPlugin,
                               BaseSolverPlugin)
from pyfr.plugins.dtstats import DtStatsPlugin
from pyfr.plugins.fluidforce import FluidForcePlugin
from pyfr.plugins.fwh import FWHPlugin
from pyfr.plugins.integrate import IntegratePlugin
from pyfr.plugins.nancheck import NaNCheckPlugin
from pyfr.plugins.pseudostats import PseudoStatsPlugin
from pyfr.plugins.pseudodtwriter import PseudodtWriterPlugin
from pyfr.plugins.pseudodtstats import PseudodtStatsPlugin
from pyfr.plugins.residual import ResidualPlugin
from pyfr.plugins.sampler import SamplerPlugin
from pyfr.plugins.source import SourcePlugin
from pyfr.plugins.tavg import TavgCLIPlugin, TavgPlugin
from pyfr.plugins.turbulence import TurbulencePlugin
from pyfr.util import subclass_where

from pyfr.plugins.pseudodtstats import PseudodtStatsPlugin

 # Order of plugin execution controlled here for writing pyfrs
from pyfr.plugins.computetime    import ComputeTimePlugin
from pyfr.plugins.rewind         import RewindPlugin
from pyfr.plugins.writer         import WriterPlugin
from pyfr.plugins.tavg           import TavgPlugin
from pyfr.plugins.pseudodtwriter import PseudodtWriterPlugin

 # Order of plugin execution NOT controlled here for writing csv
from pyfr.plugins.optimisationstats    import OptimisationStatsPlugin
from pyfr.plugins.bayesianoptimisation import BayesianOptimisationPlugin
from pyfr.plugins.modifyconfiguration  import ModifyConfigPlugin

def get_plugin(prefix, name, *args, **kwargs):
    cls = subclass_where(BasePlugin, prefix=prefix, name=name)
    return cls(*args, **kwargs)
