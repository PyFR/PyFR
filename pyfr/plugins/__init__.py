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

from pyfr.plugins.rewind         import RewindPlugin # Order of plugin execution controlled here
from pyfr.plugins.writer         import WriterPlugin # We need writer plugin after rewinding
from pyfr.plugins.tavg           import TavgPlugin   # We need  tavg  plugin after rewinding
from pyfr.plugins.pseudodtwriter import PseudodtWriterPlugin

from pyfr.plugins.optimisationstats import OptimisationStatsPlugin


def get_plugin(name, *args, **kwargs):
    return subclass_where(BasePlugin, name=name)(*args, **kwargs)
