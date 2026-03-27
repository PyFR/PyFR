from pyfr.plugins.base import BaseCLIPlugin, BasePlugin
from pyfr.plugins.cli.ascent import AscentCLIPlugin
from pyfr.plugins.cli.mesh import MeshCLIPlugin
from pyfr.plugins.cli.sampler import SamplerCLIPlugin
from pyfr.plugins.cli.tavg import TavgCLIPlugin
from pyfr.plugins.soln.ascent import AscentPlugin
from pyfr.plugins.soln.base import BaseSolnPlugin
from pyfr.plugins.soln.dtstats import DtStatsPlugin
from pyfr.plugins.soln.fluidforce import FluidForcePlugin
from pyfr.plugins.soln.fwh import FWHPlugin
from pyfr.plugins.soln.integrate import IntegratePlugin
from pyfr.plugins.soln.nancheck import NaNCheckPlugin
from pyfr.plugins.soln.residual import ResidualPlugin
from pyfr.plugins.soln.sampler import SamplerPlugin
from pyfr.plugins.soln.tavg import TavgPlugin
from pyfr.plugins.soln.writer import WriterPlugin
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.plugins.solver.source import SourcePlugin
from pyfr.plugins.solver.turbulence import TurbulencePlugin
from pyfr.util import subclass_where


def get_plugin(prefix, name, *args, **kwargs):
    cls = subclass_where(BasePlugin, prefix=prefix, name=name)
    return cls(*args, **kwargs)
