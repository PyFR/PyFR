# -*- coding: utf-8 -*-

from pyfr.plugins.base import BasePlugin
from pyfr.plugins.dtstats import DtStatsPlugin
from pyfr.plugins.fluidforce import FluidForcePlugin
from pyfr.plugins.nancheck import NaNCheckPlugin
from pyfr.plugins.pseudostats import PseudoStatsPlugin
from pyfr.plugins.residual import ResidualPlugin
from pyfr.plugins.sampler import SamplerPlugin
from pyfr.plugins.tavg import TavgPlugin
from pyfr.plugins.writer import WriterPlugin
from pyfr.util import subclass_where


def get_plugin(name, *args, **kwargs):
    return subclass_where(BasePlugin, name=name)(*args, **kwargs)
