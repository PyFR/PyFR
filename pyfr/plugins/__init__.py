# -*- coding: utf-8 -*-

from pyfr.plugins.base import BasePlugin
from pyfr.plugins.nancheck import NaNCheckPlugin
from pyfr.plugins.sampler import SamplerPlugin
from pyfr.util import subclass_where


def get_plugin(name, *args, **kwargs):
    return subclass_where(BasePlugin, name=name)(*args, **kwargs)
