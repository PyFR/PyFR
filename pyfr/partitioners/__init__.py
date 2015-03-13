# -*- coding: utf-8 -*-

from pyfr.partitioners.base import BasePartitioner
from pyfr.partitioners.metis import METISPartitioner
from pyfr.partitioners.scotch import SCOTCHPartitioner
from pyfr.util import subclass_where


def get_partitioner(name, *args, **kwargs):
    return subclass_where(BasePartitioner, name=name)(*args, **kwargs)
