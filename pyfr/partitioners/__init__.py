from pyfr.partitioners.base import BasePartitioner, write_partitioning
from pyfr.partitioners.kahip import KaHIPPartitioner
from pyfr.partitioners.metis import METISPartitioner
from pyfr.partitioners.reconstruct import reconstruct_partitioning
from pyfr.partitioners.scotch import SCOTCHPartitioner
from pyfr.util import subclass_where


def get_partitioner(name, *args, **kwargs):
    return subclass_where(BasePartitioner, name=name)(*args, **kwargs)
