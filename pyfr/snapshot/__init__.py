from pyfr.readers.native import NativeReader
from pyfr.snapshot.fieldinfo import FieldInfo
from pyfr.snapshot.file import FileSnapshot
from pyfr.snapshot.intg import IntgSnapshot
from pyfr.snapshot.region import VolumeSnapshotRegion
from pyfr.snapshot.soln import SolnSnapshot
from pyfr.snapshot.stats import StatsSnapshot


def from_file(meshf, solnf, pname=None, construct_con=False):
    reader = NativeReader(meshf, pname, construct_con=construct_con)
    mesh, soln = reader.load_subset_mesh_soln(solnf)
    return from_loaded(mesh, soln)


def from_loaded(mesh, soln):
    prefix = soln.stats.get('data', 'prefix')
    target = SolnSnapshot if prefix == 'soln' else StatsSnapshot
    return target(mesh, soln)
