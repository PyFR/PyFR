from pyfr.snapshot.base import FieldInfo, Snapshot
from pyfr.snapshot.file import FileSnapshot
from pyfr.snapshot.intg import IntgSnapshot
from pyfr.snapshot.region import (BaseSnapshotRegion, PointsSnapshotRegion,
                                  SurfaceSnapshotRegion, VolumeSnapshotRegion)
from pyfr.snapshot.sample import SnapshotSample
from pyfr.snapshot.soln import SolnSnapshot
from pyfr.snapshot.stats import StatsSnapshot
from pyfr.util import subclass_where


__all__ = [
    'FieldInfo', 'Snapshot',
    'IntgSnapshot', 'SolnSnapshot', 'StatsSnapshot', 'FileSnapshot',
    'BaseSnapshotRegion', 'VolumeSnapshotRegion', 'SurfaceSnapshotRegion',
    'PointsSnapshotRegion',
    'SnapshotSample',
    'get_snapshot_cls',
]


def get_snapshot_cls(name):
    # Registry lookup: resolves a snapshot kind name ('intg', 'soln', 'stats',
    # ...) to its class via PyFR's standard subclass_where dispatch.  Each
    # Snapshot subclass declares `name = 'xxx'`; new kinds slot in by adding
    # a subclass.
    return subclass_where(Snapshot, name=name)
