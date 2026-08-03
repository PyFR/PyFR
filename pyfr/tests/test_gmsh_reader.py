"""GmshReader physical-group handling regression tests.

gmsh namespaces physical-group IDs per dimension -- a physical group is a
(dim, tag) pair. PyFR previously flattened these into one global namespace,
rejecting valid meshes where a surface/curve group and a volume group share a
numeric ID ('Duplicate physical entity ID'). The dup* fixtures contain exactly
that collision; the uniq* fixtures are the same geometry with unique IDs. The
reader must import the dup meshes and classify boundary/volume groups by
dimension, producing output byte-identical to the unique-ID twins.
"""
import os

from pyfr.progress import NullProgressSequence
from pyfr.readers import get_reader_by_name


def _read(name):
    fname = os.path.join(os.path.dirname(__file__), name)
    return get_reader_by_name('gmsh', fname, NullProgressSequence())


def test_gmsh_dup_physical_ids_3d_v41():
    r = _read('dup4.1.msh')
    assert r._volpents == {'vol': 1}
    assert r._bfacespents['wall'] == (2, 1)
    assert r._bfacespents['outlet'] == (2, 2)
    assert r._bfacespents['bottom'] == (2, 3)


def test_gmsh_dup_physical_ids_2d():
    r = _read('q2d_full_dup.msh')
    assert r._volpents == {'vol': 1}
    assert r._bfacespents['inlet'] == (1, 1)
    assert r._bfacespents['outlet'] == (1, 2)


def test_gmsh_dup_physical_ids_import(tmp_path):
    # The crashing meshes must round-trip through the import pipeline and
    # produce output byte-identical to their unique-ID twins
    for dup, uniq in (('dup4.1.msh', 'uniq4.1.msh'),
                      ('q2d_full_dup.msh', 'q2d_full_uniq.msh')):
        dup_out = tmp_path / 'dup.pyfrm'
        uniq_out = tmp_path / 'uniq.pyfrm'
        _read(dup).write(str(dup_out), 1e-8)
        _read(uniq).write(str(uniq_out), 1e-8)
        assert dup_out.read_bytes() == uniq_out.read_bytes()
