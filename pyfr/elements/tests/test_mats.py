
from pyfr.backends.dummy import DummyBackend
from pyfr.elements.tensorprod import Hexahedra

from ConfigParser import SafeConfigParser
from io import BytesIO

import numpy as np
import pkgutil


def test_hex_gleg_ord2_csd():
    # Config for a third order spectral difference scheme
    cfg = SafeConfigParser()
    cfg.add_section('scheme')
    cfg.set('scheme', 'order', '3')
    cfg.set('scheme', 'eta', 'sd')

    # Hexahedra elements
    hexes = Hexahedra(DummyBackend(), np.random.randn(8, 20, 3), 1, cfg)

    # Load and import the reference values
    fobj = BytesIO(pkgutil.get_data(__name__, 'hex-gleg-ord3-csd.npz'))
    refm = np.load(fobj)

    assert np.allclose(refm['m0'], hexes._m0.get())
    assert np.allclose(refm['m1'], hexes._m1.get())
    assert np.allclose(refm['m2'], hexes._m2.get())
    assert np.allclose(refm['m3'], hexes._m3.get())
