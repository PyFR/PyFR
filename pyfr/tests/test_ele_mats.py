# -*- coding: utf-8 -*-

import pkgutil
from io import BytesIO

import numpy as np

from pyfr.bases.tensorprod import HexBasis
from pyfr.elements import EulerElements
from pyfr.inifile import Inifile

def test_hex_gleg_ord3_csd():
    # Config for a third order spectral difference scheme
    cfg = Inifile()
    cfg.set('mesh-elements', 'quad-rule', 'gauss-legendre')
    cfg.set('mesh-elements', 'order', '3')
    cfg.set('mesh-elements', 'vcjh-eta', 'sd')

    # Generate the hexes
    hexes = EulerElements(HexBasis, np.random.randn(8, 20, 3), cfg)

    # Load and import the reference values
    fobj = BytesIO(pkgutil.get_data(__name__, 'hex-gleg-ord3-csd.npz'))
    refm = np.load(fobj)

    assert np.allclose(refm['m0'], np.asanyarray(hexes.m0, dtype=np.float))
    assert np.allclose(refm['m1'], np.asanyarray(hexes.m1, dtype=np.float))
    assert np.allclose(refm['m2'], np.asanyarray(hexes.m2, dtype=np.float))
    assert np.allclose(refm['m3'], np.asanyarray(hexes.m3, dtype=np.float))
