# -*- coding: utf-8 -*-

import pkgutil
from io import BytesIO

import numpy as np

from pyfr.bases.tensorprod import HexBasis
from pyfr.elements import Elements
from pyfr.inifile import Inifile

def test_hex_gleg_ord3_csd():
    # Config for a third order spectral difference scheme
    cfg = Inifile()
    cfg.set('mesh-elements', 'quad-rule', 'gauss-legendre')
    cfg.set('mesh-elements', 'order', '3')
    cfg.set('mesh-elements', 'vcjh-eta', 'sd')

    # Generate the hexes
    hexes = Elements(HexBasis, np.random.randn(8, 20, 3), cfg)

    # Load and import the reference values
    fobj = BytesIO(pkgutil.get_data(__name__, 'hex-gleg-ord3-csd.npz'))
    refm = np.load(fobj)

    assert np.allclose(refm['m0'], hexes._m0)
    assert np.allclose(refm['m1'], hexes._m1)
    assert np.allclose(refm['m2'], hexes._m2)
    assert np.allclose(refm['m3'], hexes._m3)
