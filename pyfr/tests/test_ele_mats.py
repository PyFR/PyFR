# -*- coding: utf-8 -*-

from io import BytesIO
import pkgutil

import numpy as np
import sympy as sy

from pyfr.bases.tensorprod import HexBasis
from pyfr.inifile import Inifile


def test_hex_gleg_ord3_csd():
    # Config for a third order spectral difference scheme
    cfg = Inifile()
    cfg.set('mesh', 'order', '3')
    cfg.set('mesh-elements-hex', 'soln-pts', 'gauss-legendre')
    cfg.set('mesh-elements-hex', 'vcjh-eta', 'sd')

    # Generate the hexes
    hb = HexBasis(sy.symbols('p q r'), None, cfg)

    # Load and import the reference values
    fobj = BytesIO(pkgutil.get_data(__name__, 'hex-gleg-ord3-csd.npz'))
    refm = np.load(fobj)

    assert np.allclose(refm['m0'], np.asanyarray(hb.m0, dtype=np.float))
    assert np.allclose(refm['m1'], np.asanyarray(hb.m1, dtype=np.float))
    assert np.allclose(refm['m2'], np.asanyarray(hb.m2, dtype=np.float))
    assert np.allclose(refm['m3'], np.asanyarray(hb.m3, dtype=np.float))
