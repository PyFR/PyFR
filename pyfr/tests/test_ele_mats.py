# -*- coding: utf-8 -*-

from io import BytesIO
import pkgutil

import numpy as np

from pyfr.bases.tensorprod import HexBasis
from pyfr.inifile import Inifile


def test_hex_gleg_ord3():
    # Config for a third order DG scheme
    cfg = Inifile()
    cfg.set('solver', 'order', '3')
    cfg.set('solver-interfaces-quad', 'flux-pts', 'gauss-legendre')
    cfg.set('solver-elements-hex', 'soln-pts', 'gauss-legendre')

    # Generate the hexes
    hb = HexBasis(None, cfg)

    # Load and import the reference values
    fobj = BytesIO(pkgutil.get_data(__name__, 'hex-gleg-ord3.npz'))
    refm = np.load(fobj)

    assert np.allclose(refm['m0'], np.asanyarray(hb.m0, dtype=np.float))
    assert np.allclose(refm['m1'], np.asanyarray(hb.m1, dtype=np.float))
    assert np.allclose(refm['m2'], np.asanyarray(hb.m2, dtype=np.float))
    assert np.allclose(refm['m3'], np.asanyarray(hb.m3, dtype=np.float))
