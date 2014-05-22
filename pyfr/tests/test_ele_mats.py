# -*- coding: utf-8 -*-

from io import BytesIO
import pkgutil

import numpy as np

from pyfr.inifile import Inifile
from pyfr.shapes import HexShape


def test_hex_gleg_ord3():
    # Config for a third order DG scheme
    cfg = Inifile()
    cfg.set('solver', 'order', '3')
    cfg.set('solver-interfaces-quad', 'flux-pts', 'gauss-legendre')
    cfg.set('solver-elements-hex', 'soln-pts', 'gauss-legendre')

    # Generate the shape
    hs = HexShape(None, cfg)

    # Load and import the reference values
    fobj = BytesIO(pkgutil.get_data(__name__, 'hex-gleg-ord3.npz'))
    refm = np.load(fobj)

    assert np.allclose(refm['m0'], hs.m0)
    assert np.allclose(refm['m1'], hs.m1)
    assert np.allclose(refm['m2'], hs.m2)
    assert np.allclose(refm['m3'], hs.m3)
