# -*- coding: utf-8 -*-

from pyfr.quadrules.base import BaseQuadRule
from pyfr.quadrules.line import BaseLineQuadRule
from pyfr.quadrules.tri import BaseTriQuadRule
from pyfr.util import subclass_map


def get_quadrule(basecls, name, npts):
    rule_map = subclass_map(basecls, 'name')
    
    return rule_map[name](npts)
