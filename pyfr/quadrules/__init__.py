# -*- coding: utf-8 -*-

import re

from pyfr.quadrules.base import BaseQuadRule, BaseTabulatedQuadRule
from pyfr.quadrules.line import BaseLineQuadRule
from pyfr.quadrules.tet import BaseTetQuadRule
from pyfr.quadrules.tri import BaseTriQuadRule
from pyfr.util import subclass_map


def get_quadrule(eletype, rule, npts):
    eletype_map = {
        'line': BaseLineQuadRule,
        'tet': BaseTetQuadRule,
        'tri': BaseTriQuadRule
    }

    # Obtain the base quadrature rule class for this element type
    basecls = eletype_map[eletype]

    # See if rule looks like the name of a scheme
    if re.match(r'[a-zA-Z0-9\-+_]+$', rule):
        rule_map = subclass_map(basecls, 'name')
        return rule_map[rule](npts)
    # Otherwise see if it looks like a tabulation
    elif 'PTS' in rule.upper():
        # Create a suitable subclass
        rulecls = type(eletype, (BaseTabulatedQuadRule, basecls), {})

        # Instantiate and validate
        r = rulecls(rule)
        if len(r.points) != npts:
            raise ValueError('Invalid number of points for quad rule')

        return r
    # Invalid
    else:
        raise ValueError('Invalid quadrature rule')
