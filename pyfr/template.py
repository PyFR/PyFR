 # -*- coding: utf-8 -*-

import os
import pkgutil

from mako.lookup import TemplateLookup
from mako.template import Template


class DottedTemplateLookup(TemplateLookup):
    def __init__(self, pkg):
        self.dfltpkg = pkg

    def adjust_uri(self, uri, relto):
        return uri

    def get_template(self, name):
        div = name.rfind('.')

        # Break apart name into a package and base file name
        if div >= 0:
            pkg = name[:div]
            basename = name[div + 1:]
        else:
            pkg = self.dfltpkg
            basename = name

        # Attempt to load the template
        src = pkgutil.get_data(pkg, basename + '.mako')
        if not src:
            raise RuntimeError('Template "{}" not found'.format(name))

        return Template(src, lookup=self)
