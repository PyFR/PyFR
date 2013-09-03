 # -*- coding: utf-8 -*-

import os
import pkgutil

from mako.lookup import TemplateLookup
from mako.template import Template


class PkgTemplateLookup(TemplateLookup):
    def __init__(self, pkg, basedir):
        self.paths = [(pkg, basedir)]

    def add_search_path(self, pkg, basedir):
        self.paths.insert(0, (pkg, basedir))

    def del_search_path(self, pkg, basedir):
        self.paths.remove((pkg, basedir))

    def adjust_uri(self, uri, relto):
        return uri

    def get_template(self, name):
        for pkg, basedir in self.paths:
            try:
                tpl = pkgutil.get_data(pkg, os.path.join(basedir, name))
                return Template(tpl, lookup=self)
            except IOError:
                pass

        raise RuntimeError('Template "{}" not found'.format(name))
