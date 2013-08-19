 # -*- coding: utf-8 -*-

import os
import pkgutil

from mako.lookup import TemplateLookup
from mako.template import Template


class PkgTemplateLookup(TemplateLookup):
    def __init__(self, pkg, basedir):
        self.pkg = pkg
        self.basedir = basedir

    def adjust_uri(self, uri, relto):
        return uri

    def get_template(self, name):
        tpl = pkgutil.get_data(self.pkg, os.path.join(self.basedir, name))
        return Template(tpl, lookup=self)
