# -*- coding: utf-8 -*-

import os
import io

from ConfigParser import SafeConfigParser, NoSectionError, NoOptionError

class Inifile(object):
    def __init__(self, inistr=None):
        self._cp = cp = SafeConfigParser()

        # Preserve case
        cp.optionxform = str

        if inistr:
            cp.readfp(io.BytesIO(inistr))

    @staticmethod
    def load(fname):
        return Inifile(open(fname).read())

    def set(self, section, option, value):
        value = str(value)

        try:
            self._cp.set(section, option, value)
        except NoSectionError:
            self._cp.add_section(section)
            self._cp.set(section, option, value)

    def get(self, section, option, default=None, vars=None):
        try:
            return self._cp.get(section, option, vars=vars)
        except NoOptionError:
            if default is None:
                raise

            self._cp.set(section, option, str(default))
            return self._cp.get(section, option, vars=vars)

    def getpath(self, section, option, default=None, vars=None):
        path = self.get(section, option, default, vars)
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        return path

    def getfloat(self, section, option, default=None):
        return float(self.get(section, option, default))

    def getint(self, section, option, default=None):
        return int(self.get(section, option, default))

    def tostr(self):
        buf = io.BytesIO()
        self._cp.write(buf)
        return buf.getvalue()
