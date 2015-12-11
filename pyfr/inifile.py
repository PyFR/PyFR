# -*- coding: utf-8 -*-

from collections import OrderedDict
from configparser import SafeConfigParser, NoSectionError, NoOptionError
import io
import os
import re


def _ensure_float(m):
    m = m.group(0)
    return m if any(c in m for c in '.eE') else m + '.'


_sentinel = object()


class Inifile(object):
    def __init__(self, inistr=None):
        self._cp = cp = SafeConfigParser(inline_comment_prefixes=[';'])

        # Preserve case
        cp.optionxform = str

        if inistr:
            cp.read_string(inistr)

    @staticmethod
    def load(file):
        if isinstance(file, str):
            file = open(file)

        return Inifile(file.read())

    def set(self, section, option, value):
        value = str(value)

        try:
            self._cp.set(section, option, value)
        except NoSectionError:
            self._cp.add_section(section)
            self._cp.set(section, option, value)

    def hasopt(self, section, option):
        return self._cp.has_option(section, option)

    def get(self, section, option, default=_sentinel, vars=None):
        try:
            val = self._cp.get(section, option, vars=vars)
        except NoSectionError:
            if default is _sentinel:
                raise

            self._cp.add_section(section)
            val = self.get(section, option, default, vars)
        except NoOptionError:
            if default is _sentinel:
                raise

            self._cp.set(section, option, str(default))
            val = self._cp.get(section, option, vars=vars)

        return os.path.expandvars(val)

    def getpath(self, section, option, default=_sentinel, vars=None, abs=True):
        path = self.get(section, option, default, vars)
        path = os.path.expanduser(path)

        if abs:
            path = os.path.abspath(path)

        return path

    def getexpr(self, section, option, default=_sentinel, subs={}):
        expr = self.get(section, option, default)

        # Ensure the expression does not contain invalid characters
        if not re.match(r'[A-Za-z0-9 \t\n\r.,+\-*/%()]+$', expr):
            raise ValueError('Invalid characters in expression')

        # Substitute variables
        if subs:
            expr = re.sub(r'\b({0})\b'.format('|'.join(subs)),
                          lambda m: subs[m.group(1)], expr)

        # Convert integers to floats
        expr = re.sub(r'\b((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?(?!\s*])',
                      _ensure_float, expr)

        # Encase in parenthesis
        return '({0})'.format(expr)

    def getfloat(self, section, option, default=_sentinel):
        return float(self.get(section, option, default))

    def getint(self, section, option, default=_sentinel):
        return int(self.get(section, option, default))

    def items(self, section):
        return OrderedDict(self._cp.items(section))

    def items_as(self, section, type):
        iv = []

        for k, v in self._cp.items(section):
            try:
                iv.append((k, type(v)))
            except ValueError:
                pass

        return OrderedDict(iv)

    _bool_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                    '0': False, 'no': False, 'false': False, 'off': False}

    def getbool(self, section, option, default=_sentinel):
        v = self.get(section, option, default)
        return self._bool_states[v.lower()]

    def sections(self):
        return self._cp.sections()

    def tostr(self):
        buf = io.StringIO()
        self._cp.write(buf)
        return buf.getvalue()
