from ast import literal_eval
from configparser import ConfigParser, NoSectionError, NoOptionError
import io
import os
from pathlib import Path
import re


def _ensure_float(m):
    m = m[0]
    return m if any(c in m for c in '.eE') else m + '.'


def process_expr(expr, subs={}):
    if not re.match(r'[A-Za-z0-9_ \t\n\r.,+\-*/%()]+$', expr):
        raise ValueError('Invalid characters in expression')

    if subs:
        expr = re.sub(r'\b({0})\b'.format('|'.join(subs)),
                      lambda m: str(subs[m[1]]), expr)

    expr = re.sub(r'\b((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?(?![^[]*\])',
                  _ensure_float, expr)

    return f'({expr})'


_sentinel = object()


class Inifile:
    def __init__(self, inistr=None):
        self._cp = cp = ConfigParser(inline_comment_prefixes=[';', '#'])

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

    def getpath(self, section, option, default=_sentinel, vars=None,
                abs=False):
        path = Path(self.get(section, option, default, vars)).expanduser()

        if abs:
            path = path.absolute()

        return path

    def getexpr(self, section, option, default=_sentinel, subs={}):
        return process_expr(self.get(section, option, default), subs)

    _bool_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                    '0': False, 'no': False, 'false': False, 'off': False}

    def getbool(self, section, option, default=_sentinel):
        v = self.get(section, option, default)
        return self._bool_states[v.lower()]

    def getfloat(self, section, option, default=_sentinel):
        return float(self.get(section, option, default))

    def getint(self, section, option, default=_sentinel):
        return int(self.get(section, option, default))

    def getliteral(self, section, option, default=_sentinel):
        return literal_eval(self.get(section, option, default))

    def items(self, section, prefix=''):
        return self.items_as(section, lambda v: v, prefix=prefix)

    def items_as(self, section, type, prefix=''):
        iv = {}

        for k, v in self._cp.items(section):
            if k.startswith(prefix):
                try:
                    iv[k] = type(v)
                except ValueError:
                    pass

        return iv

    def remove_option(self, section, option):
        self._cp.remove_option(section, option)

    def sect_eq(self, other, section):
        try:
            sitems = dict(self._cp.items(section))
            oitems = dict(other._cp.items(section))
            return sitems == oitems
        except NoSectionError:
            return False

    def sect_diff(self, other, section):
        sitems = dict(self._cp.items(section))

        # Section missing from other, everything is new
        try:
            oitems = dict(other._cp.items(section))
        except NoSectionError:
            return {k: (v, None) for k, v in sitems.items()}

        # Return {key: (self_val, other_val)} for options that differ
        return {k: (sitems.get(k), oitems.get(k))
                for k, _ in sitems.items() ^ oitems.items()}

    def sections(self):
        return self._cp.sections()

    def rename_section(self, sfrom, sto):
        items = self._cp.items(sfrom)

        self._cp.remove_section(sfrom)
        self._cp.remove_section(sto)
        self._cp.add_section(sto)

        for k, v in items:
            self._cp.set(sto, k, v)

    def tostr(self):
        buf = io.StringIO()
        self._cp.write(buf)
        return buf.getvalue()
