import pkgutil
import re

from mako.lookup import TemplateLookup
from mako.template import Template


class DottedTemplateLookup(TemplateLookup):
    def __init__(self, pkg, dfltargs):
        self.dfltpkg = pkg
        self.dfltargs = dfltargs

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
        src = pkgutil.get_data(pkg, f'{basename}.mako')
        if not src:
            raise RuntimeError(f'Template "{name}" not found')

        # Decode bytes to string
        src = src.decode('utf-8')

        # Move py:params into args attribute if macro definition
        src = self._format_marco_def(src)

        # Subclass Template to support implicit arguments
        class DefaultTemplate(Template):
            def render(iself, *args, **kwargs):
                return super().render(*args, **self.dfltargs, **kwargs)

        return DefaultTemplate(src, lookup=self)

    def _format_marco_def(self, source):
        # Transform py:params into args attribute for <%pyfr:macro> tags
        pattern = r'(<%pyfr:macro\s+name=[\'"][^\'"]+[\'"]\s+params=[\'"])([^\'"]+)([\'"](?:\s+externs=[\'"][^\'"]*[\'"])?(?:\s+\w+=[\'"][^\'"]*[\'"])*\s*>)'

        def inject_args(match):
            prefix = match.group(1)
            params = [p.strip() for p in match.group(2).split(',')]
            suffix = match.group(3)

            # Partition params into regular and py: prefixed
            pyparams = [p[3:] for p in params if p.startswith('py:')]
            params = [p for p in params if not p.startswith('py:')]

            if not pyparams:
                return match.group(0)

            pstr = ', '.join(params)
            astr = ', '.join(pyparams)

            return f"{prefix}{pstr}' args='{astr}{suffix}"

        return re.sub(pattern, inject_args, source)
