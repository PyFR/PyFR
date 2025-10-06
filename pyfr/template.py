import hashlib
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

        # Extract macro IDs and inject registration code
        src = self._inject_macro_id_registration(src)

        # Transform py:params into args attribute
        src = self._format_macro_def(src)

        # Subclass Template to support implicit arguments
        class DefaultTemplate(Template):
            def render(iself, *args, **kwargs):
                return super().render(*args, **self.dfltargs, **kwargs)

        return DefaultTemplate(src, lookup=self)

    def _inject_macro_id_registration(self, source):
        # Extract macro definitions and compute their IDs
        ids = self._extract_macro_ids(source)

        # Generate preamble code to register IDs when template is rendered
        preamble = '<%\n'
        for name, macid in ids.items():
            preamble += f'_macro_ids.setdefault("{name}", set()).add("{macid}")\n'
        preamble += '%>\n'

        return preamble + source

    def _extract_macro_ids(self, source):
        # Extract macro definitions and compute their IDs
        pattern = r'(<%pyfr:macro\s+name=[\'"]([^\'"]+)[\'"].*?>.*?</%pyfr:macro>)'
        ids = {}

        for match in re.finditer(pattern, source, re.DOTALL):
            macro = match.group(1)
            name = match.group(2)
            macroid = hashlib.md5(macro.strip().encode()).hexdigest()
            ids[name] = macroid

        return ids

    def _format_macro_def(self, source):
        # Pattern to extract individual attributes
        apattern = r'(\w+)=[\'"]([^\'"]*)[\'"]'

        def inject_args(match):
            macro = match[0]

            # Extract all attributes from the macro definition
            attrs = dict(re.findall(apattern, macro))

            # Check if params attribute exists
            if 'params' not in attrs:
                return macro

            # Parse the params attribute
            params = [p.strip() for p in attrs['params'].split(',')]

            # Partition params into regular and py: prefixed
            pyparams = [p[3:] for p in params if p.startswith('py:')]
            params = [p for p in params if not p.startswith('py:')]

            if not pyparams:
                return macro

            # Update the params and create args
            attrs['params'] = ', '.join(params)
            attrs['args'] = ', '.join(pyparams)

            # Reconstruct the macro definition
            attrstr = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
            return f'<%pyfr:macro {attrstr}>'

        mpattern = r'<%pyfr:macro\s+[^>]+>'
        return re.sub(mpattern, inject_args, source)
