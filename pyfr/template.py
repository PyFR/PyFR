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

        # Subclass Template to support implicit arguments
        class DefaultTemplate(Template):
            def render(iself, *args, **kwargs):
                return super().render(*args, **self.dfltargs, **kwargs)

        return DefaultTemplate(src, lookup=self)

    def _get_raw_macro(self, source, name):
        # Search for macro and extract namespace header and body text
        # Returns (namespace_header, macro_body) or (None, None) if not found
        pattern = rf'<%pyfr:macro\s+name=[\'"]({name})[\'"].*?>(.*?)</%pyfr:macro>'

        def search_source(src):
            # Check if macro is in this source
            match = re.search(pattern, src, re.DOTALL)
            if match:
                # Extract namespace directives from this source
                namespaces = re.findall(r'<%namespace[^>]+/>', src)
                header = '\n'.join(namespaces)
                return header, match.group(2)

            # Search includes recursively
            includes = re.findall(r'<%include\s+file=[\'"]([^\'"]+)[\'"]', src)
            for inc in includes:
                inc_tpl = self.get_template(inc)
                result = search_source(inc_tpl.source)
                if result[0] is not None:
                    return result

            return None, None

        return search_source(source)
