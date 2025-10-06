import pkgutil

from mako.lookup import TemplateLookup
from mako.template import Template


class DottedTemplateLookup(TemplateLookup):
    def __init__(self, pkg, dfltargs):
        self.dfltpkg = pkg
        self.dfltargs = dfltargs
        self._filters = []

    def add_filter(self, func):
        self._filters.append(func)

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

        # Apply all filters to source
        for filter in self._filters:
            src = filter(src)

        # Subclass Template to support implicit arguments
        class DefaultTemplate(Template):
            def render(iself, *args, **kwargs):
                return super().render(*args, **self.dfltargs, **kwargs)

        return DefaultTemplate(src, lookup=self)
