from pyfr.cache import ObjectCache
from pyfr.util import digest


class OpenCLCompiler:
    def __init__(self, cl):
        self.cl = cl
        self.cache = ObjectCache('ocl')
        self.dev_key = (cl.dev.name, cl.dev.vendor, cl.dev.driver_version)

    def build(self, src, flags=[]):
        ckey = digest(*self.dev_key, src, flags)

        if bin := self.cache.get_bytes(ckey):
            program = self.cl.program(bin)
        else:
            program = self.cl.program(src, flags)

            self.cache.set_with_bytes(ckey, program.get_binary())

        return program
