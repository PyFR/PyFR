from ctypes import (POINTER, create_string_buffer, c_char_p, c_int, c_size_t,
                    c_void_p)

from pyfr.cache import ObjectCache
from pyfr.ctypesutil import LibWrapper
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import digest


# Possible HIPRTC exception types
class HIPRTCError(Exception): pass
class HIPRTCOutOfMemory(HIPRTCError): pass
class HIPRTCProgCreationFailure(HIPRTCError): pass
class HIPRTCInvalidInput(HIPRTCError): pass
class HIPRTCInvalidProgram(HIPRTCError): pass
class HIPRTCInvalidOption(HIPRTCError): pass
class HIPRTCCompilationError(HIPRTCError): pass
class HIPRTCInternalError(HIPRTCError): pass


class HIPRTCWrappers(LibWrapper):
    _libname = 'hiprtc'

    # Error codes
    _statuses = {
        1: HIPRTCOutOfMemory,
        2: HIPRTCProgCreationFailure,
        3: HIPRTCInvalidInput,
        4: HIPRTCInvalidProgram,
        5: HIPRTCInvalidOption,
        6: HIPRTCCompilationError,
        11: HIPRTCInternalError,
        '*': HIPRTCError
    }

    # Functions
    _functions = [
        (c_int, 'hiprtcCreateProgram', POINTER(c_void_p), c_char_p, c_char_p,
         c_int, POINTER(c_char_p), POINTER(c_char_p)),
        (c_int, 'hiprtcDestroyProgram', POINTER(c_void_p)),
        (c_int, 'hiprtcCompileProgram', c_void_p, c_int, POINTER(c_char_p)),
        (c_int, 'hiprtcGetCodeSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'hiprtcGetCode', c_void_p, c_char_p),
        (c_int, 'hiprtcGetProgramLogSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'hiprtcGetProgramLog', c_void_p, c_char_p)
    ]


class HIPRTC:
    def __init__(self):
        self.lib = HIPRTCWrappers()

    def compile(self, name, src, flags=[]):
        # Create the program
        prog = c_void_p()
        self.lib.hiprtcCreateProgram(prog, src.encode(),
                                     f'{name}.hip'.encode(), 0, None, None)

        # Try to compile it
        try:
            if flags:
                bflags = [f.encode() for f in flags]
                aflags = (c_char_p * len(flags))(*bflags)
            else:
                aflags = None

            try:
                # Perform the compilation
                self.lib.hiprtcCompileProgram(prog, len(flags), aflags)
            except HIPRTCError:
                # Fetch the log size
                logsz = c_size_t()
                self.lib.hiprtcGetProgramLogSize(prog, logsz)

                # Fetch the log itself
                log = create_string_buffer(logsz.value)
                self.lib.hiprtcGetProgramLog(prog, log)

                raise RuntimeError(log.value.decode())

            # Fetch the program size
            codesz = c_size_t()
            self.lib.hiprtcGetCodeSize(prog, codesz)

            # Fetch the program itself
            code = create_string_buffer(codesz.value)
            self.lib.hiprtcGetCode(prog, code)
        finally:
            # Destroy the program
            self.lib.hiprtcDestroyProgram(prog)

        return code.raw


class HIPCompiler:
    def __init__(self, hip):
        self.hiprtc = HIPRTC()
        self.cache = ObjectCache('hip')

        # Query the HIP runtime version
        version = c_int()
        hip.lib.hipRuntimeGetVersion(version)
        self.version = version.value

    def build(self, name, src, flags=[]):
        ckey = digest(self.version, name, src, flags)

        code = self.cache.get_bytes(ckey)
        if code is None:
            code = self.hiprtc.compile(name, src, flags)

            self.cache.set_with_bytes(ckey, code)

        return code


class HIPCompilerModule:
    def __init__(self, backend, src):
        # Prepare the source code
        src = f'extern "C"\n{{\n{src}\n}}'

        # Get the compute architecture
        arch = backend.props['gcn_arch_name']

        # Compiler flags
        flags = [f'--gpu-architecture={arch}', '-munsafe-fp-atomics']

        # Compile
        code = backend.compiler.build('kernel', src, flags)

        # Load it as a module
        self.mod = backend.hip.load_module(code)

    def get_function(self, name, argtypes):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]
        return self.mod.get_function(name, argtypes)
