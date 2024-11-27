from ctypes import (POINTER, create_string_buffer, c_char_p, c_int, c_size_t,
                    c_void_p)
import shlex

from pyfr.cache import ObjectCache
from pyfr.ctypesutil import LibWrapper
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import digest


# Possible NVRTC exception types
class NVRTCError(Exception): pass
class NVRTCOutOfMemory(NVRTCError): pass
class NVRTCProgCreationFailure(NVRTCError): pass
class NVRTCInvalidInput(NVRTCError): pass
class NVRTCInvalidProgram(NVRTCError): pass
class NVRTCInvalidOption(NVRTCError): pass
class NVRTCCompilationError(NVRTCError): pass
class NVRTCInternalError(NVRTCError): pass


class NVRTCWrappers(LibWrapper):
    _libname = 'nvrtc'

    # Error codes
    _statuses = {
        1: NVRTCOutOfMemory,
        2: NVRTCProgCreationFailure,
        3: NVRTCInvalidInput,
        4: NVRTCInvalidProgram,
        5: NVRTCInvalidOption,
        6: NVRTCCompilationError,
        11: NVRTCInternalError,
        '*': NVRTCError
    }

    # Functions
    _functions = [
        (c_int, 'nvrtcCreateProgram', POINTER(c_void_p), c_char_p, c_char_p,
         c_int, POINTER(c_char_p), POINTER(c_char_p)),
        (c_int, 'nvrtcDestroyProgram', POINTER(c_void_p)),
        (c_int, 'nvrtcCompileProgram', c_void_p, c_int, POINTER(c_char_p)),
        (c_int, 'nvrtcGetPTXSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'nvrtcGetPTX', c_void_p, c_char_p),
        (c_int, 'nvrtcGetProgramLogSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'nvrtcGetProgramLog', c_void_p, c_char_p),
        (c_int, 'nvrtcGetCUBINSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'nvrtcGetCUBIN', c_void_p, c_char_p)
    ]


class NVRTC:
    def __init__(self):
        self.lib = NVRTCWrappers()

    def compile(self, name, src, flags=[]):
        # Create the program
        prog = c_void_p()
        self.lib.nvrtcCreateProgram(prog, src.encode(), f'{name}.cu'.encode(),
                                    0, None, None)

        # Try to compile it
        try:
            if flags:
                bflags = [f.encode() for f in flags]
                aflags = (c_char_p * len(flags))(*bflags)
            else:
                aflags = None

            try:
                # Perform the compilation
                self.lib.nvrtcCompileProgram(prog, len(flags), aflags)
            except NVRTCError:
                # Fetch the log size
                logsz = c_size_t()
                self.lib.nvrtcGetProgramLogSize(prog, logsz)

                # Fetch the log itself
                log = create_string_buffer(logsz.value)
                self.lib.nvrtcGetProgramLog(prog, log)

                raise RuntimeError(log.value.decode())

            # Query the CUBIN code size
            codesz = c_size_t()
            self.lib.nvrtcGetCUBINSize(prog, codesz)

            # If that worked, fetch the CUBIN itself
            if codesz.value > 0:
                cucode = create_string_buffer(codesz.value)
                self.lib.nvrtcGetCUBIN(prog, cucode)
            # Else, assume the compiled code is PTX
            else:
                self.lib.nvrtcGetPTXSize(prog, codesz)

                cucode = create_string_buffer(codesz.value)
                self.lib.nvrtcGetPTX(prog, cucode)
        finally:
            # Destroy the program
            self.lib.nvrtcDestroyProgram(prog)

        return cucode.raw


class CUDACompiler:
    def __init__(self, cuda):
        self.nvrtc = NVRTC()
        self.cache = ObjectCache('cuda')

        # Query the CUDA driver version
        version = c_int()
        cuda.lib.cuDriverGetVersion(version)
        self.version = version.value

    def build(self, name, src, flags=[]):
        ckey = digest(self.version, name, src, flags)

        code = self.cache.get_bytes(ckey)
        if code is None:
            code = self.nvrtc.compile(name, src, flags)

            self.cache.set_with_bytes(ckey, code)

        return code


class CUDACompilerModule:
    def __init__(self, backend, src):
        # Prepare the source code
        src = f'extern "C"\n{{\n{src}\n}}'

        # Obtain the compute capability for our device
        cmajor, cminor = backend.cuda.compute_capability()

        # Compiler flags
        flags = [
            f'--gpu-architecture=sm_{cmajor}{cminor}',
            '--ftz=true',
            '--fmad=true'
        ]

        flags += shlex.split(backend.cfg.get('backend-cuda', 'cflags', ''))

        # Compile to CUDA code (either PTX or CUBIN depending on arch flag)
        cucode = backend.compiler.build('kernel', src, flags)

        # Load it as a module
        self.mod = backend.cuda.load_module(cucode)

    def get_function(self, name, argtypes):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]

        return self.mod.get_function(name, argtypes)
