# -*- coding: utf-8 -*-

from ctypes import (POINTER, create_string_buffer, c_char_p, c_int, c_size_t,
                    c_void_p)
import shlex

from pyfr.ctypesutil import LibWrapper
from pyfr.nputil import npdtype_to_ctypestype


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
        (c_int, 'nvrtcCreateProgram',  POINTER(c_void_p), c_char_p, c_char_p,
         c_int, POINTER(c_char_p), POINTER(c_char_p)),
        (c_int, 'nvrtcDestroyProgram', POINTER(c_void_p)),
        (c_int, 'nvrtcCompileProgram', c_void_p, c_int, POINTER(c_char_p)),
        (c_int, 'nvrtcGetPTXSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'nvrtcGetPTX', c_void_p, c_char_p),
        (c_int, 'nvrtcGetProgramLogSize', c_void_p, POINTER(c_size_t)),
        (c_int, 'nvrtcGetProgramLog', c_void_p, c_char_p)
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

            # Fetch the program size
            ptxsz = c_size_t()
            self.lib.nvrtcGetPTXSize(prog, ptxsz)

            # Fetch the program itself
            ptx = create_string_buffer(ptxsz.value)
            self.lib.nvrtcGetPTX(prog, ptx)
        finally:
            # Destroy the program
            self.lib.nvrtcDestroyProgram(prog)

        return ptx.raw


class SourceModule:
    def __init__(self, backend, src):
        # Prepare the source code
        src = f'extern "C"\n{{\n{src}\n}}'

        # Obtain the compute capability for our device
        cmajor, cminor = backend.cuda.compute_capability()

        # Compiler flags
        flags = [
            f'--gpu-architecture=compute_{cmajor}{cminor}',
            '--ftz=true',
            '--fmad=true'
        ]

        flags += shlex.split(backend.cfg.get('backend-cuda', 'cflags', ''))

        # Compile to PTX
        ptx = backend.nvrtc.compile('kernel', src, flags)

        # Load it as a module
        self.mod = backend.cuda.load_module(ptx)

    def get_function(self, name, argtypes):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]

        return self.mod.get_function(name, argtypes)
