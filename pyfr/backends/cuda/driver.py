# -*- coding: utf-8 -*-

from ctypes import (POINTER, cast, c_char, c_char_p, c_int, c_size_t, c_uint,
                    c_void_p, pointer)

import numpy as np

from pyfr.ctypesutil import LibWrapper


# Possible CUDA exception types
CUDAError = type('CUDAError', (Exception,), {})
CUDAInvalidValue = type('CUDAInvalidValue', (CUDAError,), {})
CUDAOutofMemory = type('CUDAOutofMemory', (CUDAError,), {})
CUDANotInitalized = type('CUDANotInitalized', (CUDAError,), {})
CUDAInvalidDevice = type('CUDAInvalidDevice', (CUDAError,), {})
CUDAECCUncorrectable = type('CUDAECCUncorrectable', (CUDAError,), {})
CUDAOSError = type('CUDAOSError', (CUDAError, OSError), {})
CUDAInvalidHandle = type('CUDAInvalidHandle', (CUDAError,), {})
CUDAIllegalAddress = type('CUDAIllegalAddress', (CUDAError,), {})
CUDALaunchOutOfResources = type('CUDALaunchOutOfResources', (CUDAError,), {})
CUDALaunchFailed = type('CUDALaunchFailed', (CUDAError,), {})
CUDASystemDriverMismatch = type('CUDASystemDriverMismatch', (CUDAError,), {})


class CUDAWrappers(LibWrapper):
    _libname = 'cuda'

    # Error codes
    _statuses = {
        1: CUDAInvalidValue,
        2: CUDAOutofMemory,
        3: CUDANotInitalized,
        101: CUDAInvalidDevice,
        214: CUDAECCUncorrectable,
        304: CUDAOSError,
        400: CUDAInvalidHandle,
        700: CUDAIllegalAddress,
        701: CUDALaunchOutOfResources,
        719: CUDALaunchFailed,
        803: CUDASystemDriverMismatch,
        '*': CUDAError
    }

    # Constants
    COMPUTE_CAPABILITY_MAJOR = 75
    COMPUTE_CAPABILITY_MINOR = 76
    EVENT_DISABLE_TIMING = 2
    FUNC_CACHE_PREFER_NONE = 0
    FUNC_CACHE_PREFER_SHARED = 1
    FUNC_CACHE_PREFER_L1 = 2
    FUNC_CACHE_PREFER_EQUAL = 3

    # Functions
    _functions = [
        (c_int, 'cuInit', c_int),
        (c_int, 'cuDeviceGet', POINTER(c_int), c_int),
        (c_int, 'cuDeviceGetCount',  POINTER(c_int)),
        (c_int, 'cuDeviceGetAttribute', POINTER(c_int), c_int, c_int),
        (c_int, 'cuDevicePrimaryCtxRetain', POINTER(c_void_p), c_int),
        (c_int, 'cuDevicePrimaryCtxRelease', c_int),
        (c_int, 'cuCtxSetCurrent', c_void_p),
        (c_int, 'cuCtxSetCacheConfig', c_int),
        (c_int, 'cuMemAlloc_v2', POINTER(c_void_p), c_size_t),
        (c_int, 'cuMemFree_v2', c_void_p),
        (c_int, 'cuMemAllocHost_v2', POINTER(c_void_p), c_size_t),
        (c_int, 'cuMemFreeHost', c_void_p),
        (c_int, 'cuMemcpy', c_void_p, c_void_p, c_size_t),
        (c_int, 'cuMemcpyAsync', c_void_p, c_void_p, c_size_t, c_void_p),
        (c_int, 'cuMemsetD8_v2', c_void_p, c_char, c_size_t),
        (c_int, 'cuStreamCreate', POINTER(c_void_p), c_uint),
        (c_int, 'cuStreamDestroy_v2', c_void_p),
        (c_int, 'cuStreamSynchronize', c_void_p),
        (c_int, 'cuStreamWaitEvent', c_void_p, c_void_p, c_uint),
        (c_int, 'cuEventCreate', POINTER(c_void_p), c_uint),
        (c_int, 'cuEventDestroy_v2', c_void_p),
        (c_int, 'cuEventRecord', c_void_p, c_void_p),
        (c_int, 'cuModuleLoadDataEx', POINTER(c_void_p), c_char_p, c_uint,
         POINTER(c_int), POINTER(c_void_p)),
        (c_int, 'cuModuleUnload', c_void_p),
        (c_int, 'cuModuleGetFunction', POINTER(c_void_p), c_void_p, c_char_p),
        (c_int, 'cuLaunchKernel', c_void_p, c_uint, c_uint, c_uint, c_uint,
         c_uint, c_uint, c_uint, c_void_p, POINTER(c_void_p), c_void_p),
        (c_int, 'cuFuncSetCacheConfig', c_void_p, c_int)
    ]

    def _transname(self, name):
        return name[:-3] if name.endswith('_v2') else name


class _CUDABase(object):
    _destroyfn = None

    def __init__(self, cuda, ptr):
        self.cuda = cuda
        self._as_parameter_ = ptr.value

    def __del__(self):
        if self._destroyfn:
            try:
                getattr(self.cuda.lib, self._destroyfn)(self)
            except AttributeError:
                pass

    def __int__(self):
        return self._as_parameter_


class CUDADevAlloc(_CUDABase):
    _destroyfn = 'cuMemFree'

    def __init__(self, cuda, nbytes):
        ptr = c_void_p()
        cuda.lib.cuMemAlloc(ptr, nbytes)

        super().__init__(cuda, ptr)
        self.nbytes = nbytes


class CUDAHostAlloc(_CUDABase):
    _destroyfn = 'cuMemFreeHost'

    def __init__(self, cuda, nbytes):
        self.nbytes = nbytes

        ptr = c_void_p()
        cuda.lib.cuMemAllocHost(ptr, nbytes)

        super().__init__(cuda, ptr)


class CUDAStream(_CUDABase):
    _destroyfn = 'cuStreamDestroy'

    def __init__(self, cuda):
        ptr = c_void_p()
        cuda.lib.cuStreamCreate(ptr, 0)

        super().__init__(cuda, ptr)

    def synchronize(self):
        self.cuda.lib.cuStreamSynchronize(self)

    def wait_for_event(self, event):
        self.cuda.lib.cuStreamWaitEvent(self, event, 0)


class CUDAEvent(_CUDABase):
    _destroyfn = 'cuEventDestroy'

    def __init__(self, cuda):
        ptr = c_void_p()
        cuda.lib.cuEventCreate(ptr, cuda.lib.EVENT_DISABLE_TIMING)

        super().__init__(cuda, ptr)

    def record(self, stream):
        self.cuda.lib.cuEventRecord(self, stream)


class CUDAModule(_CUDABase):
    _destroyfn = 'cuModuleUnload'

    def __init__(self, cuda, ptx):
        ptr = c_void_p()
        cuda.lib.cuModuleLoadDataEx(ptr, ptx, 0, None, None)

        super().__init__(cuda, ptr)

    def get_function(self, name, argspec):
        return CUDAFunction(self.cuda, self, name, argspec)


class CUDAFunction(_CUDABase):
    def __init__(self, cuda, module, name, argtypes):
        ptr = c_void_p()
        cuda.lib.cuModuleGetFunction(ptr, module, name.encode())

        super().__init__(cuda, ptr)

        # Save a reference to our underlying module
        self.module = module

        # For each argument type instantiate the corresponding ctypes type
        self._args = args = [atype() for atype in argtypes]

        # Obtain pointers to these arguments
        self._arg_ptrs = [cast(pointer(arg), c_void_p) for arg in args]
        self._arg_ptrs = (c_void_p * len(args))(*self._arg_ptrs)

    def set_cache_pref(self, *, prefer_l1=None, prefer_shared=None):
        pref = self.cuda._get_cache_pref(prefer_l1, prefer_shared)
        self.cuda.lib.cuFuncSetCacheConfig(self, pref)

    def exec_async(self, grid, block, stream, *args):
        for src, dst in zip(args, self._args):
            dst.value = getattr(src, '_as_parameter_', src)

        self.cuda.lib.cuLaunchKernel(self, *grid, *block, 0, stream,
                                     self._arg_ptrs, None)


class CUDA(object):
    def __init__(self):
        self.lib = CUDAWrappers()
        self.ctx = c_void_p()

        self.lib.cuInit(0)

    def __del__(self):
        if getattr(self, 'ctx', None):
            self.lib.cuDevicePrimaryCtxRelease(self.dev)

    def _get_cache_pref(self, prefer_l1, prefer_shared):
        if prefer_l1 is None and prefer_shared is None:
            return self.lib.FUNC_CACHE_PREFER_NONE
        elif prefer_l1 and not prefer_shared:
            return self.lib.FUNC_CACHE_PREFER_L1
        elif prefer_shared and not prefer_l1:
            return self.lib.FUNC_CACHE_PREFER_SHARED
        else:
            return self.lib.FUNC_CACHE_PREFER_EQUAL

    def device_count(self):
        count = c_int()
        self.lib.cuDeviceGetCount(count)

        return count.value

    def set_device(self, devid):
        if self.ctx:
            raise RuntimeError('Device has already been set')

        dev = c_int()
        self.lib.cuDeviceGet(dev, devid)
        self.lib.cuDevicePrimaryCtxRetain(self.ctx, dev)
        self.lib.cuCtxSetCurrent(self.ctx)
        self.dev = dev.value

    def set_cache_pref(self, *, prefer_l1=None, prefer_shared=None):
        pref = self._get_cache_pref(prefer_l1, prefer_shared)
        self.lib.cuCtxSetCacheConfig(pref)

    def compute_capability(self):
        dev, lib = self.dev, self.lib

        major, minor = c_int(), c_int()
        lib.cuDeviceGetAttribute(major, lib.COMPUTE_CAPABILITY_MAJOR, dev)
        lib.cuDeviceGetAttribute(minor, lib.COMPUTE_CAPABILITY_MINOR, dev)

        return major.value, minor.value

    def mem_alloc(self, nbytes):
        return CUDADevAlloc(self, nbytes)

    def pagelocked_empty(self, shape, dtype):
        nbytes = np.prod(shape)*np.dtype(dtype).itemsize

        alloc = CUDAHostAlloc(self, nbytes)
        alloc.__array_interface__ = {
            'version': 3,
            'typestr': np.dtype(dtype).str,
            'data': (int(alloc), False),
            'shape': tuple(shape)
        }

        return np.array(alloc, copy=False)

    def memcpy(self, dst, src, nbytes):
        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        self.lib.cuMemcpy(dst, src, nbytes)

    def memcpy_async(self, dst, src, nbytes, stream):
        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        self.lib.cuMemcpyAsync(dst, src, nbytes, stream)

    def memset(self, dst, val, nbytes):
        self.lib.cuMemsetD8(dst, val, nbytes)

    def load_module(self, ptx):
        return CUDAModule(self, ptx)

    def create_stream(self):
        return CUDAStream(self)

    def create_event(self):
        return CUDAEvent(self)
