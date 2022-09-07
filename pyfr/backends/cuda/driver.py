# -*- coding: utf-8 -*-

from ctypes import (POINTER, Structure, addressof, byref, create_string_buffer,
                    c_char, c_char_p, c_float, c_int, c_size_t, c_uint,
                    c_ulonglong, c_void_p)
from uuid import UUID

import numpy as np

from pyfr.ctypesutil import LibWrapper


# Possible CUDA exception types
class CUDAError(Exception): pass
class CUDAInvalidValue(CUDAError): pass
class CUDAOutofMemory(CUDAError): pass
class CUDANotInitalized(CUDAError): pass
class CUDANoDevice(CUDAError): pass
class CUDAInvalidDevice(CUDAError): pass
class CUDAECCUncorrectable(CUDAError): pass
class CUDAErrorInvalidPTX(CUDAError): pass
class CUDAErrorUnsupportedPTXVersion(CUDAError): pass
class CUDAOSError(CUDAError, OSError): pass
class CUDAInvalidHandle(CUDAError): pass
class CUDAIllegalAddress(CUDAError): pass
class CUDALaunchOutOfResources(CUDAError): pass
class CUDALaunchFailed(CUDAError): pass
class CUDASystemDriverMismatch(CUDAError): pass


class CUDAKernelNodeParams(Structure):
    _fields_ = [
        ('func', c_void_p),
        ('grid', c_uint * 3),
        ('block', c_uint * 3),
        ('shared_mem_bytes', c_uint),
        ('kernel_params', POINTER(c_void_p)),
        ('extra', POINTER(c_void_p))
    ]

    def __init__(self, func, grid, block, sharedb):
        super().__init__()

        # Save a reference to our underlying function
        self._func = func

        # For each argument type instantiate the corresponding ctypes type
        self._args = args = [atype() for atype in func.argtypes]

        # Obtain pointers to these arguments
        self._arg_ptrs = (c_void_p * len(args))(*map(addressof, args))

        # Fill out the structure
        self.func = int(func)
        self.grid[:] = grid
        self.block[:] = block
        self.shared_mem_bytes = sharedb
        self.kernel_params = self._arg_ptrs

    def set_arg(self, i, v):
        self._args[i].value = getattr(v, '_as_parameter_', v)

    def set_args(self, *kargs, start=0):
        for i, v in enumerate(kargs, start=start):
            self.set_arg(i, v)


class CUDAMemcpy3D(Structure):
    _fields_ = [
        ('src_x_in_bytes', c_size_t),
        ('src_y', c_size_t),
        ('src_z', c_size_t),
        ('src_lod', c_size_t),
        ('src_memory_type', c_int),
        ('src_host', c_void_p),
        ('src_device', c_void_p),
        ('src_array', c_void_p),
        ('_reserved_0', c_void_p),
        ('src_pitch', c_size_t),
        ('src_height', c_size_t),
        ('dst_x_in_bytes', c_size_t),
        ('dst_y', c_size_t),
        ('dst_z', c_size_t),
        ('dst_lod', c_size_t),
        ('dst_memory_type', c_int),
        ('dst_host', c_void_p),
        ('dst_device', c_void_p),
        ('dst_array', c_void_p),
        ('_reserved_1', c_void_p),
        ('dst_pitch', c_size_t),
        ('dst_height', c_size_t),
        ('width_in_bytes', c_size_t),
        ('height', c_size_t),
        ('depth', c_size_t)
    ]


class CUDAWrappers(LibWrapper):
    _libname = 'cuda'

    # Error codes
    _statuses = {
        1: CUDAInvalidValue,
        2: CUDAOutofMemory,
        3: CUDANotInitalized,
        100: CUDANoDevice,
        101: CUDAInvalidDevice,
        214: CUDAECCUncorrectable,
        218: CUDAErrorInvalidPTX,
        222: CUDAErrorUnsupportedPTXVersion,
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
    EVENT_DEFAULT = 0
    EVENT_DISABLE_TIMING = 2
    FUNC_ATTR_SHARED_SIZE_BYTES = 1
    FUNC_ATTR_LOCAL_SIZE_BYTES = 3
    FUNC_ATTR_NUM_REGS = 4
    FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    FUNC_ATTR_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    FUNC_CACHE_PREFER_NONE = 0
    FUNC_CACHE_PREFER_SHARED = 1
    FUNC_CACHE_PREFER_L1 = 2
    FUNC_CACHE_PREFER_EQUAL = 3
    MEMORYTYPE_UNIFIED = 4

    # Functions
    _functions = [
        (c_int, 'cuInit', c_int),
        (c_int, 'cuDeviceGet', POINTER(c_int), c_int),
        (c_int, 'cuDeviceGetCount', POINTER(c_int)),
        (c_int, 'cuDeviceGetAttribute', POINTER(c_int), c_int, c_int),
        (c_int, 'cuDeviceGetUuid_v2', 16*c_char, c_int),
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
        (c_int, 'cuStreamBeginCapture', c_void_p, c_uint),
        (c_int, 'cuStreamEndCapture', c_void_p, POINTER(c_void_p)),
        (c_int, 'cuStreamSynchronize', c_void_p),
        (c_int, 'cuEventCreate', POINTER(c_void_p), c_uint),
        (c_int, 'cuEventDestroy_v2', c_void_p),
        (c_int, 'cuEventRecord', c_void_p, c_void_p),
        (c_int, 'cuEventSynchronize', c_void_p),
        (c_int, 'cuEventElapsedTime', POINTER(c_float), c_void_p, c_void_p),
        (c_int, 'cuModuleLoadDataEx', POINTER(c_void_p), c_char_p, c_uint,
         POINTER(c_int), POINTER(c_void_p)),
        (c_int, 'cuModuleUnload', c_void_p),
        (c_int, 'cuModuleGetFunction', POINTER(c_void_p), c_void_p, c_char_p),
        (c_int, 'cuLaunchKernel', c_void_p, c_uint, c_uint, c_uint, c_uint,
         c_uint, c_uint, c_uint, c_void_p, POINTER(c_void_p), c_void_p),
        (c_int, 'cuFuncGetAttribute', POINTER(c_int), c_int, c_void_p),
        (c_int, 'cuFuncSetAttribute', c_void_p, c_int, c_int),
        (c_int, 'cuFuncSetCacheConfig', c_void_p, c_int),
        (c_int, 'cuGraphCreate', POINTER(c_void_p), c_uint),
        (c_int, 'cuGraphDestroy', c_void_p),
        (c_int, 'cuGraphAddEmptyNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t),
        (c_int, 'cuGraphAddEventRecordNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p),
        (c_int, 'cuGraphAddKernelNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, POINTER(CUDAKernelNodeParams)),
        (c_int, 'cuGraphAddChildGraphNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p),
        (c_int, 'cuGraphAddMemcpyNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, POINTER(CUDAMemcpy3D), c_void_p),
        (c_int, 'cuGraphInstantiateWithFlags', POINTER(c_void_p), c_void_p,
         c_ulonglong),
        (c_int, 'cuGraphExecKernelNodeSetParams', c_void_p, c_void_p,
         POINTER(CUDAKernelNodeParams)),
        (c_int, 'cuGraphExecDestroy', c_void_p),
        (c_int, 'cuGraphLaunch', c_void_p, c_void_p)
    ]

    def _transname(self, name):
        return name.removesuffix('_v2')


class _CUDABase:
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

    def begin_capture(self):
        self.cuda.lib.cuStreamBeginCapture(self, 0)

    def end_capture(self):
        graph = c_void_p()
        self.cuda.lib.cuStreamEndCapture(self, graph)

        return CUDAGraph(self.cuda, graph)

    def synchronize(self):
        self.cuda.lib.cuStreamSynchronize(self)


class CUDAEvent(_CUDABase):
    _destroyfn = 'cuEventDestroy'

    def __init__(self, cuda, timing=False):
        if timing:
            flags = cuda.lib.EVENT_DEFAULT
        else:
            flags = cuda.lib.EVENT_DISABLE_TIMING

        ptr = c_void_p()
        cuda.lib.cuEventCreate(ptr, flags)

        super().__init__(cuda, ptr)

    def record(self, stream):
        self.cuda.lib.cuEventRecord(self, stream)

    def synchronize(self):
        self.cuda.lib.cuEventSynchronize(self)

    def elapsed_time(self, start):
        dt = c_float()
        self.cuda.lib.cuEventElapsedTime(dt, start, self)

        return dt.value / 1e3


class CUDAModule(_CUDABase):
    _destroyfn = 'cuModuleUnload'

    def __init__(self, cuda, cucode):
        ptr = c_void_p()
        cuda.lib.cuModuleLoadDataEx(ptr, cucode, 0, None, None)

        super().__init__(cuda, ptr)

    def get_function(self, name, argspec):
        return CUDAFunction(self.cuda, self, name, argspec)


class CUDAFunction(_CUDABase):
    def __init__(self, cuda, module, name, argtypes):
        ptr = c_void_p()
        cuda.lib.cuModuleGetFunction(ptr, module, name.encode())

        super().__init__(cuda, ptr)

        # Query the register and local memory required by the function
        self.nreg = self._get_attr('num_regs')
        self.shared_mem = self._get_attr('shared_size_bytes')
        self.local_mem = self._get_attr('local_size_bytes')

        # Save a reference to our underlying module and argument types
        self.module = module
        self.argtypes = list(argtypes)

    def _get_attr(self, attr):
        attr = getattr(self.cuda.lib, f'FUNC_ATTR_{attr.upper()}')

        v = c_int()
        self.cuda.lib.cuFuncGetAttribute(byref(v), attr, self)

        return v.value

    def _set_attr(self, attr, val):
        attr = getattr(self.cuda.lib, f'FUNC_ATTR_{attr.upper()}')
        self.cuda.lib.cuFuncSetAttribute(self, attr, val)

    def set_cache_pref(self, *, prefer_l1=None, prefer_shared=None):
        pref = self.cuda._get_cache_pref(prefer_l1, prefer_shared)
        self.cuda.lib.cuFuncSetCacheConfig(self, pref)

    def set_shared_size(self, *, dynm_shared=0, carveout=None):
        self._set_attr('max_dynamic_shared_size_bytes', dynm_shared)

        if carveout is not None:
            self._set_attr('preferred_shared_memory_carveout', carveout)

    def make_params(self, grid, block, sharedb=0):
        return CUDAKernelNodeParams(self, grid, block, sharedb)

    def exec_async(self, stream, params):
        self.cuda.lib.cuLaunchKernel(self, *params.grid, *params.block,
                                     params.shared_mem_bytes, stream,
                                     params.kernel_params, None)


class CUDAGraph(_CUDABase):
    _destroyfn = 'cuGraphDestroy'

    def __init__(self, cuda, ptr=None):
        if ptr is None:
            ptr = c_void_p()
            cuda.lib.cuGraphCreate(ptr, 0)

        super().__init__(cuda, ptr)

    @staticmethod
    def _make_deps(deps):
        if deps:
            return (c_void_p * len(deps))(*deps), len(deps)
        else:
            return None, 0

    def add_empty(self, deps=None):
        ptr = c_void_p()
        self.cuda.lib.cuGraphAddEmptyNode(ptr, self, *self._make_deps(deps))

        return ptr.value

    def add_event_record(self, event, deps=None):
        ptr = c_void_p()
        self.cuda.lib.cuGraphAddEventRecordNode(ptr, self,
                                                *self._make_deps(deps), event)

        return ptr.value

    def add_kernel(self, kparams, deps=None):
        ptr = c_void_p()
        self.cuda.lib.cuGraphAddKernelNode(ptr, self, *self._make_deps(deps),
                                           kparams)

        return ptr.value

    def add_memcpy(self, dst, src, nbytes, deps=None):
        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data
        else:
            dst = getattr(dst, '_as_parameter_', dst)

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data
        else:
            src = getattr(src, '_as_parameter_', src)

        params = CUDAMemcpy3D()
        params.src_memory_type = self.cuda.lib.MEMORYTYPE_UNIFIED
        params.src_device = int(src)
        params.dst_memory_type = self.cuda.lib.MEMORYTYPE_UNIFIED
        params.dst_device = int(dst)
        params.width_in_bytes = nbytes
        params.height = 1
        params.depth = 1

        ptr = c_void_p()
        self.cuda.lib.cuGraphAddMemcpyNode(ptr, self, *self._make_deps(deps),
                                           params, self.cuda.ctx)

        return ptr.value

    def add_graph(self, graph, deps=None):
        ptr = c_void_p()
        self.cuda.lib.cuGraphAddChildGraphNode(ptr, self,
                                               *self._make_deps(deps), graph)

        return ptr.value

    def instantiate(self):
        return CUDAExecGraph(self.cuda, self)


class CUDAExecGraph(_CUDABase):
    _destroyfn = 'cuGraphExecDestroy'

    def __init__(self, cuda, graph):
        ptr = c_void_p()
        cuda.lib.cuGraphInstantiateWithFlags(ptr, graph, 0)

        super().__init__(cuda, ptr)

        # Save a reference to the graph
        self.graph = graph

    def set_kernel_node_params(self, node, kparams):
        self.cuda.lib.cuGraphExecKernelNodeSetParams(self, node, kparams)

    def launch(self, stream):
        self.cuda.lib.cuGraphLaunch(self, stream)


class CUDA:
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

    def device_uuid(self, devid):
        dev = c_int()
        self.lib.cuDeviceGet(dev, devid)

        buf = create_string_buffer(16)
        self.lib.cuDeviceGetUuid(buf, dev)

        return UUID(bytes=buf.raw)

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

    def memcpy(self, dst, src, nbytes, stream=None):
        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        if stream is None:
            self.lib.cuMemcpy(dst, src, nbytes)
        else:
            self.lib.cuMemcpyAsync(dst, src, nbytes, stream)

    def memset(self, dst, val, nbytes):
        self.lib.cuMemsetD8(dst, val, nbytes)

    def load_module(self, cucode):
        return CUDAModule(self, cucode)

    def create_stream(self):
        return CUDAStream(self)

    def create_event(self, timing=False):
        return CUDAEvent(self, timing)

    def create_graph(self):
        return CUDAGraph(self)
