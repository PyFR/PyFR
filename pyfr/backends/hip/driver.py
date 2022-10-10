# -*- coding: utf-8 -*-

from ctypes import (POINTER, Structure, addressof, create_string_buffer,
                    c_char, c_char_p, c_float, c_int, c_size_t, c_uint,
                    c_void_p)
from uuid import UUID

import numpy as np

from pyfr.ctypesutil import LibWrapper


# Possible HIP exception types
class HIPError(Exception): pass
class HIPInvalidValue(HIPError): pass
class HIPNotInitialized(HIPError): pass
class HIPOutOfMemory(HIPError): pass
class HIPInsufficientDriver(HIPError): pass
class HIPPriorLaunchFailure(HIPError): pass
class HIPInvalidDevice(HIPError): pass
class HIPECCNotCorrectable(HIPError): pass
class HIPFileNotFound(HIPError): pass
class HIPNotFound(HIPError): pass
class HIPIllegalAddress(HIPError): pass
class HIPLaunchFailure(HIPError): pass


class HIPDevProps(Structure):
    _fields_ = [
        ('s_name', c_char*256),
        ('i_total_global_mem', c_size_t),
        ('i_shared_mem_per_block', c_size_t),
        ('i_regs_per_block', c_int),
        ('i_warp_size', c_int),
        ('i_max_threads_per_block', c_int),
        ('ia_max_threads_dim', c_int*3),
        ('ia_max_grid_size', c_int*3),
        ('i_clock_rate', c_int),
        ('i_memory_clock_rate', c_int),
        ('i_memory_bus_width', c_int),
        ('i_total_const_mem', c_size_t),
        ('i_major', c_int),
        ('i_minor', c_int),
        ('i_multi_processor_count', c_int),
        ('i_l2_cache_size', c_int),
        ('i_max_threads_per_multiprocessor', c_int),
        ('i_compute_mode', c_int),
        ('i_clock_instruction_rate', c_int),
        ('b_global_int32_atomics', c_uint, 1),
        ('b_global_float_atomic_exch', c_uint, 1),
        ('b_shared_int32_atomics', c_uint, 1),
        ('b_shared_float_atomic_exch', c_uint, 1),
        ('b_float_atomic_add', c_uint, 1),
        ('b_global_int64_atomics', c_uint, 1),
        ('b_shared_int64_atomics', c_uint, 1),
        ('b_doubles', c_uint, 1),
        ('b_warp_vote', c_uint, 1),
        ('b_wrap_ballot', c_uint, 1),
        ('b_warp_shuffle', c_uint, 1),
        ('b_funnel_shift', c_uint, 1),
        ('b_thread_fence_system', c_uint, 1),
        ('b_sync_threads_ext', c_uint, 1),
        ('b_surface_funcs', c_uint, 1),
        ('b_3d_grid', c_uint, 1),
        ('b_dynamic_parallelism', c_uint, 1),
        ('b_concurrent_kernels', c_int),
        ('i_pci_domain_id', c_int),
        ('i_pci_bus_id', c_int),
        ('i_pci_device_id', c_int),
        ('i_max_shared_memory_per_multiprocessor', c_size_t),
        ('b_is_multi_gpu_board', c_int),
        ('b_can_map_host_memory', c_int),
        ('i_gcn_arch', c_int),
        ('s_gcn_arch_name', c_char*256),
        ('b_integrated', c_int),
        ('b_cooperative_launch', c_int),
        ('b_cooperative_multi_device_launch', c_int),
        ('i_max_texture_1d', c_int),
        ('ia_max_texture_2d', c_int*2),
        ('ia_max_texture_3d', c_int*3),
        ('hdp_mem_flush_cntl', c_void_p),
        ('hdp_reg_flush_cntl', c_void_p),
        ('i_mem_pitch', c_size_t),
        ('i_texture_alignment', c_size_t),
        ('i_texture_pitch_alignment', c_size_t),
        ('b_kernel_exec_timeout_enalbed', c_int),
        ('b_ecc_enalbed', c_int),
        ('b_tcc_driver', c_int),
        ('b_cooperative_multi_device_unmatched_func', c_int),
        ('b_cooperative_multi_device_unmatched_grid_dim', c_int),
        ('b_cooperative_multi_device_unmatched_block_dim', c_int),
        ('b_cooperative_multi_device_unmatched_shared_mem', c_int),
        ('b_is_large_bar', c_int),
        ('i_asic_revision', c_int),
        ('b_managed_memory', c_int),
        ('b_direct_managed_mem_access_from_host', c_int),
        ('b_concurrent_managed_access', c_int),
        ('b_pageable_memory_access', c_int),
        ('b_pageable_memory_access_using_page_tables', c_int),
        ('reserved', c_int*64)
    ]


class HIPKernelNodeParams(Structure):
    _fields_ = [
        ('block', c_uint * 3),
        ('extra', POINTER(c_void_p)),
        ('func', c_void_p),
        ('grid', c_uint * 3),
        ('kernel_params', POINTER(c_void_p)),
        ('shared_mem_bytes', c_uint)
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


class HIPWrappers(LibWrapper):
    _libname = 'amdhip64'

    # Error codes
    _statuses = {
        1: HIPInvalidValue,
        2: HIPOutOfMemory,
        3: HIPNotInitialized,
        35: HIPInsufficientDriver,
        53: HIPPriorLaunchFailure,
        101: HIPInvalidDevice,
        214: HIPECCNotCorrectable,
        301: HIPFileNotFound,
        500: HIPNotFound,
        700: HIPIllegalAddress,
        719: HIPLaunchFailure,
        '*': HIPError
    }

    # Constants
    MEMCPY_DEFAULT = 4

    # Functions
    _functions = [
        (c_int, 'hipGetDeviceCount', POINTER(c_int)),
        (c_int, 'hipGetDeviceProperties', POINTER(HIPDevProps), c_int),
        (c_int, 'hipSetDevice', c_int),
        (c_int, 'hipDeviceGetUuid', 16*c_char, c_int),
        (c_int, 'hipMalloc', POINTER(c_void_p), c_size_t),
        (c_int, 'hipFree', c_void_p),
        (c_int, 'hipHostMalloc', POINTER(c_void_p), c_size_t, c_uint),
        (c_int, 'hipHostFree', c_void_p),
        (c_int, 'hipMemcpy', c_void_p, c_void_p, c_size_t, c_int),
        (c_int, 'hipMemcpyAsync', c_void_p, c_void_p, c_size_t, c_int,
         c_void_p),
        (c_int, 'hipMemset', c_void_p, c_int, c_size_t),
        (c_int, 'hipStreamCreate', POINTER(c_void_p)),
        (c_int, 'hipStreamDestroy', c_void_p),
        (c_int, 'hipStreamBeginCapture', c_void_p, c_uint),
        (c_int, 'hipStreamEndCapture', c_void_p, POINTER(c_void_p)),
        (c_int, 'hipStreamSynchronize', c_void_p),
        (c_int, 'hipEventCreate', POINTER(c_void_p)),
        (c_int, 'hipEventRecord', c_void_p, c_void_p),
        (c_int, 'hipEventDestroy', c_void_p),
        (c_int, 'hipEventSynchronize', c_void_p),
        (c_int, 'hipEventElapsedTime', POINTER(c_float), c_void_p, c_void_p),
        (c_int, 'hipModuleLoadData', POINTER(c_void_p), c_char_p),
        (c_int, 'hipModuleUnload', c_void_p),
        (c_int, 'hipModuleGetFunction', POINTER(c_void_p), c_void_p, c_char_p),
        (c_int, 'hipModuleLaunchKernel', c_void_p, c_uint, c_uint, c_uint,
         c_uint, c_uint, c_uint, c_uint, c_void_p, POINTER(c_void_p),
         c_void_p),
        (c_int, 'hipGraphCreate', POINTER(c_void_p), c_uint),
        (c_int, 'hipGraphDestroy', c_void_p),
        (c_int, 'hipGraphAddEmptyNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t),
        (c_int, 'hipGraphAddEventRecordNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p),
        (c_int, 'hipGraphAddKernelNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, POINTER(HIPKernelNodeParams)),
        (c_int, 'hipGraphAddChildGraphNode', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p),
        (c_int, 'hipGraphAddMemcpyNode1D', POINTER(c_void_p), c_void_p,
         POINTER(c_void_p), c_size_t, c_void_p, c_void_p, c_size_t, c_int),
        (c_int, 'hipGraphInstantiate', POINTER(c_void_p), c_void_p, c_void_p,
         c_char_p, c_size_t),
        (c_int, 'hipGraphExecKernelNodeSetParams', c_void_p, c_void_p,
         POINTER(HIPKernelNodeParams)),
        (c_int, 'hipGraphExecDestroy', c_void_p),
        (c_int, 'hipGraphLaunch', c_void_p, c_void_p)
    ]


class _HIPBase:
    _destroyfn = None

    def __init__(self, hip, ptr):
        self.hip = hip
        self._as_parameter_ = ptr.value

    def __del__(self):
        if self._destroyfn:
            try:
                getattr(self.hip.lib, self._destroyfn)(self)
            except AttributeError:
                pass

    def __int__(self):
        return self._as_parameter_


class HIPDevAlloc(_HIPBase):
    _destroyfn = 'hipFree'

    def __init__(self, hip, nbytes):
        self.nbytes = nbytes

        ptr = c_void_p()
        hip.lib.hipMalloc(ptr, nbytes)

        super().__init__(hip, ptr)


class HIPHostAlloc(_HIPBase):
    _destroyfn = 'hipHostFree'

    def __init__(self, hip, nbytes):
        self.nbytes = nbytes

        ptr = c_void_p()
        hip.lib.hipHostMalloc(ptr, nbytes, 0)

        super().__init__(hip, ptr)


class HIPStream(_HIPBase):
    _destroyfn = 'hipStreamDestroy'

    def __init__(self, hip):
        ptr = c_void_p()
        hip.lib.hipStreamCreate(ptr)

        super().__init__(hip, ptr)

    def begin_capture(self):
        self.hip.lib.hipStreamBeginCapture(self, 0)

    def end_capture(self):
        graph = c_void_p()
        self.hip.lib.hipStreamEndCapture(self, graph)

        return HIPGraph(self.hip, graph)

    def synchronize(self):
        self.hip.lib.hipStreamSynchronize(self)


class HIPEvent(_HIPBase):
    _destroyfn = 'hipEventDestroy'

    def __init__(self, hip):
        ptr = c_void_p()
        hip.lib.hipEventCreate(ptr)

        super().__init__(hip, ptr)

    def record(self, stream):
        self.hip.lib.hipEventRecord(self, stream)

    def synchronize(self):
        self.hip.lib.hipEventSynchronize(self)

    def elapsed_time(self, start):
        dt = c_float()
        self.hip.lib.hipEventElapsedTime(dt, start, self)

        return dt.value / 1e3


class HIPModule(_HIPBase):
    _destroyfn = 'hipModuleUnload'

    def __init__(self, hip, code):
        ptr = c_void_p()
        hip.lib.hipModuleLoadData(ptr, code)

        super().__init__(hip, ptr)

    def get_function(self, name, argspec):
        return HIPFunction(self.hip, self, name, argspec)


class HIPFunction(_HIPBase):
    def __init__(self, hip, module, name, argtypes):
        ptr = c_void_p()
        hip.lib.hipModuleGetFunction(ptr, module, name.encode())

        super().__init__(hip, ptr)

        # Save a reference to our underlying module and argument types
        self.module = module
        self.argtypes = list(argtypes)

    def make_params(self, grid, block, sharedb=0):
        return HIPKernelNodeParams(self, grid, block, sharedb)

    def exec_async(self, stream, params):
        self.hip.lib.hipModuleLaunchKernel(self, *params.grid, *params.block,
                                           params.shared_mem_bytes, stream,
                                           params.kernel_params, None)


class HIPGraph(_HIPBase):
    _destroyfn = 'hipGraphDestroy'

    def __init__(self, hip, ptr=None):
        if ptr is None:
            ptr = c_void_p()
            hip.lib.hipGraphCreate(ptr, 0)

        super().__init__(hip, ptr)

    @staticmethod
    def _make_deps(deps):
        if deps:
            return (c_void_p * len(deps))(*deps), len(deps)
        else:
            return None, 0

    def add_empty(self, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddEmptyNode(ptr, self, *self._make_deps(deps))

        return ptr.value

    def add_event_record(self, event, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddEventRecordNode(ptr, self,
                                                *self._make_deps(deps), event)

        return ptr.value

    def add_kernel(self, kparams, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddKernelNode(ptr, self, *self._make_deps(deps),
                                           kparams)

        return ptr.value

    def add_memcpy(self, dst, src, nbytes, deps=None):
        kind = self.hip.lib.MEMCPY_DEFAULT

        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        ptr = c_void_p()
        self.hip.lib.hipGraphAddMemcpyNode1D(ptr, self, *self._make_deps(deps),
                                             dst, src, nbytes, kind)

        return ptr.value

    def add_graph(self, graph, deps=None):
        ptr = c_void_p()
        self.hip.lib.hipGraphAddChildGraphNode(ptr, self,
                                               *self._make_deps(deps), graph)

        return ptr.value

    def instantiate(self):
        return HIPExecGraph(self.hip, self)


class HIPExecGraph(_HIPBase):
    _destroyfn = 'hipGraphExecDestroy'

    def __init__(self, hip, graph):
        ptr = c_void_p()
        hip.lib.hipGraphInstantiate(ptr, graph, None, None, 0)

        super().__init__(hip, ptr)

        # Save a reference to the graph
        self.graph = graph

    def set_kernel_node_params(self, node, kparams):
        self.hip.lib.hipGraphExecKernelNodeSetParams(self, node, kparams)

    def launch(self, stream):
        self.hip.lib.hipGraphLaunch(self, stream)


class HIP:
    def __init__(self):
        self.lib = HIPWrappers()

    def device_count(self):
        count = c_int()
        self.lib.hipGetDeviceCount(count)

        return count.value

    def device_properties(self, devid):
        props = HIPDevProps()
        self.lib.hipGetDeviceProperties(props, devid)

        dprops = {}
        for name, *t in props._fields_:
            if name.startswith('b_'):
                dprops[name[2:]] = bool(getattr(props, name))
            elif name.startswith('i_'):
                dprops[name[2:]] = int(getattr(props, name))
            elif name.startswith('ia_'):
                dprops[name[3:]] = [int(v) for v in getattr(props, name)]
            elif name.startswith('s_'):
                dprops[name[2:]] = getattr(props, name).decode()

        return dprops

    def device_uuid(self, devid):
        buf = create_string_buffer(16)
        self.lib.hipDeviceGetUuid(buf, devid)

        return UUID(bytes=buf.raw)

    def set_device(self, devid):
        self.lib.hipSetDevice(devid)

    def mem_alloc(self, nbytes):
        return HIPDevAlloc(self, nbytes)

    def pagelocked_empty(self, shape, dtype):
        nbytes = np.prod(shape)*np.dtype(dtype).itemsize

        alloc = HIPHostAlloc(self, nbytes)
        alloc.__array_interface__ = {
            'version': 3,
            'typestr': np.dtype(dtype).str,
            'data': (int(alloc), False),
            'shape': tuple(shape)
        }

        return np.array(alloc, copy=False)

    def memcpy(self, dst, src, nbytes, stream=None):
        kind = self.lib.MEMCPY_DEFAULT

        if isinstance(dst, (np.ndarray, np.generic)):
            dst = dst.ctypes.data

        if isinstance(src, (np.ndarray, np.generic)):
            src = src.ctypes.data

        if stream is None:
            self.lib.hipMemcpy(dst, src, nbytes, kind)
        else:
            self.lib.hipMemcpyAsync(dst, src, nbytes, kind, stream)

    def memset(self, dst, val, nbytes):
        self.lib.hipMemset(dst, val, nbytes)

    def load_module(self, code):
        return HIPModule(self, code)

    def create_stream(self):
        return HIPStream(self)

    def create_event(self):
        return HIPEvent(self)

    def create_graph(self):
        return HIPGraph(self)
