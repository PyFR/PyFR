# -*- coding: utf-8 -*-

from ctypes import (POINTER, Structure, cast, c_char, c_char_p, c_int,
                    c_size_t, c_uint, c_void_p, pointer)

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
        (c_int, 'hipGetDeviceProperties', POINTER(HIPDevProps), c_int),
        (c_int, 'hipSetDevice', c_int),
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
        (c_int, 'hipStreamSynchronize', c_void_p),
        (c_int, 'hipModuleLoadData', POINTER(c_void_p), c_char_p),
        (c_int, 'hipModuleUnload', c_void_p),
        (c_int, 'hipModuleGetFunction', POINTER(c_void_p), c_void_p, c_char_p),
        (c_int, 'hipModuleLaunchKernel', c_void_p, c_uint, c_uint, c_uint,
         c_uint, c_uint, c_uint, c_uint, c_void_p, POINTER(c_void_p), c_void_p)
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

    def synchronize(self):
        self.hip.lib.hipStreamSynchronize(self)


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

        # Save a reference to our underlying module
        self.module = module

        # For each argument type instantiate the corresponding ctypes type
        self._args = args = [atype() for atype in argtypes]

        # Obtain pointers to these arguments
        self._arg_ptrs = [cast(pointer(arg), c_void_p) for arg in args]
        self._arg_ptrs = (c_void_p * len(args))(*self._arg_ptrs)

    def exec_async(self, grid, block, stream, *args):
        for src, dst in zip(args, self._args):
            dst.value = getattr(src, '_as_parameter_', src)

        self.hip.lib.hipModuleLaunchKernel(self, *grid, *block, 0, stream,
                                           self._arg_ptrs, 0)


class HIP:
    def __init__(self):
        self.lib = HIPWrappers()

    def set_device(self, devid):
        self.lib.hipSetDevice(devid)

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
