# -*- coding: utf-8 -*-

import pycuda.driver as cuda


def memcpy2d_htod(dev, host, spitch, dpitch, width, height):
    copy = cuda.Memcpy2D()
    copy.set_src_host(host)
    copy.set_dst_device(dev)
    copy.src_pitch = spitch
    copy.dst_pitch = dpitch
    copy.width_in_bytes = width
    copy.height = height
    copy(aligned=True)


def memcpy2d_dtoh(host, dev, spitch, dpitch, width, height):
    copy = cuda.Memcpy2D()
    copy.set_src_device(dev)
    copy.set_dst_host(host)
    copy.src_pitch = spitch
    copy.dst_pitch = dpitch
    copy.width_in_bytes = width
    copy.height = height
    copy(aligned=True)
