**************
[backend-cuda]
**************

Parameterises the CUDA backend with

#. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``round-robin`` | ``local-rank`` | ``uuid``

#. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``cuda-aware``

#. ``cflags`` --- additional NVIDIA realtime compiler (``nvrtc``) flags:

    *string*

#. ``cublas-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 512:

    *int*

#. ``gimmik-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 8:

    *int*

#. ``gimmik-nbench`` --- number of benchmarking runs for each
   kernel, defaults to 5:

    *int*

Example::

    [backend-cuda]
    device-id = round-robin
    mpi-type = standard
