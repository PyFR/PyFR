**************
[backend-cuda]
**************

Parameterises the CUDA backend with

1. ``device-id`` --- method for selecting which device(s) to run on:

     *int* | ``round-robin`` | ``local-rank`` | ``uuid``

2. ``mpi-type`` --- type of MPI library that is being used:

     ``standard`` | ``cuda-aware``

3. ``cflags`` --- additional NVIDIA realtime compiler (``nvrtc``) flags:

    *string*

4. ``cublas-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 512:

    *int*

5. ``gimmik-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 8:

    *int*

6. ``gimmik-nbench`` --- number of benchmarking runs for each
   kernel, defaults to 5:

    *int*

Example::

    [backend-cuda]
    device-id = round-robin
    mpi-type = standard
