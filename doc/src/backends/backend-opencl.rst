****************
[backend-opencl]
****************

Parameterises the OpenCL backend with

1. ``platform-id`` --- for selecting platform id:

    *int* | *string*

2. ``device-type`` --- for selecting what type of device(s) to run on:

    ``all`` | ``cpu`` | ``gpu`` | ``accelerator``

3. ``device-id`` --- for selecting which device(s) to run on:

    *int* | *string* | ``local-rank`` | ``uuid``

4. ``gimmik-max-nnz`` --- cutoff for GiMMiK in terms of the number of
   non-zero entires in a constant matrix, defaults to 2048:

    *int*

5. ``gimmik-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 8:

    *int*

6. ``gimmik-nbench`` --- number of benchmarking runs for each
   kernel, defaults to 5:

     *int*

Example::

    [backend-opencl]
    platform-id = 0
    device-type = gpu
    device-id = local-rank
    gimmik-max-nnz = 512
