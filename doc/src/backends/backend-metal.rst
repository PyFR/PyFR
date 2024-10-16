****************
[backend-metal]
****************

Parameterises the Metal backend with

#. ``gimmik-max-nnz`` --- cutoff for GiMMiK in terms of the number of
   non-zero entires in a constant matrix, defaults to 2048:

    *int*

#. ``gimmik-nkerns`` --- number of kernel algorithms to try when
   benchmarking, defaults to 18:

    *int*

#. ``gimmik-nbench`` --- number of benchmarking runs for each
   kernel, defaults to 40:

     *int*

Example::

    [backend-metal]
    gimmik-max-nnz = 512
