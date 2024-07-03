****************
[backend-openmp]
****************

Parameterises the OpenMP backend with

1. ``cc`` --- C compiler:

    *string*

2. ``cflags`` --- additional C compiler flags:

    *string*

3. ``alignb`` --- alignment requirement in bytes; must be a power of
   two and at least 32:

    *int*

4. ``schedule`` --- OpenMP loop scheduling scheme:

    ``static`` | ``dynamic`` | ``dynamic, n`` | ``guided`` | ``guided, n``

    where *n* is a positive integer.

Example::

    [backend-openmp]
    cc = gcc
