*********
[backend]
*********

Parameterises the backend with

1. ``precision`` --- number precision, note not all backends support 
   double precision:

    ``single`` | ``double``

2.  ``memory-model`` --- if to enable support for large working sets;
    should be ``normal`` unless a memory-model error is encountered:

    ``normal`` | ``large``

3. ``rank-allocator`` --- MPI rank allocator:

    ``linear`` | ``random``

4. ``collect-wait-times`` --- if to track MPI request wait times or not:

    ``True`` | ``False``

5. ``collect-wait-times-len`` --- size of the wait time history buffer:

     *int*

Example::

    [backend]
    precision = double
    rank-allocator = linear
