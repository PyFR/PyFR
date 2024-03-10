******************
[soln-plugin-tavg]
******************

Time average quantities. Parameterised with

1. ``nsteps`` --- accumulate the average every ``nsteps`` time steps:

    *int*

2. ``dt-out`` --- write to disk every ``dt-out`` time units:

    *float*

3. ``tstart`` --- time at which to start accumulating average data:

    *float*

4. ``mode`` --- output file accumulation mode:

    ``continuous`` | ``windowed``

    In continuous mode each output file contains average data from
    ``tstart`` up until the time at which the file is written. In windowed 
    mode each output file only contains average data for the preceding 
    ``dt-out`` time units. The default is ``windowed``. Average data files
    obtained using the windowed mode can be accumulated after-the-fact using
    the CLI.

5. ``std-mode`` --- standard deviation reporting mode:

    ``summary`` | ``all``

    If to output full standard deviation fields or just summary
    statistics.  In lieu of a complete field, summary instead reports
    the maximum and average standard deviation for each field. The
    default is ``summary`` with ``all`` doubling the size of the
    resulting files.

6. ``basedir`` --- relative path to directory where outputs will be
   written:

    *string*

7. ``basename`` --- pattern of output names:

    *string*

8. ``precision`` --- output file number precision:

    ``single`` | ``double``

    The default is ``single``. Note that this only impacts the output,
    with statistic accumulation *always* being performed in double
    precision.

9. ``region`` --- region to be written, specified as either the
   entire domain using ``*``, a combination of the geometric shapes
   specified in :ref:`regions`, or a sub-region of elements that have
   faces on a specific domain boundary via the name of the domain
   boundary:

    ``*`` | ``shape(args, ...)`` | *string*

10. ``avg``-*name* --- expression to time average, written as a
    function of the primitive variables and gradients thereof;
    multiple expressions, each with their own *name*, may be specified:

    *string*

11. ``fun-avg``-*name* --- expression to compute at file output time,
    written as a function of any ordinary average terms; multiple
    expressions, each with their own *name*, may be specified:

    *string*

Example::

    [soln-plugin-tavg]
    nsteps = 10
    dt-out = 2.0
    mode = windowed
    basedir = .
    basename = files-{t:06.2f}

    avg-u = u
    avg-v = v
    avg-uu = u*u
    avg-vv = v*v
    avg-uv = u*v

    fun-avg-upup = uu - u*u
    fun-avg-vpvp = vv - v*v
    fun-avg-upvp = uv - u*v

This plugin also exposes functionality via a CLI. The following
functions are available

1. ``pyfr tavg merge`` --- average together multiple time average
   files into a single time average file. The averaging times are
   read from the file and do not need to be evenly spaced in time.

   Example::

    pyfr tavg merge avg-1.00.pyfrs avg-2.00.pyfrs avg-10.00.pyfrs merged_avg.pyfrs
