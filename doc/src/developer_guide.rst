.. highlightlang:: python

***************
Developer Guide
***************

=========
PyFR-Mako
=========

Overview
--------

Platform portability of PyFR is achieved, in part, via use of an
inbuilt templating language derived from `Mako
<http://www.makotemplates.org/>`_, henceforth referred to as PyFR-Mako.
Non-linear point-wise functionality is specified using PyFR-Mako.
PyFR-Mako specifications are then converted into platform specific
low-level code at runtime, which is compiled/linked/loaded.

PyFR-Mako Kernels
-----------------

PyFR-Mako kernels are specifications of point-wise functionality that
can be invoked directly from within PyFR. They are opened with a header
of the form::

    <%pyfr:kernel name='kernel-name' ndim='data-dimensionality' [argument-name='argument-intent argument-attribute argument-data-type' ...]>
                      
where 

1. ``kernel-name`` --- name of kernel

    *string*
    
2. ``data-dimensionality`` --- dimensionality of data

    *int*
    
3. ``argument-name`` --- name of argument

    *string*
    
4. ``argument-intent`` --- intent of argument

    ``in`` | ``out`` | ``inout``
    
5. ``argument-attribute`` --- attribute of argument

    ``mpi`` | ``scalar`` | ``view``
        
6. ``argument-data-type`` --- data type of argument

    *string*
    
and are closed with a footer of the form::

     </%pyfr:kernel>

PyFR-Mako Macros
----------------

PyFR-Mako macros are specifications of point-wise functionality that
cannot be invoked directly from within PyFR, but can be embedded into
PyFR-Mako kernels. PyFR-Mako macros can be viewed as building blocks
for PyFR-mako kernels. They are opened with a header of the form::

    <%pyfr:macro name='macro-name' params='[parameter-name, ...]'>
                      
where

1. ``macro-name`` --- name of macro

    *string*
    
2. ``parameter-name`` --- name of parameter

    *string*
    
and are closed with a footer of the form::

    </%pyfr:macro>
    
PyFR-Mako macros are embedded within a kernel using an expression of
the following form::

    ${pyfr.expand('macro-name', ['parameter-name', ...])};
    
where

1. ``macro-name`` --- name of the macro

    *string*
    
2. ``parameter-name`` --- name of parameter

    *string*    
      
Syntax
------

Basic Functionality
^^^^^^^^^^^^^^^^^^^

Basic functionality can be expressed using a restricted subset of the C
programming language. Specifically, use of the following is allowed:

1. ``+,-,*,/`` --- basic arithmetic

2. ``sin, cos, tan`` --- basic trigonometric functions

3. ``exp`` --- exponential

4. ``pow`` --- power

5. ``fabs`` --- absolute value

6. ``output = ( condition ? satisfied : unsatisfied )`` --- ternary if

However, conditional if statements, as well as for/while loops, are
not allowed.

Expression Substitution
^^^^^^^^^^^^^^^^^^^^^^^

Mako expression substitution can be used to facilitate PyFR-Mako kernel
specification. A Python expression :code:`expression` prescribed thus
:code:`${expression}` is substituted for the result when the PyFR-Mako
kernel specification is interpreted at runtime.

Example::

        E = s[${nvars - 1}]

Conditionals
^^^^^^^^^^^^

Mako conditionals can be used to facilitate PyFR-Mako kernel
specification. Conditionals are opened with :code:`% if condition:` and
closed with :code:`% endif`. Note that such conditionals are evaluated
when the PyFR-Mako kernel specification is interpreted at runtime, they
are not embedded into the low-level kernel.

Example::

        % if ndims == 2:
            fout[0][1] += t_xx;     fout[1][1] += t_xy;
            fout[0][2] += t_xy;     fout[1][2] += t_yy;
            fout[0][3] += u*t_xx + v*t_xy + ${-c['mu']*c['gamma']/c['Pr']}*T_x;
            fout[1][3] += u*t_xy + v*t_yy + ${-c['mu']*c['gamma']/c['Pr']}*T_y;
        % endif
     
Loops
^^^^^

Mako loops can be used to facilitate PyFR-Mako kernel specification.
Loops are opened with :code:`% for condition:` and closed with :code:`%
endfor`. Note that such loops are unrolled when the PyFR-Mako kernel
specification is interpreted at runtime, they are not embedded into the
low-level kernel.

Example::

        % for i in range(ndims):
            rhov[${i}] = s[${i + 1}];
            v[${i}] = invrho*rhov[${i}];
        % endfor
