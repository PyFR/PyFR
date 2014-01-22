# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Enable support for double precision
#pragma OPENCL EXTENSION cl_khr_fp64: enable

// Typedefs
typedef ${pyfr.npdtype_to_ctype(fpdtype)} fpdtype_t;

${next.body()}
