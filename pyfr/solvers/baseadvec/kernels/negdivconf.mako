# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// #include <stdio.h>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'>
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
% endfor

% if trsrc:
if (affected[0] > 0.0){
// Turbulent characteristic lengths (radii of influence)
fpdtype_t lturbref[${ndims}] = ${lturbref};

// Guassian constants
fpdtype_t GCsInv[${ndims}][${ndims}] = ${GCsInv};
fpdtype_t csimax[${ndims}][${ndims}] = ${csimax};
fpdtype_t OneOverSigmaProd = pow(${sigmaInv}, ${ndims});

// Initialize the utilde to 0.0
fpdtype_t utilde[${ndims}] = {0.0};

// Working variables
fpdtype_t g, arg;
fpdtype_t csi[${ndims}];

// Loop over the eddies
for (int n=0; n<${N}; n++){
    // printf("Eddy: t=%f, eddies_loc=(%f, %f, %f), n=%d\n", t, eddies_loc[0][n], eddies_loc[1][n], eddies_loc[2][n], n);

    //U,V,W
    % for j in range(ndims):
        csi[2] = fabs((ploc[2] - eddies_loc[2][n])/lturbref[2]);
        if (csi[2] < csimax[2][${j}]){

            csi[1] = fabs((ploc[1] - eddies_loc[1][n])/lturbref[1]);
            if (csi[1] < csimax[1][${j}]){

                csi[0] = fabs((ploc[0] - eddies_loc[0][n])/lturbref[0]);
                if (csi[0] < csimax[0][${j}]){

                    arg = 0.0;
                    g   = 1.0;
                    % for i in range(ndims):
                        g   *= GCsInv[${i}][${j}];
                        arg += pow(${sigmaInv}*csi[${i}], 2);
                    % endfor

                    g *= OneOverSigmaProd*exp(-0.5*arg);

                    // Accumulate taking into account this components strength
                    utilde[${j}] += g*eddies_strength[${j}][n];
                }
            }
        }
    % endfor
}

// order is important here.
utilde[2] = aij[3]*utilde[2];
utilde[1] = aij[1]*utilde[0] + aij[2]*utilde[1];
utilde[0] = aij[0]*utilde[0];

// source term for synthetic turbulence, only for the momentum equations for the
// moment. Multiply by the density to make it dimensionally consistent for a
// compressible solver.
% for i in range(ndims):
    % if system == 'compr':
        tdivtconf[${i} + 1] += u[0]*factor[${i}]*utilde[${i}];
    % else:
        tdivtconf[${i} + 1] += factor[${i}]*utilde[${i}];
    % endif
% endfor


// TODO add pressure (i.e. energy) and density fluctuations for Ma > 0.3 flows,
// (compressible solver only, of course).
}
% endif
</%pyfr:kernel>
