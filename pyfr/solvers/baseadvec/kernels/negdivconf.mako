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
if (affected > 0.0){
// Turbulent characteristic lengths (radii of influence)
fpdtype_t lturbref[${ndims}] = ${lturbref};

// Initialize the utilde to 0.0
fpdtype_t utilde[${ndims}] = {0.0};

// Working variables
fpdtype_t arg;
fpdtype_t csi[${ndims}];

// Compute csi_max[3][3] on the fly for this point as lturb/lturb ref.
// make sure the minimum value is not less than cmm for stability reasons.
fpdtype_t csimax[${ndims}][${ndims}] = {{1.0}};
% for i in range(ndims):
    % for j in range(ndims):
        csimax[${i}][${j}] = min(max(${lturbex[i][j]}/lturbref[${i}], ${cmm}), 1.0);
    % endfor
% endfor

// Loop over the eddies
for (int n=0; n<${N}; n++){
    // printf("Eddy: t=%f, eddies_loc=(%f, %f, %f), n=%d\n", t, eddies_loc[0][n], eddies_loc[1][n], eddies_loc[2][n], n);

    //U,V,W
    csi[2] = fabs((ploc[2] - eddies_loc[2][n])/lturbref[2]);
    % for j in range(ndims):
        if (csi[2] < csimax[2][${j}]){

            csi[1] = fabs((ploc[1] - eddies_loc[1][n])/lturbref[1]);
            if (csi[1] < csimax[1][${j}]){

                csi[0] = fabs((ploc[0] - eddies_loc[0][n])/lturbref[0]);
                if (csi[0] < csimax[0][${j}]){

                    arg = 0.0;
                    % for i in range(ndims):
                        arg += csi[${i}]*csi[${i}];
                    % endfor
                    arg *= ${arg_const};

                    // Accumulate taking into account this components strength
                    utilde[${j}] += gauss_const[${j}]*exp(arg)*eddies_strength[${j}][n];
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


// Add density and energy fluctuations (compressible solver only, of course).
% if system == 'compr':
    // density
    fpdtype_t rM2 = ${rhomeanex}*pow(${Mmeanex}, 2);
    tdivtconf[0] += factor[${Ubulkdir}]*${rhofluctfactor}*rM2*utilde[${Ubulkdir}];

    // energy equation
    fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};
    tdivtconf[${ndims} + 1] += factor[${Ubulkdir}]*0.5*u[0]*udotu_fluct;
% endif
}
% endif
</%pyfr:kernel>
