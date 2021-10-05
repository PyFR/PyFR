# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

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
fpdtype_t lturbrefinv[${ndims}] = ${lturbrefinv};

// Initialize the utilde to 0.0
fpdtype_t utilde[${ndims}] = {0.0};

// Working variables
fpdtype_t arg, arg2, pexp;
fpdtype_t csi[${ndims}];

// Compute csi_max[3][3] on the fly for this point as lturb/lturb ref.
// make sure the minimum value is not less than cmm for stability reasons.
fpdtype_t csimax[${ndims}][${ndims}] = {{1.0}};
% for i, j in pyfr.ndrange(ndims, ndims):
    csimax[${i}][${j}] = min(max(${lturbex[i][j]}*lturbrefinv[${i}], ${cmm}), 1.0);
% endfor

// Loop over the eddies
for (int n=0; n<${N}; n++){
    % for j in range(ndims):
        csi[${j}] = fabs((ploc[${j}] - eddies_loc[${j}][n])*lturbrefinv[${j}]);
    % endfor

    arg = 0.0;
    % for i in range(ndims):
        arg += csi[${i}]*csi[${i}];
    % endfor
    arg *= ${arg_const};

    arg2 = arg*arg;
    // polynomial approximation of exp(arg)
    pexp = 0.0006401098467575985*arg2*arg2*arg + 0.012770347332254533*arg2*arg2 + 0.10273980299300066*arg2*arg + 0.42908015717558884*arg2 + 0.9716933049901688*arg + 1.0;

    //U,V,W
    // Accumulate taking into account this components strength
    % for j in range(ndims):
        if (csi[2] < csimax[2][${j}]){
            if (csi[1] < csimax[1][${j}]){
                if (csi[0] < csimax[0][${j}]){
                    utilde[${j}] += gauss_const[${j}]*pexp*eddies_strength[${j}][n];
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
        tdivtconf[${i + 1}] += u[0]*factor[${i}]*utilde[${i}];
    % else:
        tdivtconf[${i + 1}] += factor[${i}]*utilde[${i}];
    % endif
% endfor


// Add density and energy fluctuations (compressible solver only, of course).
% if system == 'compr':
    // density
    fpdtype_t rM2 = ${rhomeanex}*${Mmeanex}*${Mmeanex};
    tdivtconf[0] += factor[${Ubulkdir}]*${rhofluctfactor}*rM2*utilde[${Ubulkdir}];

    // energy equation
    fpdtype_t udotu_fluct = ${pyfr.dot('utilde[{i}]', i=(0, ndims))};
    tdivtconf[${ndims + 1}] += factor[${Ubulkdir}]*0.5*u[0]*udotu_fluct;
% endif
}
% endif
</%pyfr:kernel>
