<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% invsq2 = 2**-0.5 %>

<%pyfr:macro name='WU_dot_div-cartesian' params='div, Phi, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;
  fpdtype_t invcsq = invc*invc;

  fpdtype_t nx = norm_nl[0];
  fpdtype_t ny = norm_nl[1];

  fpdtype_t gmo = ${c['gamma'] - 1.0};

  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

% if ndims == 2:

  Phi[0] = div[0] - gmo*invcsq*(div[0]*k - div[1]*v[0] - div[2]*v[1] + div[3]);
  Phi[1] = invrho*(div[0]*(ny*v[0] - nx*v[1]) - div[1]*ny + div[2]*nx);
  Phi[2] = ${invsq2}*invc*invrho*(gmo*(div[3] + div[0]*k - (div[1]*v[0] + div[2]*v[1])) + c*(div[1]*nx + div[2]*ny - div[0]*(nx*v[0] + ny*v[1])));
  Phi[3] = ${invsq2}*invc*invrho*(gmo*(div[3] + div[0]*k - (div[1]*v[0] + div[2]*v[1])) - c*(div[1]*nx + div[2]*ny - div[0]*(nx*v[0] + ny*v[1])));

% elif ndims == 3:

  fpdtype_t nz = norm_nl[2];

  Phi[0] = gmo*nx*invcsq*(-div[4] - div[0]*k + div[1]*v[0] + div[2]*v[1] + div[3]*v[2]) + invrho*(-div[3]*ny + div[2]*nz + div[0]*(nx*rho - nz*v[1] + ny*v[2]));
  Phi[1] = gmo*ny*invcsq*(-div[4] - div[0]*k + div[1]*v[0] + div[2]*v[1] + div[3]*v[2]) + invrho*( div[3]*nx - div[1]*nz + div[0]*(ny*rho + nz*v[0] - nx*v[2]));
  Phi[2] = gmo*nz*invcsq*(-div[4] - div[0]*k + div[1]*v[0] + div[2]*v[1] + div[3]*v[2]) + invrho*(-div[2]*nx + div[1]*ny + div[0]*(nz*rho - ny*v[0] + nx*v[1]));
  Phi[3] = invc*invrho*${invsq2}*(gmo*(div[4] + div[0]*k - (div[1]*v[0] + div[2]*v[1] + div[3]*v[2])) + c*(div[1]*nx + div[2]*ny + div[3]*nz - div[0]*(nx*v[0] + ny*v[1] + nz*v[2])));
  Phi[4] = invc*invrho*${invsq2}*(gmo*(div[4] + div[0]*k - (div[1]*v[0] + div[2]*v[1] + div[3]*v[2])) - c*(div[1]*nx + div[2]*ny + div[3]*nz - div[0]*(nx*v[0] + ny*v[1] + nz*v[2])));

% endif
</%pyfr:macro>


<%pyfr:macro name='WUinv_dot_Phi-cartesian' params='Phi, div, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;

  fpdtype_t nx = norm_nl[0];
  fpdtype_t ny = norm_nl[1];

  fpdtype_t gmo = ${c['gamma'] - 1.0};

  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

  % if ndims == 2:

  div[0] = Phi[0] + ${invsq2}*rho*invc*(Phi[2] + Phi[3]);
  div[1] = -Phi[1]*ny*rho + Phi[0]*v[0] + ${invsq2}*rho*invc*(Phi[2]*(c*nx + v[0]) - Phi[3]*(c*nx - v[0]));
  div[2] =  Phi[1]*nx*rho + Phi[0]*v[1] + ${invsq2}*rho*invc*(Phi[2]*(c*ny + v[1]) - Phi[3]*(c*ny - v[1]));
  div[3] = k*Phi[0] - Phi[1]*rho*(ny*v[0] - nx*v[1]) + ${invsq2}*rho*((Phi[3] + Phi[2])*(c/gmo + k*invc) + (Phi[2] - Phi[3])*(nx*v[0] + ny*v[1]));

  % elif ndims == 3:
  fpdtype_t nz = norm_nl[2];

  div[0] = Phi[0]*nx + Phi[1]*ny + Phi[2]*nz + ${invsq2}*invc*rho*(Phi[3] + Phi[4]);
  div[1] = rho*(Phi[2]*ny - Phi[1]*nz) + Phi[0]*nx*v[0] + Phi[1]*ny*v[0] + Phi[2]*nz*v[0] + ${invsq2}*invc*rho*(Phi[3]*(c*nx+v[0]) + Phi[4]*(-c*nx + v[0]));
  div[2] = rho*(Phi[0]*nz - Phi[2]*nx) + Phi[0]*nx*v[1] + Phi[1]*ny*v[1] + Phi[2]*nz*v[1] + ${invsq2}*invc*rho*(Phi[3]*(c*ny+v[1]) + Phi[4]*(-c*ny + v[1]));
  div[3] = rho*(Phi[1]*nx - Phi[0]*ny) + Phi[0]*nx*v[2] + Phi[1]*ny*v[2] + Phi[2]*nz*v[2] + ${invsq2}*invc*rho*(Phi[3]*(c*nz+v[2]) + Phi[4]*(-c*nz + v[2]));
  div[4] = Phi[0]*(k*nx + nz*rho*v[1] - ny*rho*v[2]) +
          Phi[1]*(k*ny - nz*rho*v[0] + nx*rho*v[2]) +
          Phi[2]*(k*nz + ny*rho*v[0] - nx*rho*v[1]) +
        rho*invc*${invsq2}/gmo*(Phi[3]*(c*c + gmo*k + c*gmo*(nx*v[0] + ny*v[1] + nz*v[2]))
                              + Phi[4]*(c*c + gmo*k - c*gmo*(nx*v[0] + ny*v[1] + nz*v[2])));

  % endif
</%pyfr:macro>


<%pyfr:macro name='WU_dot_div-normal' params='div, Phi, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;
  fpdtype_t invcsq = invc*invc;
  fpdtype_t gmo = ${c['gamma'] - 1.0};
  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

  fpdtype_t un = ${pyfr.dot('norm_nl[{i}]','v[{i}]', i=ndims)};
  fpdtype_t ut1 = ${pyfr.dot('t1[{i}]','v[{i}]', i=ndims)};
  % if ndims == 3:
  fpdtype_t ut2 = ${pyfr.dot('t2[{i}]','v[{i}]', i=ndims)};
  % endif

  fpdtype_t div_v = ${pyfr.dot('div[{i} + 1]','v[{i}]', i=ndims)};
  fpdtype_t div_norm = ${pyfr.dot('div[{i} + 1]','norm_nl[{i}]', i=ndims)};
  fpdtype_t div_t1 = ${pyfr.dot('div[{i} + 1]','t1[{i}]', i=ndims)};
  % if ndims == 3:
  fpdtype_t div_t2 = ${pyfr.dot('div[{i} + 1]','t2[{i}]', i=ndims)};
  % endif

  Phi[0] = div[0] - gmo*invcsq*(div[0]*k - div_v + div[${nvars - 1}]);

  % for i in range(ndims - 1):
  Phi[${i + 1}] = invrho*(div_t${i+1} - div[0]*ut${i+1});
  % endfor

  Phi[${nvars - 2}] = ${invsq2}*invc*invrho*(gmo*(div[0]*k - div_v + div[${nvars - 1}]) + c*(div_norm - div[0]*un));
  Phi[${nvars - 1}] = ${invsq2}*invc*invrho*(gmo*(div[0]*k - div_v + div[${nvars - 1}]) - c*(div_norm - div[0]*un));
</%pyfr:macro>

<%pyfr:macro name='WUinv_dot_Phi-normal' params='Phi, div, u, p, v'>
  fpdtype_t rho = u[0];
  fpdtype_t invrho = 1.0/rho;
  fpdtype_t c = sqrt(${c['gamma']}*p*invrho);
  fpdtype_t invc = 1.0/c;
  fpdtype_t gmo = ${c['gamma'] - 1.0};
  fpdtype_t k = 0.5*(${pyfr.dot('v[{i}]', i=ndims)});

  fpdtype_t un = ${pyfr.dot('norm_nl[{i}]','v[{i}]', i=ndims)};
  fpdtype_t ut1 = ${pyfr.dot('t1[{i}]','v[{i}]', i=ndims)};
  % if ndims == 3:
  fpdtype_t ut2 = ${pyfr.dot('t2[{i}]','v[{i}]', i=ndims)};
  % endif

  div[0] = Phi[0] + ${invsq2}*rho*invc*(Phi[${nvars - 2}] + Phi[${nvars - 1}]);

  % for i in range(ndims):
  div[${i + 1}] =  Phi[0]*v[${i}] + rho*(${'+'.join([f'Phi[{j+1}]*t{j+1}[{i}]' for j in range(ndims-1)])}) + ${invsq2}*rho*invc*(Phi[${nvars-2}]*(c*norm_nl[${i}] + v[${i}]) - Phi[${nvars - 1}]*(c*norm_nl[${i}] - v[${i}]));
  % endfor

  div[${nvars - 1}] = k*Phi[0] +
                     rho*(${'+'.join([f'Phi[{j+1}]*ut{j+1}' for j in range(ndims-1)])}) +
                     ${invsq2}*rho*(
                                    (Phi[${nvars - 2}] + Phi[${nvars - 1}])*(c/gmo + k*invc) +
                                    (Phi[${nvars - 2}] - Phi[${nvars - 1}])*un
                                   );

</%pyfr:macro>


<%pyfr:macro name='WUinv_dot_Phi-noop' params='Phi, div, u, p, v'>
% for var in range(nvars):
div[${var}] = Phi[${var}];
% endfor
</%pyfr:macro>
<%pyfr:macro name='WU_dot_div-noop' params='div, Phi, u, p, v'>
% for var in range(nvars):
Phi[${var}] = div[${var}];
% endfor
</%pyfr:macro>
