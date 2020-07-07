# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% t_tol = 0.99 %>

// Transforms from m=[1,0,0]^T
// See Moler and Hughes 1999
<%pyfr:macro name='transform_from' params='n,t,u,offset'>

% if ndims == 2:

    u[offset+0] = n[0]*t[offset+0] - n[1]*t[offset+1];
    u[offset+1] = n[1]*t[offset+0] + n[0]*t[offset+1];
    
% elif ndims == 3:

    u[0] = t[0];
    if (fabs(n[0]) < ${t_tol}){
        fpdtype_t h = 1./(1. + n[0]);

        u[offset+0] =  n[0]*t[offset+0] - n[1]*t[offset+1] - n[2]*t[offset+2];
    	u[offset+1] =  n[1]*t[offset+0] + (n[0] + h*n[2]*n[2])*t[offset+1] - h*n[1]*n[2]*t[offset+2];
    	u[offset+2] =  n[2]*t[offset+0] - h*n[1]*n[2]*t[offset+1] + (n[0] + h*n[1]*n[1])*t[offset+2];
    }
    else if (fabs(n[1]) < fabs(n[2])){
        fpdtype_t h = 1./(1. - n[1]);
	
        u[offset+0] = n[0]*t[offset+0] +  (1. - h*n[0]*n[0])*t[offset+1] - h*n[0]*n[2]*t[offset+2];
	u[offset+1] = n[1]*t[offset+0] + n[0]*t[offset+1] + n[2]*t[offset+2];
	u[offset+2] = n[2]*t[offset+0] - h*n[0]*n[2]*t[offset+1] + (1. - h*n[2]*n[2])*t[offset+2];
    }
    else{
       fpdtype_t h = 1./(1. - n[2]);
       
       u[offset+0] = n[0]*t[offset+0] - h*n[0]*n[1]*t[offset+1] + (1. - h*n[0]*n[0])*t[offset+2];
       u[offset+1] = n[1]*t[offset+0] + (1. - h*n[1]*n[1])*t[offset+1] - h*n[0]*n[1]*t[offset+2];
       u[offset+2] = n[2]*t[offset+0] + n[1]*t[offset+1] + n[0]*t[offset+2];
    }


% endif
</%pyfr:macro>