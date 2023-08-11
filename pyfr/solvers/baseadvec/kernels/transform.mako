<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

## Transforms to [1, 0, 0]^T from n
## See Moler and Hughes 1999
<%pyfr:macro name='transform_to' params='n, u, t, off'>
% if ndims == 2:
    t[off + 0] = n[0]*u[off + 0] + n[1]*u[off + 1];
    t[off + 1] = n[0]*u[off + 1] - n[1]*u[off + 0];
% elif ndims == 3:
    if (fabs(n[0]) < 0.99)
    {
        fpdtype_t h = 1/(1 + n[0]);

        t[off + 0] = n[0]*u[off + 0] + n[1]*u[off + 1] + n[2]*u[off + 2];
        t[off + 1] = (n[0] + h*n[2]*n[2])*u[off + 1] - n[1]*u[off + 0]
                     - h*n[1]*n[2]*u[off + 2];
        t[off + 2] = (n[0] + h*n[1]*n[1])*u[off + 2] - n[2]*u[off + 0]
                        - h*n[1]*n[2]*u[off + 1];
    }
    else if (fabs(n[1]) < fabs(n[2]))
    {
        fpdtype_t h = 1/(1 - n[1]);

        t[off + 0] = n[0]*u[off + 0] + n[1]*u[off + 1] + n[2]*u[off + 2];
        t[off + 1] = (1 - h*n[0]*n[0])*u[off + 0] + n[0]*u[off + 1]
                     - h*n[0]*n[2]*u[off + 2];
        t[off + 2] = n[2]*u[off + 1] - h*n[0]*n[2]*u[off + 0]
                     + (1 - h*n[2]*n[2])*u[off + 2];
    }
    else
    {
        fpdtype_t h = 1/(1 - n[2]);

        t[off + 0] = n[0]*u[off + 0] + n[1]*u[off + 1] + n[2]*u[off + 2];
        t[off + 1] = (1 - h*n[1]*n[1])*u[off + 1] - h*n[0]*n[1]*u[off + 0]
                     + n[1]*u[off + 2];
        t[off + 2] = (1 - h*n[0]*n[0])*u[off + 0] - h*n[0]*n[1]*u[off + 1]
                     + n[0]*u[off + 2];
    }
% endif
</%pyfr:macro>

## Transforms from [1, 0, 0]^T to n
<%pyfr:macro name='transform_from' params='n, t, u, off'>
% if ndims == 2:
    u[off + 0] = n[0]*t[off + 0] - n[1]*t[off + 1];
    u[off + 1] = n[1]*t[off + 0] + n[0]*t[off + 1];
% elif ndims == 3:
    if (fabs(n[0]) < 0.99)
    {
        fpdtype_t h = 1/(1 + n[0]);

        u[off + 0] = n[0]*t[off + 0] - n[1]*t[off + 1] - n[2]*t[off + 2];
        u[off + 1] = n[1]*t[off + 0] + (n[0] + h*n[2]*n[2])*t[off + 1]
                     - h*n[1]*n[2]*t[off + 2];
        u[off + 2] = n[2]*t[off + 0] - h*n[1]*n[2]*t[off + 1]
                     + (n[0] + h*n[1]*n[1])*t[off + 2];
    }
    else if (fabs(n[1]) < fabs(n[2]))
    {
        fpdtype_t h = 1/(1 - n[1]);

        u[off + 0] = n[0]*t[off + 0] +  (1 - h*n[0]*n[0])*t[off + 1]
                     - h*n[0]*n[2]*t[off + 2];
        u[off + 1] = n[1]*t[off + 0] + n[0]*t[off + 1] + n[2]*t[off + 2];
        u[off + 2] = n[2]*t[off + 0] - h*n[0]*n[2]*t[off + 1]
                     + (1 - h*n[2]*n[2])*t[off + 2];
    }
    else
    {
        fpdtype_t h = 1/(1 - n[2]);

        u[off + 0] = n[0]*t[off + 0] - h*n[0]*n[1]*t[off + 1]
                     + (1 - h*n[0]*n[0])*t[off + 2];
        u[off + 1] = n[1]*t[off + 0] + (1 - h*n[1]*n[1])*t[off + 1]
                     - h*n[0]*n[1]*t[off + 2];
        u[off + 2] = n[2]*t[off + 0] + n[1]*t[off + 1] + n[0]*t[off + 2];
    }
% endif
</%pyfr:macro>
