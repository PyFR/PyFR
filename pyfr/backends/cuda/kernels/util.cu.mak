# -*- coding: utf-8 -*-

<%def name="dot(l,r)" filter="trim">
<%
    lr = '({})*({})'.format(l, r)
    return '(' + ' + '.join(lr.format(k) for k in range(ndims)) + ')'
%>
</%def>
