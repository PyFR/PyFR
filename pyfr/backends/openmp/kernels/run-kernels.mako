# -*- coding: utf-8 -*-
<%inherit file='base'/>

void run_kernels(int n, void (**kerns)(void *), void **kargs)
{
	for (int i = 0; i < n; i++)
		kerns[i](kargs[i]);
}
