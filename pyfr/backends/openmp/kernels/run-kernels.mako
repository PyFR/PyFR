# -*- coding: utf-8 -*-
<%inherit file='base'/>

struct kfunargs
{
    void (*fun)(void *);
    void *args;
};

void run_kernels(int off, int n, const struct kfunargs *kfa)
{
    for (int i = off; i < off + n; i++)
        kfa[i].fun(kfa[i].args);
}
