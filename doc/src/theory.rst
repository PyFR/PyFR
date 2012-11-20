======
Theory
======

Introduction
------------

Theoretical studies and numerical experiments suggest that high-order
methods for unstructured grids can provide solutions to otherwise
intractable fluid flow problems within complex geometries. However,
despite their potential benefits, the use of unstructured high-order
methods remains limited in both academia and industry. There are various
reasons for this situation. These include difficulties generating
unstructured high-order meshes, problems resolving discontinuous
solutions such as shock waves, and the relative complexity of
unstructured high-order methods (compared with low-order schemes)
[VJ2011]_.

In an effort to address the latter issue of complexity, Huynh proposed
the flux reconstruction (FR) approach to high-order methods [H2007]_,
which provides a simple and intuitive framework within which various
well known high-order schemes can be cast. In particular, using the FR
approach one can recover both collocation based nodal discontinuous
Galerkin (DG) methods of the type described by Hesthaven and Warburton
[HW2008]_, and spectral difference (SD) methods (at least for a linear
flux function), which were originally proposed by Kopriva and Kolias
[KK1996]_, and later generalized by Liu, Vinokur and Wang [LVW2006]_.

In addition to recovering known methods, the FR approach also
facilitates the definition of new schemes. Huynh previously identified
several new unstructured high-order methods using the FR approach
[H2007]_ [H2009]_. Of particular note is the so called $g_2$ method,
which Huynh showed to be stable for various orders of accuracy (using
von Neumann analysis), and which can be made more efficient than other
FR methods via judicious placement of the solution points
\cite{huynh07}. Additionally, Huynh proposed various guidelines for
selecting the so called flux correction function, which determines
numerous properties of the associated FR scheme. In particular, for
one-dimensional (1D) FR schemes, Huynh suggested (based on von Neumann
analysis) that if a flux correction function of degree $k+1$ is
orthogonal to all polynomials of degree $k-2$ then the resulting scheme
will be linearly stable. Recently, Vincent, Castonguay and Jameson
proved this assertion to be true using an energy method [VCJ2011]_, and
consequently identified a range of FR schemes (parameterized by a single
scalar), which are guaranteed to be linearly stable for all orders of
accuracy. These linearly stable FR schemes will henceforth be referred
to as Vincent-Castonguay-Jameson-Huynh (VCJH) schemes.

Flux Reconstruction
-------------------

Overview
~~~~~~~~

FR schemes are similar to nodal DG schemes, which are arguably the most
popular type of unstructured high-order method (at least in the field of
computational aerodynamics). Like nodal DG schemes, FR schemes utilize a
high-order (nodal) polynomial basis to approximate the solution within
each element of the computational domain, and like nodal DG schemes, FR
schemes do not explicitly enforce inter-element solution continuity.
However, unlike nodal DG schemes, FR methods are based solely on the
governing system in a differential form. A description of the FR
approach in 1D is presented below. For further information see the
original paper of Huynh \cite{huynh07}.

Preliminaries
~~~~~~~~~~~~~

Consider solving the following 1D scalar conservation law .. math::
\frac{\partial u}{\partial t}+\frac{\partial f}{\partial x}=0
\label{governing} \end{equation} within an arbitrary domain
$\Omegabold$, where $x$ is a spatial coordinate, $t$ is time, $u=u(x,t)$
is a conserved scalar quantity and $f=f(u)$ is the flux of $u$ in the
$x$ direction. Additionally, consider partitioning $\Omegabold$ into $N$
distinct elements, each denoted $\Omegabold_n=\{x|x_n<x<x_{n+1}\}$, such
that

.. math::

    \Omega=\bigcup_{n=0}^{N-1}\Omega_n,\hspace{1cm}\bigcap_{n=0}^{N-1}\Omega_n=\emptyset.
    The FR approach requires $u$ is approximated in each $\Omegabold_n$
    by a function $u^{\delta}_n=u^{\delta}_n(x,t)$, which is a
    polynomial of degree $k$ within $\Omegabold_n$, and identically zero
    elsewhere. Additionally, the FR approach requires $f$ is
    approximated in each $\Omegabold_n$ by a function
    $f^{\delta}_n=f^{\delta}_n(x,t)$, which is a polynomial of degree
    $k+1$ within $\Omegabold_n$, and identically zero elsewhere.
    Consequently, when employing the FR approach, a total approximate
    solution $u^{\delta}=u^{\delta}(x,t)$ and a total approximate flux
    $f^{\delta}=f^{\delta}(x,t)$ can be defined within $\Omegabold$ as

.. math::

    u^{\delta}=\sum_{n=0}^{N-1}u_n^{\delta}\approx
    u,\hspace{1cm}f^{\delta}=\sum_{n=0}^{N-1}f_n^{\delta}\approx f,

where no level of inter-element continuity in $u^{\delta}$ is explicitly
enforced. However, $f^{\delta}$ is required to be C0 continuous at
element interfaces.

Note the requirement that each $f^{\delta}_n$ is one degree higher than
each $u^{\delta}_n$, which consequently ensures the divergence of
$f^{\delta}_n$ is of the same degree as $u^{\delta}_n$ within
$\Omegabold_n$.

Stages
~~~~~~

From an implementation perspective, it is advantageous to transform each
$\Omegabold_n$ to a standard element $\Omegabold_S=\{\hat{x}|-1\leq
\hat{x}\leq 1\}$ via the mapping

.. math:: \hat{x}=\Gamma_n(x)=2\left(\frac{x-x_n}{x_{n+1}-x_n}\right)-1,

which has the inverse

.. math::
x=\Gamma_n^{-1}(\hat{x})=\left(\frac{1-\hat{x}}{2}\right)x_n+\left(\frac{1+\hat{x}}{2}\right)x_{n+1}.

Having performed such a transformation, the evolution of $u_n^{\delta}$
within any individual $\Omegabold_n$ (and thus the evolution of
$u^{\delta}$ within $\Omegabold$) can be determined by solving the
following transformed equation within the standard element
$\Omegabold_S$

.. math::

    \frac{\partial\hat{u}^{\delta}}{\partial
    t}+\frac{\partial\hat{f}^{\delta}}{\partial\hat{x}}=0,
    \label{stand_ele_govern}

where

.. math::

    \hat{u}^{\delta}=\hat{u}^{\delta}(\hat{x},t)=u^{\delta}_n(\Gamma_n^{-1}(\hat{x}),t)

is a polynomial of degree $k$,

.. math::

    \hat{f}^{\delta}=\hat{f}^{\delta}(\hat{x},t)=\frac{f^{\delta}_n(\Gamma_n^{-1}(\hat{x}),t)}{h_n},

is a polynomial of degree $k+1$, and $h_n=(x_{n+1}-x_{n})/2$.

The FR approach to solving Eq. \eqref{stand_ele_govern} within the
standard element $\Omegabold_S$ can be described in five stages. The
first stage involves representing $\hat{u}^{\delta}$ in terms of a nodal
basis as follows

.. math::

    \hat{u}^{\delta}=\sum_{i=0}^{k}\hat{u}^{\delta}_{i}\;l_i,
    \label{trans_soln} where $l_i$ are Lagrange polynomials defined as

.. math::

    l_i=\prod_{j=0, j\neq
    i}^{k}\left(\frac{\hat{x}-\hat{x}_j}{\hat{x}_i-\hat{x}_j}\right),

$\hat{x}_i$ ($i=0$ to $k$) are $k+1$ distinct solution points within
$\Omegabold_S$, and $\hat{u}^{\delta}_{i}=\hat{u}^{\delta}_{i}(t)$
($i=0$ to $k$) are values of $\hat{u}^{\delta}$ at the solution points
$\hat{x}_i$.

The second stage of the FR approach involves constructing a degree $k$
polynomial $\hat{f}^{D}=\hat{f}^{D}(\hat{x},t)$, defined as the
approximate transformed discontinuous flux within $\Omegabold_S$.
Specifically, $\hat{f}^{D}$ is obtained via a collocation projection at
the $k+1$ solution points, and can hence be expressed as

.. math::

    \hat{f}^{D}=\sum_{i=0}^{k}\hat{f}^{D}_{i}\;l_i

where the coefficients $\hat{f}^{D}_{i}=\hat{f}^{D}_{i}(t)$ are simply
values of the transformed flux at each solution point $\hat{x}_i$
(evaluated directly from the approximate solution). The flux
$\hat{f}^{D}$ is termed discontinuous since it is calculated directly
from the approximate solution, which is in general piecewise
discontinuous between elements.

The third stage of the FR approach involves evaluating the approximate
solution at either end of the standard element $\Omegabold_S$
(\textit{i.e.} at $\hat{x}=\pm 1$). These values, in conjunction with
analogous information from adjoining elements, are then used to
calculate numerical interface fluxes. The exact methodology for
calculating such numerical interface fluxes will depend on the nature of
the equations being solved. For example, when solving the Euler
equations one may use a Roe type approximate Riemann solver
\cite{roe81}, or any other two-point flux formula that provides for an
upwind bias. In what follows the numerical interface fluxes associated
with the left and right hand ends of $\Omegabold_S$ (and transformed
appropriately for use in $\Omegabold_S$) will be denoted $\hat{f}^{I}_L$
and $\hat{f}^{I}_R$ respectively.

The penultimate stage of the FR approach involves constructing the
degree $k+1$ polynomial $\hat{f}^{\delta}$, by adding a correction flux
$\hat{f}^{C}=\hat{f}^{C}(\hat{x},t)$ of degree $k+1$ to $\hat{f}^{D}$,
such that their sum equals the transformed numerical interface flux at
$\hat{x}=\pm 1$, yet in some sense follows $\hat{f}^{D}$ within the
interior of $\Omegabold_S$. In order to define $\hat{f}^{C}$ such that
it satisfies the above requirements, consider first defining degree
$k+1$ correction functions $g_L=g_L(\hat{x})$ and $g_R=g_R(\hat{x})$ to
approximate zero (in some sense) within $\Omegabold_S$, as well as
satisfying

.. math::

    g_L(-1)=1,\hspace{0.5cm}g_L(1)=0, \label{gL_BCs}

.. math::

    g_R(-1)=0,\hspace{0.5cm}g_R(1)=1,

and

.. math::

    g_L(\hat{x})=g_R(-\hat{x}).

A suitable expression for $\hat{f}^{C}$ can now be written in terms of
$g_L$ and $g_R$ as

.. math::
\hat{f}^{C}=(\hat{f}^{I}_L-\hat{f}^{D}_L)g_L+(\hat{f}^{I}_R-\hat{f}^{D}_R)g_R,

where $\hat{f}^{D}_L=\hat{f}^{D}(-1,t)$ and
$\hat{f}^{D}_R=\hat{f}^{D}(1,t)$. Using this expression, the degree
$k+1$ approximate transformed total flux $\hat{f}^{\delta}$ within
$\Omegabold_S$ can be constructed from the discontinuous and correction
fluxes as follows

.. math::

    \hat{f}^{\delta}=\hat{f}^{D}+\hat{f}^{C}=\hat{f}^{D}+(\hat{f}^{I}_L-\hat{f}^{D}_L)g_L+(\hat{f}^{I}_R-\hat{f}^{D}_R)g_R.

The final stage of the FR approach involves evaluating the divergence of
$\hat{f}^{\delta}$ at each solution point $\hat{x}_i$ using the
expression

.. math::

    \frac{\partial\hat{f}^{\delta}}{\partial\hat{x}}(\hat{x}_i)=\sum_{j=0}^{k}\hat{f}^{D}_j\;\frac{\mathrm{d}l_j}{\mathrm{d}\hat{x}}(\hat{x}_i)+(\hat{f}^{I}_L-\hat{f}^{D}_L)\frac{\mathrm{d}g_{L}}{\mathrm{d}\hat{x}}(\hat{x}_i)+(\hat{f}^{I}_R-\hat{f}^{D}_R)\frac{\mathrm{d}g_{R}}{\mathrm{d}\hat{x}}(\hat{x}_i).

These values can then be used to advance $\hat{u}^{\delta}$ in time via
a suitable temporal discretization of the following semi-discrete
expression

.. math::

    \frac{\mathrm{d}\hat{u}^{\delta}_i}{\mathrm{d}t}=-\frac{\partial\hat{f}^{\delta}}{\partial
    \hat{x}}(\hat{x}_i).

Comments
~~~~~~~~

The nature of a particular FR scheme depends solely on three factors,
namely the location of the solution points $\hat{x}_i$, the methodology
for calculating the interface fluxes :math:`\hat{f}^{I}_L` and
$\hat{f}^{I}_R$, and the form of the flux correction functions $g_L$
(and thus $g_R$). Huynh \cite{huynh07} showed previously that a
collocation based nodal DG scheme is recovered in 1D if the corrections
functions $g_L$ and $g_R$ are the right and left Radau polynomials
respectively. Also, Huynh \cite{huynh07} showed that SD type methods can
be recovered (at least for a linear flux function) if the correction
functions $g_L$ and $g_R$ are set to zero at a set of $k$ points within
$\Omegabold_S$ (located symmetrically about the origin). Several
additional forms of $g_L$ (and thus $g_R$) have also suggested by Huynh
\cite{huynh07,huynh2009}, leading to the development of new schemes with
various stability and accuracy properties.

.. _VCJH:

Vincent-Castonguay-Jameson-Huynh Schemes
----------------------------------------

Overview
~~~~~~~~

VCJH schemes can be recovered if the corrections functions $g_L$ and
$g_R$ are defined as .. math::
g_L=\frac{(-1)^{k}}{2}\left[L_{k}-\left(\frac{\eta_{k}L_{k-1}+L_{k+1}}{1+\eta_k}\right)\right],
\label{final_left} \end{equation} and .. math::
g_R=\frac{1}{2}\left[L_{k}+\left(\frac{\eta_{k}L_{k-1}+L_{k+1}}{1+\eta_k}\right)\right],
\label{final_right} \end{equation} where .. math::
\eta_k=\frac{c(2k+1)(a_kk!)^2}{2}, \end{equation} .. math::
a_{k}=\frac{(2k)!}{2^{k}(k!)^2}, \end{equation} $L_k$ is a Legendre
polynomial of degree $k$ (normalized to equal unity at $\hat{x}=1$), and
$c$ is a free scalar parameter that must lie within the range

.. math::

    c_{-}<c<c_{\infty}, \label{c_range}

where

.. math::

    c_{-}=\frac{-2}{(2k+1)(a_kk!)^2}, and $c_{\infty}=\infty$.

Such correction functions ensure that if $\Omegabold$ is periodic the
resulting FR scheme will be linearly stable for any $k$ in the broken
Sobolev type norm $||u^{\delta}||_{k,2}$, defined as .. math::
||u^{\delta}||_{k,2}=\left[\sum_{n=1}^{N}\int_{x_n}^{x_{n+1}}(u_n^{\delta})^2+\frac{c}{2}(h_n)^{2k}\left(\frac{\partial^k
u_n^{\delta}}{\partial x^k}\right)^2\mathrm{d}x\right]^{1/2}.
\end{equation} For full details of how these schemes are derived see the
original paper of Vincent, Castonguay and Jameson \cite{vcj_jsc_2010}.

\subsection{Recovery of Known Methods}

Several known methods can be recovered from the range of VCJH schemes.
In particular if $c=c_{DG}$, where .. math:: c_{DG}=0, \end{equation}
then a collocation based nodal DG scheme is recovered
\cite{vcj_jsc_2010}. Alternatively, if $c=c_{SD}$ where

.. math::

    c_{SD}=\frac{2k}{(2k+1)(k+1)(a_kk!)^2},

then an SD method is recovered (at least for a linear flux function)
\cite{vcj_jsc_2010}.

It is in fact the only SD method that can be recovered from the range of
VCJH schemes. Further, it is identical to the SD scheme that Jameson
\cite{jameson10} proved to be linearly stable, which is the same as the
only SD scheme that Huynh found to be devoid of instabilities using von
Neumann analysis \cite{huynh07}. Finally, if $c=c_{HU}$ where

.. math::

    c_{HU}=\frac{2(k+1)}{(2k+1)k(a_kk!)^2},

then a so called $g_2$ FR method is recovered \cite{vcj_jsc_2010}. Such
a scheme was originally identified by Huynh \cite{huynh07} to be
particularly stable, and can be made more efficient than other FR
methods via judicious placement of the solution points. For further
details see the original paper of Huynh \cite{huynh07}.

.. [H2007] Huynh, H. T. (2007). A flux reconstruction approach to
   high-order schemes including discontinuous Galerkin methods. AIAA Paper
   2007-4079.

.. [H2009] Huynh, H. T. (2009). A reconstruction approach to high-order
   schemes including discontinuous {Galerkin} for diffusion. AIAA Paper
   2009-403.

.. [HW2008] Hesthaven, J. S., & Warburton, T. (2008). Nodal
   discontinuous Galerkin methods - Algorithms, analysis, and applications.
   Springer.

.. [KK1996] Kopriva, D. A., & Kolias, J. H. (1996). A conservative
   staggered-grid Chebyshev multidomain method for compressible flows. J
   Comput Phys, 125(1), 244-261.

.. [LVW2006] Liu, Y., Vinokur, M., & Wang, Z. J. (2006). Spectral
   difference method for unstructured grids {I}: {Basic} formulation. J
   Comput Phys, 216(2), 780-801.

.. [VJ2011] Vincent, P. E., & Jameson, A. (2011). Facilitating the
   adoption of unstructured high-order methods amongst a wider community of
   fluid dynamicists. Math Mod Nat Phenom, 6(3), 97-140.

.. [VCJ2011] Vincent, P. E., Castonguay, P., & Jameson, A. (2011). A new
   class of high-order energy stable flux reconstruction schemes. J Sci
   Comput, 47(1), 50-72.
