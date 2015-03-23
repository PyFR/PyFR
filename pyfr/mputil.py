# -*- coding: utf-8 -*-

import mpmath as mp


def jacobi(n, a, b, z):
    j = [1]

    if n >= 1:
        j.append(((a + b + 2)*z + a - b) / 2)
    if n >= 2:
        apb, bbmaa = a + b, b*b - a*a

        for q in range(2, n + 1):
            qapbpq, apbp2q = q*(apb + q), apb + 2*q
            apbp2qm1, apbp2qm2  = apbp2q - 1, apbp2q - 2

            aq = mp.mpf(apbp2q*apbp2qm1)/(2*qapbpq)
            bq = mp.mpf(apbp2qm1*bbmaa)/(2*qapbpq*apbp2qm2)
            cq = mp.mpf(apbp2q)*(a + q - 1)*(b + q - 1)/(qapbpq*apbp2qm2)

            # Update
            j.append((aq*z - bq)*j[-1] - cq*j[-2])

    return j


def jacobian(f, x):
    h = mp.sqrt(mp.eps)
    h2 = h / 2

    J = []
    for j in range(len(x)):
        xph2 = [xk if k != j else xk + h2 for k, xk in enumerate(x)]
        xmh2 = [xk if k != j else xk - h2 for k, xk in enumerate(x)]

        fxph2 = f(*xph2)
        fxmh2 = f(*xmh2)

        J.append([(fp - fm) / h for fp, fm in zip(fxph2, fxmh2)])

    return list(zip(*J))
