from statistics import NormalDist

import numpy as np


def tw_mean_var(t, x):
    # Time-weighted mean and variance via the trapezoidal rule
    dt = t[-1] - t[0]
    m = np.trapezoid(x, t, axis=0)/dt

    return m, np.trapezoid((x - m)**2, t, axis=0)/dt


def mser(t, x):
    # Marginal standard error rule initial-transient truncation point
    seg = np.diff(t)
    s1 = 0.5*(x[1:] + x[:-1])*seg
    s2 = 0.5*(x[1:]**2 + x[:-1]**2)*seg

    # Suffix trapezoidal integrals of x and x**2 over [t_d, t_N]
    i1 = np.concatenate([np.cumsum(s1[::-1])[::-1], [0]])
    i2 = np.concatenate([np.cumsum(s2[::-1])[::-1], [0]])

    # Estimates beyond the half-way point are unreliable; exclude them
    n = np.searchsorted(t, 0.5*(t[0] + t[-1]))

    # Truncated variance of the mean, assuming independent samples
    T = t[-1] - t[:n]
    g = (i2[:n] - i1[:n]**2/T)/T**2

    return int(g.argmin())


def tau_int(t, x, maxlag=None, nslots=200):
    # Integrated autocorrelation time and effective sample count
    T = t[-1] - t[0]
    maxlag = maxlag or T/10

    lags, r = acf(t, x, maxlag, nslots)

    # Empty slots yield NaN correlations; drop them before integrating
    v = ~np.isnan(r)
    lags, r = lags[v], r[v]

    # Integrate the correlogram up to its first zero crossing
    neg = np.flatnonzero(r < 0)
    n = neg[0] + 1 if len(neg) else len(r)
    tau = np.trapezoid(r[:n], lags[:n])

    # An uncorrelated series still only has its sample count
    tau = max(tau, T/(2*len(t)))

    return tau, T/(2*tau)


def mean_ci(t, x, level=0.68, maxlag=None):
    # Mean with an autocorrelation-aware confidence interval
    m, var = tw_mean_var(t, x)
    tau, neff = tau_int(t, x, maxlag)

    se = np.sqrt(var/neff)
    z = NormalDist().inv_cdf(0.5 + 0.5*level)

    return m, se, z*se, tau, neff


def _hann_wts(t, t0, twin):
    # Trapezoidal quadrature weights with a Hann taper in time
    h = 0.5 - 0.5*np.cos(2*np.pi*(t - t0)/twin)

    return h*np.gradient(t)


def _nudft_windows(t, twin, overlap, nyqfac, gapfac):
    # Yield (idx, wts, floc) for each admissible analysis window
    step = twin*(1 - overlap)

    t0 = t[0]
    while t0 + twin <= t[-1] + 1e-9*twin:
        m = (t >= t0) & (t <= t0 + twin)
        if m.sum() > 8:
            ts = t[m]
            dts = np.diff(ts)

            # Windows containing a data gap are discarded outright
            if dts.max() <= gapfac*np.median(dts):
                # Only frequencies below the local Nyquist are credible
                yield m, _hann_wts(ts, t0, twin), nyqfac*0.5/dts.max()
        t0 += step


def psd(t, x, twin=None, overlap=0.5, nyqfac=0.9, gapfac=5.0):
    # One-sided PSD via a quadrature-weighted nonuniform DFT with
    # Welch averaging over Hann-tapered windows of duration twin
    x = x.reshape(len(t), -1)
    if twin is None:
        twin = (t[-1] - t[0])/8

    # Frequencies up to the global median Nyquist rate
    nf = int(0.5*twin / np.median(np.diff(t)))
    f = np.arange(1, nf + 1) / twin

    acc, cnt = np.zeros((nf, x.shape[1])), np.zeros(nf)
    for m, w, floc in _nudft_windows(t, twin, overlap, nyqfac, gapfac):
        ts, xs = t[m], x[m]
        xs = xs - w @ xs / w.sum()

        v = f <= floc
        E = np.exp(-2j*np.pi*np.outer(ts, f[v]))
        X = (xs*w[:, None]).T @ E

        acc[v] += (np.abs(X)**2).T
        cnt[v] += 1

    if not cnt.max():
        raise ValueError('No admissible analysis windows')

    # One-sided factor 2 over the Hann-squared time integral of 3T/8
    v = cnt > 0
    return f[v], np.squeeze(16*acc[v]/(3*twin*cnt[v, None]))


def csd(t, x, y, twin=None, overlap=0.5, nyqfac=0.9, gapfac=5.0):
    # Auto and cross spectra of a signal pair on shared timestamps
    if twin is None:
        twin = (t[-1] - t[0])/8

    nf = int(0.5*twin / np.median(np.diff(t)))
    f = np.arange(1, nf + 1) / twin

    pxx, pyy = np.zeros(nf), np.zeros(nf)
    pxy = np.zeros(nf, dtype=complex)
    cnt = np.zeros(nf)
    for m, w, floc in _nudft_windows(t, twin, overlap, nyqfac, gapfac):
        ts = t[m]
        xs = x[m] - w @ x[m] / w.sum()
        ys = y[m] - w @ y[m] / w.sum()

        v = f <= floc
        E = np.exp(-2j*np.pi*np.outer(ts, f[v]))
        X, Y = (xs*w) @ E, (ys*w) @ E

        pxx[v] += np.abs(X)**2
        pyy[v] += np.abs(Y)**2
        pxy[v] += np.conj(X)*Y
        cnt[v] += 1

    if cnt.max() < 2:
        raise ValueError('Cross spectra require at least two windows')

    v = cnt >= 2
    return f[v], pxx[v]/cnt[v], pyy[v]/cnt[v], pxy[v]/cnt[v], cnt[v]


def tone(t, x, fmin, fmax, refine=8):
    # Locate the dominant tone via a floating-mean least-squares scan
    dt = t[-1] - t[0]

    # Coarse scan at the natural resolution, then a refined local scan
    f = np.arange(fmin, fmax, 0.5/dt)
    a = ls_amplitude(t, x, f)
    fpk = f[a.argmax()]

    f = np.linspace(fpk - 1/dt, fpk + 1/dt, 16*refine)
    a = ls_amplitude(t, x, f)

    return f[a.argmax()], a.max()


def ls_amplitude(t, x, f):
    # Floating-mean least-squares sinusoid amplitude at frequencies f
    w = np.gradient(t)
    w = w / w.sum()
    y = x - w @ x

    amps = np.empty(len(f))
    for k, fk in enumerate(f):
        wt = 2*np.pi*fk*t
        c, s = np.cos(wt), np.sin(wt)
        C, S = w @ c, w @ s
        YC, YS = w @ (y*c), w @ (y*s)
        CC, SS = w @ (c*c) - C*C, w @ (s*s) - S*S
        CS = w @ (c*s) - C*S

        D = CC*SS - CS*CS
        a = (YC*SS - YS*CS)/D
        b = (YS*CC - YC*CS)/D
        amps[k] = np.hypot(a, b)

    return amps


def _slot_pairs(t, maxlag, blk=512):
    # Yield index blocks (i, j) with 0 < t[j] - t[i] <= maxlag
    jhi = np.searchsorted(t, t + maxlag, side='right')

    for i0 in range(0, len(t), blk):
        i1 = min(i0 + blk, len(t))
        i = np.repeat(np.arange(i0, i1), jhi[i0:i1] - np.arange(i0, i1) - 1)
        j = np.concatenate([np.arange(k + 1, jhi[k]) for k in range(i0, i1)])

        if len(i):
            yield i, j


def _slot_acc(lag, vals, dtau, nslots):
    # Fuzzy slotting accumulator with triangular basis weights
    acc = np.zeros(nslots + 2)
    wts = np.zeros(nslots + 2)

    ell = lag / dtau
    k = np.minimum(ell.astype(int), nslots)
    b = 1 - (ell - k)

    np.add.at(acc, k, b*vals)
    np.add.at(wts, k, b)
    np.add.at(acc, k + 1, (1 - b)*vals)
    np.add.at(wts, k + 1, 1 - b)

    return acc[:nslots + 1], wts[:nslots + 1]


def acf(t, x, maxlag, nslots=200):
    # Slotted autocorrelation with local normalisation and fuzzy slots
    dtau = maxlag / nslots
    x = x - tw_mean_var(t, x)[0]

    num = np.zeros(nslots + 1)
    d1 = np.zeros(nslots + 1)
    d2 = np.zeros(nslots + 1)
    for i, j in _slot_pairs(t, maxlag):
        lag = t[j] - t[i]
        a, _ = _slot_acc(lag, x[i]*x[j], dtau, nslots)
        num += a
        d1 += _slot_acc(lag, x[i]*x[i], dtau, nslots)[0]
        d2 += _slot_acc(lag, x[j]*x[j], dtau, nslots)[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / np.sqrt(d1*d2)

    r[0] = 1
    return dtau*np.arange(nslots + 1), r


def xcf(t, x, y, maxlag, nslots=200):
    # Slotted cross-correlation over lags in [-maxlag, maxlag]
    dtau = maxlag / nslots
    x = x - tw_mean_var(t, x)[0]
    y = y - tw_mean_var(t, y)[0]

    nxy = np.zeros(nslots + 1)
    nyx = np.zeros(nslots + 1)
    wxy = np.zeros(nslots + 1)
    for i, j in _slot_pairs(t, maxlag):
        lag = t[j] - t[i]
        a, w = _slot_acc(lag, x[i]*y[j], dtau, nslots)
        nxy += a
        wxy += w
        nyx += _slot_acc(lag, y[i]*x[j], dtau, nslots)[0]

    # Also include the zero-lag products; ryx[0] is dropped on output
    nxy[0] += x @ y
    wxy[0] += len(x)

    with np.errstate(divide='ignore', invalid='ignore'):
        norm = wxy*np.sqrt(tw_mean_var(t, x)[1]*tw_mean_var(t, y)[1])
        rxy = np.where(wxy > 0, nxy / norm, 0)
        ryx = np.where(wxy > 0, nyx / norm, 0)

    # Positive lag means the first signal leads the second
    lags = dtau*np.arange(-nslots, nslots + 1)
    return lags, np.concatenate([ryx[:0:-1], rxy])


def structfun(t, x, orders, maxlag, nslots=200, signed=False):
    # Temporal structure functions via direct pair binning
    dtau = maxlag / nslots

    acc = np.zeros((len(orders), nslots + 1))
    wts = np.zeros(nslots + 1)
    for i, j in _slot_pairs(t, maxlag):
        lag = t[j] - t[i]
        d = x[j] - x[i]
        if not signed:
            d = np.abs(d)

        for k, n in enumerate(orders):
            a, w = _slot_acc(lag, d**n, dtau, nslots)
            acc[k] += a
        wts += w

    with np.errstate(divide='ignore', invalid='ignore'):
        sf = acc / wts

    tau = dtau*np.arange(nslots + 1)
    return tau[1:], sf[:, 1:]
