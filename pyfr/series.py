from statistics import NormalDist

import numpy as np


# Complex entries per transform matrix block
NUDFT_BLOCK = 1 << 22


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


def _piecewise_integral(t, x, q):
    # Integral of the piecewise-linear x from t[0] to each point in q
    dt = np.diff(t)
    seg = 0.5*(x[1:] + x[:-1])*dt
    acc = np.concatenate([[0], np.cumsum(seg)])

    # Locate the interval holding each q, with the last closing at t[-1]
    i = np.searchsorted(t, q, side='right') - 1
    at_end = i == len(t) - 1
    i = np.minimum(i, len(t) - 2)

    # Whole intervals below q, plus the part swept out inside the last
    dq, dx, dt = q - t[i], x[i + 1] - x[i], t[i + 1] - t[i]
    slope = np.divide(dx, dt, out=np.zeros_like(dq), where=dt != 0)
    vals = acc[i] + x[i]*dq + 0.5*slope*dq*dq

    return np.where(at_end, acc[-1], vals)


def _batch_lrv(t, x, mean, batch):
    # Estimate the long-run variance from half-overlapping time batches
    T = t[-1] - t[0]
    step = 0.5*batch
    nwin = int(np.floor((T - batch) / step)) + 1
    starts = t[0] + step*np.arange(nwin)

    # Time-weighted batch means from differences of the running integral
    ends = starts + batch
    ihi = _piecewise_integral(t, x, ends)
    ilo = _piecewise_integral(t, x, starts)
    bmean = (ihi - ilo) / batch

    return batch*T*np.var(bmean - mean, ddof=1) / (T - batch)


def tau_int(t, x, maxlag=None):
    # Integrated autocorrelation time and effective sample count
    if maxlag is not None and maxlag <= 0:
        raise ValueError('Maximum lag must be positive')

    T = t[-1] - t[0]
    mean, var = tw_mean_var(t, x)

    if var == 0:
        tau = T / (2*len(t))
    else:
        # Integrated timescale from multiscale physical-time batch means
        if np.any((dt := np.diff(t)) < 0):
            raise ValueError('Sample times must be nondecreasing')

        if not len(posdt := dt[dt > 0]):
            raise ValueError('Sample times must span a nonzero interval')

        if maxlag is None:
            maxbatch = T / 8
        else:
            maxbatch = min(maxlag, T / 4)
        if maxbatch <= 0:
            raise ValueError('Maximum lag must be positive')

        minbatch = min(maxbatch, max(T / 128, 8*np.median(posdt)))

        batches, batch = [], minbatch
        while batch <= maxbatch:
            batches.append(batch)
            batch *= 2
        if batches[-1] < 0.9*maxbatch:
            batches.append(maxbatch)

        lrvs = [_batch_lrv(t, x, mean, b) for b in batches]

        # Avoid optimistic cancellation at a single periodic batch duration
        lrv = max(lrvs[-min(3, len(lrvs)):])

        # An uncorrelated series still only has its sample count
        lrv = max(lrv, var*T / len(t))
        tau = lrv / (2*var)

    return tau, T/(2*tau)


def mean_ci(t, x, level=0.68, maxlag=None):
    # Mean with an autocorrelation-aware confidence interval
    m, var = tw_mean_var(t, x)
    tau, neff = tau_int(t, x, maxlag)

    se = np.sqrt(var/neff)
    z = NormalDist().inv_cdf(0.5 + 0.5*level)

    return m, se, z*se, tau, neff


def _hann_wts(ts, twin):
    # Trapezoidal quadrature weights with a Hann taper in time
    h = 0.5 - 0.5*np.cos(2*np.pi*ts / twin)

    return h*np.gradient(ts)


def _nudft_windows(t, twin, overlap, nyqfac, gapfac):
    # Yield the mask, times, weights and frequency limit per window
    step = twin*(1 - overlap)

    t0 = t[0]
    while t0 + twin <= t[-1] + 1e-9*twin:
        m = (t >= t0) & (t <= t0 + twin)
        if m.sum() > 8:
            # Times are taken from the window origin to keep phases small
            ts = t[m] - t0
            dts = np.diff(ts)

            # Windows containing a data gap are discarded outright
            if dts.max() <= gapfac*np.median(dts):
                # Only frequencies below the local Nyquist are credible
                yield m, ts, _hann_wts(ts, twin), nyqfac*0.5 / dts.max()
        t0 += step


def _nudft_blocks(t, f0, df, nf):
    # Yield (lo, hi, E) with E[:, k - lo] = exp(-2i*pi*t*(f0 + k*df))
    if not (blk := min(nf, max(1, NUDFT_BLOCK // len(t)))):
        return

    # Powers of w by repeated doubling, rather than one exp per entry
    w = np.exp(-2j*np.pi*df*t)
    v = np.empty((len(t), blk), dtype=complex)
    v[:, 0] = 1

    m = 1
    while m < blk:
        k = min(m, blk - m)
        v[:, m:m + k] = v[:, :k]*(v[:, m - 1]*w)[:, None]
        m += k

    u = v[:, -1]*w
    c = np.exp(-2j*np.pi*f0*t)

    for lo in range(0, nf, blk):
        hi = min(lo + blk, nf)
        yield slice(lo, hi), c[:, None]*v[:, :hi - lo]
        c *= u


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
    for m, ts, w, floc in _nudft_windows(t, twin, overlap, nyqfac, gapfac):
        xs = x[m]
        xs = (xs - w @ xs / w.sum())*w[:, None]

        nv = np.searchsorted(f, floc, side='right')
        for sl, E in _nudft_blocks(ts, 1 / twin, 1 / twin, nv):
            acc[sl] += (np.abs(xs.T @ E)**2).T

        cnt[:nv] += 1

    if not cnt.max():
        raise ValueError('No admissible analysis windows')

    # One-sided factor 2 over the Hann-squared time integral of 3T/8
    v = cnt > 0
    return f[v], np.squeeze(16*acc[v]/(3*twin*cnt[v, None]))


def csd(t, x, y, twin=None, overlap=0.5, nyqfac=0.9, gapfac=5.0):
    # Auto and cross spectra of x against every column of y at times t
    y = y.reshape(len(t), -1)
    twin = twin or (t[-1] - t[0]) / 8

    nf = int(0.5*twin / np.median(np.diff(t)))
    f = np.arange(1, nf + 1) / twin

    pxx, pyy = np.zeros(nf), np.zeros((nf, y.shape[1]))
    pxy = np.zeros((nf, y.shape[1]), dtype=complex)
    cnt = np.zeros(nf)
    for m, ts, w, floc in _nudft_windows(t, twin, overlap, nyqfac, gapfac):
        xs, ys = x[m], y[m]
        xs = (xs - w @ xs / w.sum())*w
        ys = (ys - w @ ys / w.sum())*w[:, None]

        nv = np.searchsorted(f, floc, side='right')
        for sl, E in _nudft_blocks(ts, 1 / twin, 1 / twin, nv):
            X, Y = xs @ E, ys.T @ E

            pxx[sl] += np.abs(X)**2
            pyy[sl] += (np.abs(Y)**2).T
            pxy[sl] += (np.conj(X)*Y).T

        cnt[:nv] += 1

    if cnt.max() < 2:
        raise ValueError('Cross spectra require at least two windows')

    # Cross terms stay one column per signal, even when there is just one
    v = cnt >= 2
    return (f[v], pxx[v] / cnt[v], pyy[v] / cnt[v, None],
            pxy[v] / cnt[v, None], cnt[v])


def tone(t, x, fmin, fmax, refine=128):
    # Locate the dominant tone via a floating-mean least-squares scan
    if fmin >= fmax:
        raise ValueError('Minimum frequency must be less than maximum')

    dt = t[-1] - t[0]

    # Coarse scan at the natural resolution, then a refined local scan
    f = np.arange(fmin, fmax, 0.5/dt)
    a = ls_amplitude(t, x, f)
    fpk = f[a.argmax()]

    # Keep the refined scan inside the requested band
    flo, fhi = max(fpk - 1 / dt, fmin), min(fpk + 1 / dt, fmax)
    f = np.linspace(flo, fhi, refine)
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


def _affine_sums(pfx, lo, hi, a, b):
    # Sums of the affine weights a*t + b, alone and against each payload
    hi = np.maximum(hi, lo)
    n = hi - lo

    pt, *pv = pfx
    out = np.empty((1 + len(pv)//2, len(n)))
    np.add(a*(pt[hi] - pt[lo]), b*n, out=out[0])

    for k, (p, ptv) in enumerate(zip(pv[::2], pv[1::2]), start=1):
        np.add(a*(ptv[hi] - ptv[lo]), b*(p[hi] - p[lo]), out=out[k])

    return out


def _tri_prefix(t, vals):
    # Prefix sums of t and of every payload, both plain and against t
    csum = lambda v: np.concatenate([[0], np.cumsum(v)])

    pfx = [csum(t)]
    for v in vals:
        pfx += [csum(v), csum(t*v)]

    return pfx


def _tri_slots(t, pfx, maxlag, nslots):
    # Triangular fuzzy-slot weight sums, yielded one slot at a time
    dtau = maxlag / nslots
    idx = np.arange(len(t))

    # Slot zero has no rising flank, so any edge at or below i will do
    el = ec = idx

    for s in range(nslots + 1):
        l = max(0, (c := s*dtau) - dtau)

        # Rising flank of the triangle, covering lags l up to c
        b = -(t + l) / dtau
        slot = _affine_sums(pfx, np.maximum(idx + 1, el), ec, 1 / dtau, b)

        # Falling flank up to u, an edge the next slot reuses as centre
        u = min(maxlag, c + dtau)
        eu = np.searchsorted(t, t + u)
        b = (t + c + dtau) / dtau
        slot += _affine_sums(pfx, np.maximum(idx + 1, ec), eu, -1 / dtau, b)

        yield slot

        # Adjacent slots share an edge, so carry it over rather than redo it
        el, ec = ec, eu


def _tri_acc(t, x, maxlag, nslots):
    # Per-slot cross term and the two local normalisers needed by acf
    x2 = x*x

    # Rebasing to zero keeps the prefix differences well conditioned
    t = t - t[0]
    pfx = _tri_prefix(t, [x, x2])

    num, d1, d2 = np.zeros((3, nslots + 1))
    for s, (sw, swx, swx2) in enumerate(_tri_slots(t, pfx, maxlag, nslots)):
        num[s], d1[s], d2[s] = x @ swx, x2 @ sw, swx2.sum()

    return num, d1, d2


def _tri_xacc(t, x, y, maxlag, nslots):
    # As _tri_acc but for the cross terms between two distinct signals
    t = t - t[0]
    pfx = _tri_prefix(t, [x, *y.T])

    nxy, nyx = np.zeros((2, nslots + 1, y.shape[1]))
    wts = np.zeros(nslots + 1)
    for s, slot in enumerate(_tri_slots(t, pfx, maxlag, nslots)):
        sw, swx, swy = slot[0], slot[1], slot[2:]
        nxy[s], nyx[s], wts[s] = swy @ x, y.T @ swx, sw.sum()

    return nxy, nyx, wts


def acf(t, x, maxlag, nslots=200):
    # Slotted autocorrelation with local normalisation and fuzzy slots
    dtau = maxlag / nslots

    x = x - tw_mean_var(t, x)[0]
    num, d1, d2 = _tri_acc(t, x, maxlag, nslots)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / np.sqrt(d1*d2)

    r[0] = 1
    return dtau*np.arange(nslots + 1), r


def xcf(t, x, y, maxlag, nslots=200):
    # Slotted cross-correlation of a reference against one or more signals
    y = y.reshape(len(t), -1)
    dtau = maxlag / nslots
    (mx, vx), (my, vy) = tw_mean_var(t, x), tw_mean_var(t, y)
    x, y = x - mx, y - my

    nxy, nyx, wxy = _tri_xacc(t, x, y, maxlag, nslots)

    # Also include the zero-lag products; nyx[0] is dropped on output
    nxy[0] += x @ y
    wxy[0] += len(x)

    # Positive lag means the first signal leads the second
    num = np.concatenate([nyx[:0:-1], nxy])
    w = np.concatenate([wxy[:0:-1], wxy])[:, None]

    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(w > 0, num / (w*np.sqrt(vx*vy)), 0)

    lags = dtau*np.arange(-nslots, nslots + 1)
    return lags, np.squeeze(r)


def structfun(t, x, orders, maxlag, nslots=200, signed=False):
    # Temporal structure functions via direct pair binning
    dtau = maxlag / nslots

    acc = np.zeros((len(orders), nslots + 1))
    wts = np.zeros(nslots + 1)

    # Expanding the difference into prefix sums cancels the high orders away
    for i, j in _slot_pairs(t, maxlag):
        lag, d = t[j] - t[i], x[j] - x[i]
        d = d if signed else np.abs(d)

        for k, n in enumerate(orders):
            a, w = _slot_acc(lag, d**n, dtau, nslots)
            acc[k] += a
        wts += w

    with np.errstate(divide='ignore', invalid='ignore'):
        sf = acc / wts

    tau = dtau*np.arange(nslots + 1)
    return tau[1:], sf[:, 1:]
