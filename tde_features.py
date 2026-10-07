"""
tde_features.py - light-curve feature families for the XGBoost TDE classifier.

Input: {band: (t, flux, flux_err)} (MJD, uJy, uJy; sorted, zeros removed) and the F99 extinction
A_lambda per band. Every band is median-subtracted and dust-corrected first. There is no redshift
for live objects, so everything is observer-frame (the old z_* family and the black-body R/L
features were always NaN and are gone).

Families (column prefix -> family)
  bazin_    Bazin fit: amplitude, t0 offset, rise/fall time, baseline, chi2
  potica_   "Potica-like" TDE model fit + colour differences between bands        (NOTE 1)
  shape_    rise time, FWHM before/after peak, asymmetry                          (NOTE 2)
  stat_     basic statistics of the S/N curve (max-S/N epoch, MAD, ...)
  mexhat_   Mexican-hat (Ricker) wavelet power spectrum
  flare_/base_  statistics of the flare part and of the baseline part
  bin_      phase bins around peak, seasonality, colour binning
  pl_       post-peak power-law decay slope and fit quality (vs. exponential)
  fft_      FFT of the linearly resampled curve
  stack_    stacked g+r+i light curve, statistics on 4 channels (flux, flux_norm, snr, snr_norm)
  av_       Avocado/PLAsTiCC-style: time to 80%/20% of peak, widths, S/N counts, per-band S/N
  bb_       cheap black-body fit (temperature only) from per-band amplitudes

NOTE 1: the Potica model equation isn't public; ``potica_*`` uses a stand-in
        (sigmoid rise x (1 + dt/tau)^-alpha). Replace ``_tde_shape`` if you get the real one.
NOTE 2: the original used a 2D Gaussian Process (redback); interpolation + light smoothing on a
        200-point grid is used instead.
"""
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, median_abs_deviation, skew

BANDS = ["u", "g", "r", "i", "z", "y"]
LAMBDA_NM = {"u": 367.0, "g": 482.0, "r": 622.0, "i": 754.0, "z": 869.0, "y": 971.0}
N_GRID = 200
SNR_SIG = 3.0

# --------------------------------------------------------------------------- #
# Feature name registry (fixed so every object returns the same columns)
# --------------------------------------------------------------------------- #
MEXHAT_WIDTHS = [2, 4, 8, 16, 32]
PHASE_EDGES = [-np.inf, -20.0, 0.0, 20.0, 60.0, np.inf]      # days from peak
COLOUR_EDGES = [0.0, 20.0, 60.0, np.inf]
_STACK_CHANNELS = ["flux", "fluxn", "snr", "snrn"]
_STACK_STATS = ["mean", "std", "max", "min", "mad"]

FAMILY_KEYS = {
    "bazin": ["bazin_amp", "bazin_t0_off", "bazin_trise", "bazin_tfall",
              "bazin_base", "bazin_chi2", "bazin_fall_rise"],
    "potica": ["potica_amp", "potica_t0_off", "potica_trise", "potica_tau", "potica_alpha",
               "potica_chi2", "potica_col_ug", "potica_col_gr", "potica_col_ri",
               "potica_col_iz", "potica_col_zy"],
    "shape": ["shape_rise_time", "shape_fwhm_pre", "shape_fwhm_post", "shape_fwhm",
              "shape_asym", "shape_peak_snr", "shape_end_frac", "shape_start_frac",
              "shape_peak_time_frac", "shape_band"],
    "stat": ["stat_maxsnr", "stat_t_maxsnr", "stat_t_maxsnr_frac", "stat_flux_maxsnr",
             "stat_band_maxsnr", "stat_mad_snr", "stat_median_snr", "stat_std_snr",
             "stat_skew_snr", "stat_kurt_snr", "stat_frac_pos", "stat_frac_neg",
             "stat_n_obs", "stat_span"],
    "mexhat": [f"mexhat_p{w}" for w in MEXHAT_WIDTHS] + ["mexhat_logpow", "mexhat_peak_scale"],
    "flare_baseline": ["flare_n", "flare_duration", "flare_dur_frac", "flare_mean_snr",
                       "flare_skew", "flare_kurt", "flare_area", "flare_rise_decay",
                       "flare_max_slope", "base_n", "base_std", "base_mad",
                       "base_max_minus_median", "base_amp_ratio", "base_chi2", "base_skew"],
    "binning": ([f"bin_phase_{k}" for k in range(len(PHASE_EDGES) - 1)]
                + [f"bin_season_{k}" for k in range(4)]
                + [f"bin_col_{p}_w{k}" for p in ("gr", "ri") for k in range(3)]
                + ["bin_col_gr_evol", "bin_col_ri_evol"]),
    "powerlaw": ["pl_alpha", "pl_r2", "pl_rmse", "pl_resid_53", "pl_exp_r2", "pl_r2_diff", "pl_n"],
    "fft": ["fft_logpow", "fft_pow_snr", "fft_dom_freq", "fft_dom_frac", "fft_low_frac",
            "fft_centroid", "fft_entropy"],
    "stack": ([f"stack_{s}_{c}" for c in _STACK_CHANNELS for s in _STACK_STATS]
              + ["stack_skew", "stack_kurt", "stack_n", "stack_tpeak_frac",
                 "stack_slope_pre", "stack_slope_post"]),
    "avocado": (["av_tfwd_80", "av_tfwd_20", "av_tbwd_80", "av_tbwd_20",
                 "av_pos_width", "av_neg_width", "av_abs_diff",
                 "av_frac_bkg", "av_frac_gt5", "av_frac_gt10", "av_frac_ltm5", "av_tw5", "av_tw10"]
                + [f"av_total_snr_{b}" for b in BANDS]),
    "blackbody": ["bb_logT", "bb_rchi2", "bb_logT_late", "bb_dlogT"],
}
FAMILY_ORDER = list(FAMILY_KEYS)          # also the order in which families are computed

_FAMILY_PREFIXES = {
    "bazin_": "bazin", "potica_": "potica", "shape_": "shape", "stat_": "stat",
    "mexhat_": "mexhat", "flare_": "flare_baseline", "base_": "flare_baseline",
    "bin_": "binning", "pl_": "powerlaw", "fft_": "fft", "stack_": "stack",
    "av_": "avocado", "bb_": "blackbody",
}


def families_for(cols):
    """Families that own at least one of the given feature columns."""
    return {fam for c in cols for p, fam in _FAMILY_PREFIXES.items() if c.startswith(p)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_H, _KB, _C = 6.62607e-27, 1.380649e-16, 2.99792458e10
_T_GRID = np.logspace(np.log10(3000.0), np.log10(60000.0), 120)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _planck_nu(nu, T):
    x = np.clip(_H * nu / (_KB * T), 1e-6, 700.0)
    return 2 * _H * nu ** 3 / _C ** 2 / np.expm1(x)


def _fit_bb(amps):
    """Black-body temperature from per-band amplitudes (uJy). Relative-error weights, grid over T,
    analytic scale. Returns {"T", "rchi2"} or None."""
    names = [b for b in BANDS if b in amps and amps[b] > 0]
    if len(names) < 3:
        return None
    a = np.array([amps[b] for b in names])
    nu = _C / (np.array([LAMBDA_NM[b] for b in names]) * 1e-7)
    w = _planck_nu(nu[None, :], _T_GRID[:, None]) / a[None, :]
    s = w.sum(1) / (w ** 2).sum(1)
    chi = ((1.0 - s[:, None] * w) ** 2).sum(1)
    j = int(np.argmin(chi))
    return {"T": _T_GRID[j], "rchi2": chi[j] / max(len(names) - 2, 1)}


def _prepare_bands(bands, a_lam):
    """{band: (t, f, e)}: median-subtracted, dust-corrected; bands with < 3 points dropped."""
    out = {}
    for name, (t, f, e) in bands.items():
        m = np.isfinite(f) & np.isfinite(t)
        if m.sum() < 3:
            continue
        t, f, e = t[m], f[m], e[m]
        good = np.isfinite(e) & (e > 0)
        e = np.where(good, e, np.median(e[good]) if good.any() else 1.0)
        f = f - np.median(f)
        a = a_lam.get(name)
        if a is not None and np.isfinite(a):
            k = 10 ** (0.4 * float(a))
            f, e = f * k, e * k
        out[name] = (t, f, e)
    return out


def _clip_exp(x):
    return np.exp(np.clip(x, -50, 50))


def _bazin(t, A, t0, trise, tfall, B):
    return A * _clip_exp(-(t - t0) / tfall) / (1.0 + _clip_exp(-(t - t0) / trise)) + B


def _tde_shape(t, t0, tr, tau, alpha):
    """Stand-in for the Potica shape: sigmoid rise x power-law decay (peak ~ 1)."""
    s = 1.0 / (1.0 + _clip_exp(-(t - t0) / tr))
    d = 1.0 + np.maximum(t - t0, 0.0) / tau
    return s * d ** (-alpha)


def _tde_model(t, A, B, t0, tr, tau, alpha):
    return B + A * _tde_shape(t, t0, tr, tau, alpha)


def _ricker(points, a):
    x = np.arange(points) - (points - 1) / 2.0
    amp = 2.0 / (np.sqrt(3.0 * a) * np.pi ** 0.25)
    return amp * (1.0 - x ** 2 / a ** 2) * np.exp(-x ** 2 / (2.0 * a ** 2))


def _red_chi2(y, model, err, n_par):
    return float(np.sum(((y - model) / err) ** 2) / max(len(y) - n_par, 1))


def _colour(fa, fb):
    if fa > 0 and fb > 0:
        return float(np.clip(-2.5 * np.log10(fa / fb), -10, 10))
    return np.nan


def _grid_curve(t, f):
    grid = np.linspace(t[0], t[-1], N_GRID)
    raw = np.interp(grid, t, f)
    w = 7
    sm = np.convolve(np.pad(raw, (w // 2, w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    return grid, raw, sm


# --------------------------------------------------------------------------- #
# Per-object context (computed lazily, shared between families)
# --------------------------------------------------------------------------- #
class _Obj:
    def __init__(self, bands):
        self.bands = bands
        self.primary = _primary_band(bands)
        self._cache = {}

    def get(self, name, fn):
        if name not in self._cache:
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    self._cache[name] = fn()
            except Exception:       # a failed fit must not kill the whole feature vector
                self._cache[name] = None
        return self._cache[name]

    @property
    def curve(self):
        if self.primary is None:
            return None
        return self.get("curve", lambda: _curve_ctx(self.bands, self.primary))

    @property
    def potica(self):
        if self.primary is None:
            return None
        return self.get("potica_fit", lambda: _potica_fit(self))

    @property
    def merged(self):
        return self.get("merged", lambda: _merged(self.bands))

    @property
    def windows(self):
        """Mean flux per band in the phase windows COLOUR_EDGES (needs the peak time)."""
        def build():
            tp = self.curve["tp"]
            res = {}
            for b, (t, f, e) in self.bands.items():
                ph = t - tp
                res[b] = [f[(ph >= lo) & (ph < hi)].mean() if ((ph >= lo) & (ph < hi)).any() else np.nan
                          for lo, hi in zip(COLOUR_EDGES[:-1], COLOUR_EDGES[1:])]
            return res
        return self.get("windows", build) if self.curve else None


def _primary_band(bands):
    """Band with the highest peak S/N (needs >= 5 epochs)."""
    best, best_snr = None, -np.inf
    for name, (t, f, e) in bands.items():
        if len(t) < 5:
            continue
        s = np.max(f / e)
        if s > best_snr:
            best, best_snr = name, s
    return best


def _merged(bands):
    t = np.concatenate([v[0] for v in bands.values()])
    f = np.concatenate([v[1] for v in bands.values()])
    e = np.concatenate([v[2] for v in bands.values()])
    b = np.concatenate([[n] * len(v[0]) for n, v in bands.items()])
    return t, f, e, b


def _curve_ctx(bands, primary):
    t, f, e = bands[primary]
    grid, raw, sm = _grid_curve(t, f)
    base = np.median(sm)
    ip = int(np.argmax(sm))
    amp = sm[ip] - base
    if not np.isfinite(amp) or amp <= 0:
        return None
    above = sm > base + 0.1 * amp
    lo = hi = ip
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < N_GRID - 1 and above[hi + 1]:
        hi += 1
    return dict(t=t, f=f, e=e, grid=grid, raw=raw, sm=sm, base=base, ip=ip, tp=grid[ip],
                amp=amp, lo=lo, hi=hi, span=grid[-1] - grid[0])


def _potica_fit(o):
    t, f, e = o.bands[o.primary]
    tt = t - t[0]
    span = max(tt[-1], 1.0)
    scale = np.max(np.abs(f))
    if scale <= 0 or len(t) < 7:
        return None
    fn, en = f / scale, e / scale
    ip = int(np.argmax(fn))
    p0 = [1.0, 0.0, tt[ip] - 0.05 * span, max(0.05 * span, 2.0), max(0.1 * span, 5.0), 5 / 3]
    bounds = ([0, -2, -0.3 * span, 0.5, 1.0, 0.2], [100, 2, 1.2 * span, span, 5 * span, 5.0])
    popt, _ = curve_fit(_tde_model, tt, fn, p0=p0, bounds=bounds, sigma=en,
                        absolute_sigma=True, maxfev=1000)
    A, B, t0, tr, tau, alpha = popt
    res = {
        "potica_amp": A * scale, "potica_t0_off": t0 - tt[ip], "potica_trise": tr,
        "potica_tau": tau, "potica_alpha": alpha,
        "potica_chi2": _red_chi2(fn, _tde_model(tt, *popt), en, 6),
    }
    amps = {}                                   # per-band amplitude of the fitted shape
    for name, (tb, fb, eb) in o.bands.items():
        if len(tb) < 3:
            continue
        s = _tde_shape(tb - t[0], t0, tr, tau, alpha)
        M = np.vstack([s, np.ones_like(s)]).T / eb[:, None]
        coef, *_ = np.linalg.lstsq(M, fb / eb, rcond=None)
        amps[name] = coef[0]
    for a, b in zip("ugriz", "grizy"):
        if a in amps and b in amps and amps[a] > 0 and amps[b] > 0:
            res[f"potica_col_{a}{b}"] = float(np.clip(-2.5 * np.log10(amps[a] / amps[b]), -10, 10))
    return {"res": res, "amps": amps}


# --------------------------------------------------------------------------- #
# Family functions: each returns a dict of (a subset of) its keys, or None
# --------------------------------------------------------------------------- #
def _fam_stat(o):
    t, f, e, b = o.merged
    snr = f / e
    imax = int(np.argmax(snr))
    span = t.max() - t.min()
    t_rel = t[imax] - t.min()
    bname = b[imax]
    return {
        "stat_maxsnr": snr[imax], "stat_t_maxsnr": t_rel,
        "stat_t_maxsnr_frac": t_rel / span if span > 0 else np.nan,
        "stat_flux_maxsnr": f[imax],
        "stat_band_maxsnr": BANDS.index(bname) if bname in BANDS else np.nan,
        "stat_mad_snr": median_abs_deviation(snr, scale="normal"),
        "stat_median_snr": np.median(snr), "stat_std_snr": np.std(snr),
        "stat_skew_snr": skew(snr), "stat_kurt_snr": kurtosis(snr),
        "stat_frac_pos": np.mean(snr > SNR_SIG), "stat_frac_neg": np.mean(snr < -SNR_SIG),
        "stat_n_obs": len(snr), "stat_span": span,
    }


def _fam_bazin(o):
    if o.primary is None:
        return None
    t, f, e = o.bands[o.primary]
    tt = t - t[0]
    span = max(tt[-1], 1.0)
    scale = np.max(np.abs(f))
    if scale <= 0 or len(t) < 6:
        return None
    fn, en = f / scale, e / scale
    ip = int(np.argmax(fn))
    p0 = [1.5, tt[ip], max(0.05 * span, 2.0), max(0.2 * span, 5.0), 0.0]
    bounds = ([0, -0.2 * span, 0.5, 0.5, -2], [100, 1.2 * span, span, 5 * span, 2])
    popt, _ = curve_fit(_bazin, tt, fn, p0=p0, bounds=bounds, sigma=en,
                        absolute_sigma=True, maxfev=1000)
    A, t0, tr, tf_, B = popt
    return {
        "bazin_amp": A * scale, "bazin_t0_off": t0 - tt[ip], "bazin_trise": tr,
        "bazin_tfall": tf_, "bazin_base": B * scale,
        "bazin_chi2": _red_chi2(fn, _bazin(tt, *popt), en, 5), "bazin_fall_rise": tf_ / tr,
    }


def _fam_potica(o):
    p = o.potica
    return p["res"] if p else None


def _fam_shape(o):
    cv = o.curve
    if not cv:
        return None
    grid, sm, base, ip, tp, amp = cv["grid"], cv["sm"], cv["base"], cv["ip"], cv["tp"], cv["amp"]

    def before(level):
        idx = np.where(sm[:ip + 1] <= level)[0]
        return grid[idx[-1]] if len(idx) else np.nan

    def after(level):
        idx = np.where(sm[ip:] <= level)[0]
        return grid[ip + idx[0]] if len(idx) else np.nan

    half = base + 0.5 * amp
    fw_pre, fw_post = tp - before(half), after(half) - tp
    span = cv["span"]
    return {
        "shape_rise_time": tp - before(base + 0.05 * amp),
        "shape_fwhm_pre": fw_pre, "shape_fwhm_post": fw_post, "shape_fwhm": fw_pre + fw_post,
        "shape_asym": (fw_post - fw_pre) / (fw_post + fw_pre) if (fw_post + fw_pre) > 0 else np.nan,
        "shape_peak_snr": amp / np.median(cv["e"]),
        "shape_end_frac": (sm[-1] - base) / amp, "shape_start_frac": (sm[0] - base) / amp,
        "shape_peak_time_frac": (tp - grid[0]) / span if span > 0 else np.nan,
        "shape_band": BANDS.index(o.primary) if o.primary in BANDS else np.nan,
    }


def _fam_mexhat(o):
    cv = o.curve
    if not cv:
        return None
    raw = cv["raw"]
    sd = np.std(raw)
    if sd <= 0:
        return None
    z_raw = (raw - raw.mean()) / sd
    powers = np.array([np.mean(np.convolve(z_raw, _ricker(min(int(10 * a), N_GRID), a), mode="same") ** 2)
                       for a in MEXHAT_WIDTHS])
    tot = powers.sum()
    if tot <= 0:
        return None
    res = {f"mexhat_p{a}": p / tot for a, p in zip(MEXHAT_WIDTHS, powers)}
    res["mexhat_logpow"] = np.log10(tot)
    res["mexhat_peak_scale"] = MEXHAT_WIDTHS[int(np.argmax(powers))]
    return res


def _fam_flare_baseline(o):
    cv = o.curve
    if not cv:
        return None
    t, f, e = cv["t"], cv["f"], cv["e"]
    grid, sm, base, tp, amp = cv["grid"], cv["sm"], cv["base"], cv["tp"], cv["amp"]
    lo, hi, span = cv["lo"], cv["hi"], cv["span"]
    t_lo, t_hi = grid[lo], grid[hi]
    in_flare = (t >= t_lo) & (t <= t_hi)
    tf_, ff, ef = t[in_flare], f[in_flare], e[in_flare]
    res = {
        "flare_n": in_flare.sum(),
        "flare_duration": t_hi - t_lo,
        "flare_dur_frac": (t_hi - t_lo) / span if span > 0 else np.nan,
        "flare_area": _trapz(sm[lo:hi + 1] - base, grid[lo:hi + 1]) / amp,
        "flare_rise_decay": (tp - t_lo) / (t_hi - tp) if t_hi > tp else np.nan,
    }
    if len(tf_) >= 2:
        res["flare_mean_snr"] = np.mean(ff / ef)
        dt = np.diff(tf_)
        ok = dt > 0
        if ok.any():
            res["flare_max_slope"] = np.max(np.abs(np.diff(ff)[ok] / dt[ok])) / amp
    if len(tf_) >= 4:
        res["flare_skew"], res["flare_kurt"] = skew(ff), kurtosis(ff)
    fb, eb = f[~in_flare], e[~in_flare]
    res["base_n"] = len(fb)
    if len(fb) >= 3:
        med = np.median(fb)
        res.update({
            "base_std": np.std(fb), "base_mad": median_abs_deviation(fb, scale="normal"),
            "base_max_minus_median": np.max(fb) - med, "base_amp_ratio": (np.max(fb) - med) / amp,
            "base_chi2": _red_chi2(fb, med, eb, 1), "base_skew": skew(fb),
        })
    return res


def _fam_binning(o):
    res = {}
    cv = o.curve
    if cv:
        t, f, amp, tp = cv["t"], cv["f"], cv["amp"], cv["tp"]
        ph = t - tp
        for k, (lo, hi) in enumerate(zip(PHASE_EDGES[:-1], PHASE_EDGES[1:])):
            m = (ph >= lo) & (ph < hi)
            if m.any():
                res[f"bin_phase_{k}"] = f[m].mean() / amp
        win = o.windows
        for pair, (a, b) in {"gr": ("g", "r"), "ri": ("r", "i")}.items():
            if a in win and b in win:
                cols = [_colour(win[a][k], win[b][k]) for k in range(3)]
                for k in range(3):
                    res[f"bin_col_{pair}_w{k}"] = cols[k]
                res[f"bin_col_{pair}_evol"] = cols[1] - cols[0]
    # seasonality: mean S/N per quarter of the year (all bands merged)
    t, f, e, _ = o.merged
    q = np.clip(np.floor((t % 365.25) / 365.25 * 4).astype(int), 0, 3)
    snr = f / e
    for k in range(4):
        if (q == k).any():
            res[f"bin_season_{k}"] = snr[q == k].mean()
    return res


def _fam_powerlaw(o):
    cv = o.curve
    if not cv:
        return None
    sm, grid, ip, tp, amp, base = cv["sm"], cv["grid"], cv["ip"], cv["tp"], cv["amp"], cv["base"]
    y_all = (sm - base) / amp
    below = np.where(y_all[ip + 1:] < 0.05)[0]
    stop = ip + 1 + (below[0] if len(below) else len(y_all) - ip - 1)
    y, x = y_all[ip + 1:stop], grid[ip + 1:stop] - tp
    if len(y) < 4:
        return None
    lx, ly = np.log10(x), np.log10(y)
    slope, icpt = np.polyfit(lx, ly, 1)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    rmse = np.sqrt(np.mean((ly - (icpt + slope * lx)) ** 2))
    resid53 = np.std(ly + (5.0 / 3.0) * lx)
    exp_r2 = np.corrcoef(x, ly)[0, 1] ** 2
    return {"pl_alpha": -slope, "pl_r2": r2, "pl_rmse": rmse, "pl_resid_53": resid53,
            "pl_exp_r2": exp_r2, "pl_r2_diff": r2 - exp_r2, "pl_n": len(y)}


def _fam_fft(o):
    cv = o.curve
    if not cv:
        return None
    raw = cv["raw"]
    P = np.abs(np.fft.rfft(raw - raw.mean())) ** 2 / len(raw)
    P = P[1:]                                    # drop DC
    tot = P.sum()
    if tot <= 0:
        return None
    freqs = np.arange(1, len(P) + 1) / cv["span"]                   # cycles / day
    p = P / tot
    return {
        "fft_logpow": np.log10(P.mean()),
        "fft_pow_snr": P.mean() / np.median(cv["e"]) ** 2,
        "fft_dom_freq": freqs[int(np.argmax(P))],
        "fft_dom_frac": p.max(),
        "fft_low_frac": p[:3].sum(),
        "fft_centroid": (freqs * p).sum(),
        "fft_entropy": -(p * np.log(p + 1e-12)).sum() / np.log(len(p)),
    }


def _fam_stack(o):
    sel = [b for b in ("g", "r", "i") if b in o.bands]
    if not sel:
        return None
    t = np.concatenate([o.bands[b][0] for b in sel])
    f = np.concatenate([o.bands[b][1] for b in sel])
    e = np.concatenate([o.bands[b][2] for b in sel])
    if len(t) < 5:
        return None
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]
    snr = f / e
    fmax, smax = f.max(), snr.max()
    ch = {"flux": f, "snr": snr,
          "fluxn": f / fmax if fmax > 0 else None, "snrn": snr / smax if smax > 0 else None}
    res = {}
    for name, x in ch.items():
        if x is None:
            continue
        res[f"stack_mean_{name}"], res[f"stack_std_{name}"] = x.mean(), x.std()
        res[f"stack_max_{name}"], res[f"stack_min_{name}"] = x.max(), x.min()
        res[f"stack_mad_{name}"] = median_abs_deviation(x, scale="normal")
    ip = int(np.argmax(f))
    rest = t - t[0]
    xn = ch["fluxn"] if ch["fluxn"] is not None else f
    span = rest[-1]
    res.update({"stack_skew": skew(f), "stack_kurt": kurtosis(f), "stack_n": len(f),
                "stack_tpeak_frac": rest[ip] / span if span > 0 else np.nan})
    if ip >= 2 and rest[ip] > rest[0]:
        res["stack_slope_pre"] = np.polyfit(rest[:ip + 1], xn[:ip + 1], 1)[0]
    if len(f) - ip >= 3 and rest[-1] > rest[ip]:
        res["stack_slope_post"] = np.polyfit(rest[ip:], xn[ip:], 1)[0]
    return res


def _fam_avocado(o):
    res = {}
    cv = o.curve
    if cv:
        sm, grid, ip, tp, amp, base = cv["sm"], cv["grid"], cv["ip"], cv["tp"], cv["amp"], cv["base"]
        for frac, tag in ((0.8, "80"), (0.2, "20")):
            level = base + frac * amp
            idx = np.where(sm[ip:] <= level)[0]
            if len(idx):
                res[f"av_tfwd_{tag}"] = grid[ip + idx[0]] - tp
            idx = np.where(sm[:ip + 1] <= level)[0]
            if len(idx):
                res[f"av_tbwd_{tag}"] = tp - grid[idx[-1]]
        step = grid[1] - grid[0]
        x = sm - base
        res["av_pos_width"] = np.sum(np.clip(x, 0, None)) * step / amp
        res["av_neg_width"] = (np.sum(np.clip(x, None, 0)) / x.min() * step) if x.min() < 0 else 0.0
        res["av_abs_diff"] = np.sum(np.abs(np.diff(sm))) / amp
    t, f, e, b = o.merged
    snr = f / e
    res["av_frac_bkg"] = np.mean(np.abs(snr) < SNR_SIG)
    res["av_frac_gt5"], res["av_frac_gt10"] = np.mean(snr > 5), np.mean(snr > 10)
    res["av_frac_ltm5"] = np.mean(snr < -5)
    for thr in (5, 10):
        ts = t[np.abs(snr) > thr]
        if len(ts) >= 2:
            res[f"av_tw{thr}"] = ts.max() - ts.min()
    for name, (tb, fb, eb) in o.bands.items():
        if name in BANDS:
            res[f"av_total_snr_{name}"] = np.sqrt(np.sum((fb / eb) ** 2))
    return res


def _fam_blackbody(o):
    p = o.potica
    if not p:
        return None
    fit = _fit_bb(p["amps"])
    if not fit:
        return None
    res = {"bb_logT": np.log10(fit["T"]), "bb_rchi2": fit["rchi2"]}
    win = o.windows
    if win:
        late = {b: v[1] for b, v in win.items()}          # window 20-60 days after peak
        fl = _fit_bb(late)
        if fl:
            res["bb_logT_late"] = np.log10(fl["T"])
            res["bb_dlogT"] = res["bb_logT_late"] - res["bb_logT"]
    return res


_FAMILY_FUNCS = {
    "bazin": _fam_bazin, "potica": _fam_potica, "shape": _fam_shape, "stat": _fam_stat,
    "mexhat": _fam_mexhat, "flare_baseline": _fam_flare_baseline, "binning": _fam_binning,
    "powerlaw": _fam_powerlaw, "fft": _fam_fft, "stack": _fam_stack, "avocado": _fam_avocado,
    "blackbody": _fam_blackbody,
}


def extract_object_features(bands, a_lam, families=None):
    """Features of the requested families (default: all) as a dict. Missing -> absent/NaN."""
    families = FAMILY_ORDER if families is None else [f for f in FAMILY_ORDER if f in families]
    out = {k: np.nan for fam in families for k in FAMILY_KEYS[fam]}
    bands = _prepare_bands(bands, a_lam)
    if not bands:
        return out
    o = _Obj(bands)
    for fam in families:
        r = o.get("fam_" + fam, lambda fam=fam: _FAMILY_FUNCS[fam](o))
        if r:
            out.update({k: v for k, v in r.items() if k in out})
    return out
