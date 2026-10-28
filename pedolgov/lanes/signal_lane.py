from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, gammaln, logsumexp
from scipy.stats import gamma

from pedolgov.lanes.contracts import Contingency
from pedolgov.manifest.config import SignalConfig

_EPS = 0.5


def _cells(c: Contingency) -> tuple[float, float, float, float]:
    return c.a + _EPS, c.b + _EPS, c.c + _EPS, c.d + _EPS


def proportional_reporting_ratio(c: Contingency) -> float:
    a, b, cc, d = _cells(c)
    return (a / (a + b)) / (cc / (cc + d))


def reporting_odds_ratio(c: Contingency) -> float:
    a, b, cc, d = _cells(c)
    return (a * d) / (b * cc)


def reporting_odds_ratio_ci_lower(c: Contingency) -> float:
    a, b, cc, d = _cells(c)
    ln_ror = math.log((a * d) / (b * cc))
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / cc + 1.0 / d)
    return math.exp(ln_ror - 1.96 * se)


def chi_square_yates(c: Contingency) -> float:
    a, b, cc, d = float(c.a), float(c.b), float(c.c), float(c.d)
    n = a + b + cc + d
    if n == 0:
        return 0.0
    row1, row2 = a + b, cc + d
    col1, col2 = a + cc, b + d
    denom = row1 * row2 * col1 * col2
    if denom == 0:
        return 0.0
    num = n * max(0.0, abs(a * d - b * cc) - n / 2.0) ** 2
    return num / denom


def bcpnn_information_component(c: Contingency) -> float:
    a, b, cc, d = _cells(c)
    n = a + b + cc + d
    expected = (a + b) * (a + cc) / n
    return math.log2(a / expected)


def bcpnn_lower_bound(c: Contingency) -> float:
    a, b, cc, d = c.a, c.b, c.c, c.d
    n = a + b + cc + d
    gamma_total = (n + 2.0) ** 2 / ((a + b + 1.0) * (a + cc + 1.0))
    mean = math.log2(
        (a + 1.0) * (n + 2.0) ** 2 / ((n + gamma_total) * (a + b + 1.0) * (a + cc + 1.0))
    )
    variance = (
        (n - a + gamma_total - 1.0) / ((a + 1.0) * (n + 1.0 + gamma_total))
        + (n - a - b + 1.0) / ((a + b + 1.0) * (n + 3.0))
        + (n - a - cc + 1.0) / ((a + cc + 1.0) * (n + 3.0))
    ) / math.log(2.0) ** 2
    return mean - 1.96 * math.sqrt(max(variance, 0.0))


def fit_gamma_mixture(
    observed: np.ndarray, expected: np.ndarray
) -> tuple[float, float, float, float, float]:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.ndim != 1 or observed.shape != expected.shape or len(observed) < 5:
        raise ValueError("fitting requires at least five observed and expected counts")
    if (
        np.any(observed < 0)
        or np.any(expected <= 0)
        or not np.isfinite(observed).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("counts and expected values must be finite and valid")

    def objective(parameters: np.ndarray) -> float:
        a1, b1, a2, b2 = np.exp(parameters[:4])
        p = 1.0 / (1.0 + np.exp(-parameters[4]))

        def component(a: float, b: float) -> np.ndarray:
            return (
                gammaln(observed + a)
                - gammaln(a)
                - gammaln(observed + 1)
                + a * np.log(b / (b + expected))
                + observed * np.log(expected / (b + expected))
            )

        return -float(
            np.sum(
                logsumexp(
                    np.stack((np.log(p) + component(a1, b1), np.log1p(-p) + component(a2, b2))),
                    axis=0,
                )
            )
        )

    result = minimize(
        objective,
        np.log([1.0, 1.0, 4.0, 4.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-8, 8)] * 4 + [(-8, 8)],
    )
    if not result.success:
        raise RuntimeError(f"empirical Bayes prior fit failed: {result.message}")
    a1, b1, a2, b2 = np.exp(result.x[:4])
    p = 1.0 / (1.0 + np.exp(-result.x[4]))
    return float(a1), float(b1), float(a2), float(b2), float(p)


def gamma_mixture_estimates(
    c: Contingency, prior: tuple[float, float, float, float, float]
) -> tuple[float, float]:
    a1, b1, a2, b2, p = prior
    if min(a1, b1, a2, b2) <= 0 or not 0 < p < 1 or c.total <= 0:
        raise ValueError("invalid empirical Bayes prior or empty table")
    expected = (c.a + c.b) * (c.a + c.c) / c.total
    if expected <= 0:
        raise ValueError("expected count must be positive")

    def log_marginal(a: float, b: float) -> float:
        return float(
            gammaln(c.a + a)
            - gammaln(a)
            + a * math.log(b / (b + expected))
            + c.a * math.log(expected / (b + expected))
        )

    logs = np.array([math.log(p) + log_marginal(a1, b1), math.log1p(-p) + log_marginal(a2, b2)])
    weight = float(np.exp(logs[0] - logsumexp(logs)))
    shape1, shape2 = a1 + c.a, a2 + c.a
    rate1, rate2 = b1 + expected, b2 + expected
    ebgm = math.exp(
        weight * (digamma(shape1) - math.log(rate1))
        + (1 - weight) * (digamma(shape2) - math.log(rate2))
    )

    def cdf(x: float) -> float:
        return weight * gamma.cdf(x, shape1, scale=1 / rate1) + (1 - weight) * gamma.cdf(
            x, shape2, scale=1 / rate2
        )

    hi = max(gamma.ppf(0.95, shape1, scale=1 / rate1), gamma.ppf(0.95, shape2, scale=1 / rate2))
    return ebgm, float(brentq(lambda x: cdf(x) - 0.05, 0.0, hi))


@dataclass(frozen=True)
class SignalVerdict:
    prr: float
    ror_ci_lower: float
    ic: float
    ebgm: float
    ic025: float
    eb05: float
    chi2: float
    positive_methods: int
    consensus_fraction: float
    is_signal: bool


def consensus_signal(c: Contingency, cfg: SignalConfig) -> SignalVerdict:
    prr = proportional_reporting_ratio(c)
    chi2 = chi_square_yates(c)
    ror_lo = reporting_odds_ratio_ci_lower(c)
    ic = bcpnn_information_component(c)
    if cfg.eb_prior is None:
        raise ValueError("signal.eb_prior must be fitted from the reference database")
    ic025 = bcpnn_lower_bound(c)
    ebgm, eb05 = gamma_mixture_estimates(c, cfg.eb_prior)
    flags = [
        prr >= cfg.prr_min and chi2 >= cfg.chi2_min,
        ror_lo > cfg.ror_ci_min,
        ic025 > 0.0,
        eb05 >= 2.0,
    ]
    positive = sum(1 for f in flags if f)
    return SignalVerdict(
        prr=prr,
        ror_ci_lower=ror_lo,
        ic=ic,
        ebgm=ebgm,
        ic025=ic025,
        eb05=eb05,
        chi2=chi2,
        positive_methods=positive,
        consensus_fraction=positive / len(flags),
        is_signal=positive >= cfg.consensus_min,
    )
