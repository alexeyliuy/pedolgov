from __future__ import annotations

import math
from dataclasses import dataclass

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
    num = n * (abs(a * d - b * cc) - n / 2.0) ** 2
    return num / denom


def bcpnn_information_component(c: Contingency) -> float:
    a, b, cc, d = _cells(c)
    n = a + b + cc + d
    expected = (a + b) * (a + cc) / n
    return math.log2(a / expected)


def empirical_bayes_geometric_mean(c: Contingency) -> float:
    a, b, cc, d = _cells(c)
    n = a + b + cc + d
    expected = (a + b) * (a + cc) / n
    alpha = 2.0
    return (a + alpha) / (expected + alpha)


@dataclass(frozen=True)
class SignalVerdict:
    prr: float
    ror_ci_lower: float
    ic: float
    ebgm: float
    chi2: float
    positive_methods: int
    consensus_fraction: float
    is_signal: bool


def consensus_signal(c: Contingency, cfg: SignalConfig) -> SignalVerdict:
    prr = proportional_reporting_ratio(c)
    chi2 = chi_square_yates(c)
    ror_lo = reporting_odds_ratio_ci_lower(c)
    ic = bcpnn_information_component(c)
    ebgm = empirical_bayes_geometric_mean(c)
    flags = [
        prr >= cfg.prr_min and chi2 >= cfg.chi2_min,
        ror_lo > cfg.ror_ci_min,
        ic > 0.0,
        ebgm >= 2.0,
    ]
    positive = sum(1 for f in flags if f)
    return SignalVerdict(
        prr=prr,
        ror_ci_lower=ror_lo,
        ic=ic,
        ebgm=ebgm,
        chi2=chi2,
        positive_methods=positive,
        consensus_fraction=positive / len(flags),
        is_signal=positive >= cfg.consensus_min,
    )
