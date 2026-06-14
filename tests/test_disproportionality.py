from __future__ import annotations

import math

from pedolgov.lanes.contracts import Contingency
from pedolgov.lanes.signal_lane import (
    bcpnn_information_component,
    chi_square_yates,
    consensus_signal,
    proportional_reporting_ratio,
    reporting_odds_ratio,
)
from pedolgov.manifest.config import SignalConfig


def test_prr_matches_definition() -> None:
    c = Contingency(a=30, b=70, c=10, d=9890)
    a, b, cc, d = 30.5, 70.5, 10.5, 9890.5
    expected = (a / (a + b)) / (cc / (cc + d))
    assert math.isclose(proportional_reporting_ratio(c), expected, rel_tol=1e-9)


def test_ror_matches_definition() -> None:
    c = Contingency(a=40, b=60, c=20, d=9880)
    a, b, cc, d = 40.5, 60.5, 20.5, 9880.5
    expected = (a * d) / (b * cc)
    assert math.isclose(reporting_odds_ratio(c), expected, rel_tol=1e-9)


def test_strong_signal_passes_consensus() -> None:
    c = Contingency(a=120, b=80, c=20, d=9780)
    verdict = consensus_signal(c, SignalConfig())
    assert verdict.is_signal
    assert verdict.positive_methods >= 3
    assert proportional_reporting_ratio(c) >= 2.0


def test_null_signal_fails_consensus() -> None:
    c = Contingency(a=2, b=198, c=98, d=9702)
    verdict = consensus_signal(c, SignalConfig())
    assert not verdict.is_signal
    assert verdict.consensus_fraction < 0.75


def test_chi_square_and_ic_signs() -> None:
    strong = Contingency(a=120, b=80, c=20, d=9780)
    null = Contingency(a=2, b=198, c=98, d=9702)
    assert chi_square_yates(strong) > chi_square_yates(null)
    assert bcpnn_information_component(strong) > bcpnn_information_component(null)
