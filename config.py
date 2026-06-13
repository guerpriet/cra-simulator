"""Dataclasses for the full simulator configuration."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class Run:
    n: int = 5000
    seed: int = 1
    threshold: float = 0.3   # credit decision threshold (reject if PD > threshold)


@dataclass
class FeatureSet:
    selected: list[str] = field(default_factory=list)
    n_noise: int = 0


@dataclass
class W:
    feature: str
    weight: float = 0.0


@dataclass
class PD:
    weights: list[W] = field(default_factory=list)
    intercept: float = -3.0   # baseline log-odds; σ(-3) ≈ 4.7 % base default rate


@dataclass
class Path:
    feature: str
    strength: float = 0.0


@dataclass
class SensitiveCfg:
    """Settings for one sensitive attribute: bias term, CF paths, reference group."""
    name: str
    bias: float = 0.0
    paths: list[Path] = field(default_factory=list)
    reference_cat: str = ""


@dataclass
class Fair:
    attrs: list[SensitiveCfg] = field(default_factory=list)
    counterfactual: bool = False


@dataclass
class Reject:
    accept_rate: float = 0.5
    mechanism: Literal["MAR", "MNAR"] = "MAR"
    selection_features: list[str] = field(default_factory=list)


@dataclass
class LGD:
    base: float = 0.45
    workout_months: int = 24
    weights: list[W] = field(default_factory=list)


@dataclass
class EAD:
    base_util: float = 0.4
    ccf: float = 0.5
    weights: list[W] = field(default_factory=list)


@dataclass
class Shift:
    n_target: int = 5000
    covariate_shifts: dict[str, float] = field(default_factory=dict)
    lgd_drift: float = 0.0
    drop_in_target: list[str] = field(default_factory=list)


@dataclass
class CausalLink:
    driver: str = ""
    targets: list[str] = field(default_factory=list)
    strength: float = 0.0


@dataclass
class Cfg:
    run: Run = field(default_factory=Run)
    features: FeatureSet = field(default_factory=FeatureSet)
    pd: PD = field(default_factory=PD)
    fair: Fair = field(default_factory=Fair)
    reject: Reject = field(default_factory=Reject)
    lgd: LGD = field(default_factory=LGD)
    ead: EAD = field(default_factory=EAD)
    shift: Shift = field(default_factory=Shift)
    causal: list[CausalLink] = field(default_factory=list)

    def to_dict(self) -> dict: return asdict(self)


# ---------------------------------------------------------------------------
# Suggested default causal links — economically motivated relationships.
# These are loaded as session defaults (state.py) and shown by the
# "Suggest defaults" button on the Features page.
# ---------------------------------------------------------------------------
DEFAULT_CAUSAL_LINKS: list[CausalLink] = [
    # More missed payments → lower bureau score (strong negative)
    CausalLink(
        driver="missed_payments_24m",
        targets=["bureau_score"],
        strength=-1.0,
    ),
    # Better bureau score → lower interest rate (risk-based pricing)
    CausalLink(
        driver="bureau_score",
        targets=["interest_rate_pct"],
        strength=-1.0,
    ),
    # Higher income → larger loan requests and more savings
    CausalLink(
        driver="net_income",
        targets=["loan_amount", "savings"],
        strength=1.2,
    ),
    # Age → accumulated credit history and savings (mild positive)
    CausalLink(
        driver="age",
        targets=["debt_history", "current_debt", "savings"],
        strength=0.4,
    ),
]
