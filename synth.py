"""Feature-wise synthesis, causal links, noise columns, target models."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit, logit as _logit
from catalog import CATALOG, Spec
from config import Cfg, W, CausalLink, Path

DIST_OPTIONS = [
    "normal", "uniform", "lognormal", "exponential",
    "beta", "right_skewed", "left_skewed",
]

DIST_LABELS = {
    "normal":       "Normal  N(mean, sd)",
    "uniform":      "Uniform  U(min, max)",
    "lognormal":    "Log-normal  LogN(μ, σ)  — positive, right-skewed",
    "exponential":  "Exponential  Exp(scale) + offset",
    "beta":         "Beta  Beta(α, β) scaled to [lo, hi]",
    "right_skewed": "Right-skewed  Gamma(shape, scale) + offset",
    "left_skewed":  "Left-skewed  –Gamma(shape, scale) + offset",
}

DIST_DEFAULT_PARAMS: dict[str, dict] = {
    "normal":       {"dist": "normal",      "mean": 0.0,  "sd": 1.0},
    "uniform":      {"dist": "uniform",     "min": 0.0,   "max": 1.0},
    "lognormal":    {"dist": "lognormal",   "mu": 0.0,    "sigma": 0.5},
    "exponential":  {"dist": "exponential", "scale": 1.0, "offset": 0.0},
    "beta":         {"dist": "beta",        "alpha": 2.0, "beta_": 2.0, "lo": 0.0, "hi": 1.0},
    "right_skewed": {"dist": "right_skewed","shape": 2.0, "scale": 1.0, "offset": 0.0},
    "left_skewed":  {"dist": "left_skewed", "shape": 2.0, "scale": 1.0, "offset": 0.0},
}

# ─── FIX 1: Fields that must be integers in the output ───────────────────────
# These are generated as floats by the distribution samplers and then rounded.
INT_FIELDS: frozenset[str] = frozenset({"age", "dependants", "term_months", "missed_payments_24m"})

# ─── FIX 2: Income scaling factors by employment status ───────────────────────
# Applied before causal links so that downstream loan_amount / savings
# correlations reflect realistic income levels per employment type.
_EMPLOYMENT_INCOME_SCALE: dict[str, float] = {
    "unemployed": 0.35,
    "student":    0.28,
    "retired":    0.65,
    # employed, self_employed, civil_servant: 1.0 (unchanged)
}
# Employment statuses where tenure makes no sense → set to 0
_EMPLOYMENT_TENURE_ZERO: frozenset[str] = frozenset({"unemployed", "student"})


# ──────────────────────────────────────────────────────────────────────────────
# Standardise helper (no stored _z columns)
# ──────────────────────────────────────────────────────────────────────────────
def _z(df: pd.DataFrame, feature: str) -> np.ndarray | None:
    if feature not in df.columns:
        return None
    col = df[feature]
    if col.dtype.kind not in "iuf":
        return None
    x = col.astype(float).to_numpy()
    mean = np.nanmean(x)
    sd   = np.nanstd(x) or 1.0
    z = (x - mean) / sd
    # NaN entries (e.g. ltv=NaN for unsecured loans) contribute 0 to any model score
    return np.where(np.isnan(z), 0.0, z)


# ──────────────────────────────────────────────────────────────────────────────
# Per-distribution generators
# ──────────────────────────────────────────────────────────────────────────────
def _gen_num(s: Spec, n: int, rng: np.random.Generator) -> np.ndarray:
    p = s.params
    dist = p.get("dist", "normal")

    if dist == "normal":
        x = rng.normal(p.get("mean", 0.0), max(p.get("sd", 1.0), 1e-9), n)
    elif dist == "uniform":
        lo, hi = p.get("min", 0.0), p.get("max", 1.0)
        if lo >= hi: hi = lo + 1.0
        x = rng.uniform(lo, hi, n)
    elif dist == "lognormal":
        x = rng.lognormal(p.get("mu", 0.0), max(p.get("sigma", 0.5), 1e-9), n)
    elif dist == "exponential":
        x = rng.exponential(max(p.get("scale", 1.0), 1e-9), n) + p.get("offset", 0.0)
    elif dist == "beta":
        a  = max(p.get("alpha", 2.0), 1e-3)
        b  = max(p.get("beta_", 2.0), 1e-3)
        lo, hi = p.get("lo", 0.0), p.get("hi", 1.0)
        if lo >= hi: hi = lo + 1.0
        x = lo + (hi - lo) * rng.beta(a, b, n)
    elif dist == "right_skewed":
        x = rng.gamma(max(p.get("shape", 2.0), 1e-3),
                      max(p.get("scale", 1.0), 1e-9), n) + p.get("offset", 0.0)
    elif dist == "left_skewed":
        x = -rng.gamma(max(p.get("shape", 2.0), 1e-3),
                       max(p.get("scale", 1.0), 1e-9), n) + p.get("offset", 0.0)
    else:
        x = rng.normal(p.get("mean", 0.0), max(p.get("sd", 1.0), 1e-9), n)

    # Apply min/max clipping (skipped for uniform/beta which are bounded by construction)
    if dist not in ("uniform", "beta"):
        if "min" in p: x = np.maximum(x, p["min"])
        if "max" in p: x = np.minimum(x, p["max"])
    return x


def _gen_cat(s: Spec, n: int, rng: np.random.Generator) -> np.ndarray:
    cats  = s.params["categories"]
    probs = np.asarray(s.params.get("probs", [1 / len(cats)] * len(cats)), float)
    probs /= probs.sum()
    return rng.choice(cats, n, p=probs)


def _gen_bin(s: Spec, n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < s.params.get("p", 0.5)).astype(int)


def tokenise(specs: list[Spec], n: int, rng: np.random.Generator) -> pd.DataFrame:
    cols = {}
    for s in specs:
        if   s.dtype == "num": cols[s.name] = _gen_num(s, n, rng)
        elif s.dtype == "cat": cols[s.name] = _gen_cat(s, n, rng)
        elif s.dtype == "bin": cols[s.name] = _gen_bin(s, n, rng)
    return pd.DataFrame(cols)


# ──────────────────────────────────────────────────────────────────────────────
# FIX 6 — re-clamp features to catalog bounds after causal adjustments
# ──────────────────────────────────────────────────────────────────────────────
def clamp_to_bounds(df: pd.DataFrame, specs: list[Spec]) -> pd.DataFrame:
    """Re-apply catalog min/max bounds after causal link shifts."""
    df = df.copy()
    for s in specs:
        if s.dtype != "num" or s.name not in df.columns:
            continue
        p = s.params
        if "min" in p:
            df[s.name] = np.maximum(df[s.name].to_numpy(float), p["min"])
        if "max" in p:
            df[s.name] = np.minimum(df[s.name].to_numpy(float), p["max"])
    # Also clamp lognormal/exponential/right_skewed that have implicit lower bound at 0
    for s in specs:
        if s.dtype != "num" or s.name not in df.columns:
            continue
        if s.params.get("dist") in ("lognormal", "exponential", "right_skewed"):
            offset = s.params.get("offset", 0.0)
            lo = max(s.params.get("min", offset), offset)
            df[s.name] = np.maximum(df[s.name].to_numpy(float), lo)
    return df


def apply_causal(df: pd.DataFrame, links: list[CausalLink]) -> pd.DataFrame:
    """Shift target features by strength × σ(target) × z(driver). Raw columns only."""
    df = df.copy()
    for L in links:
        if not L.driver or L.driver not in df.columns or L.strength == 0:
            continue
        z_driver = _z(df, L.driver)
        if z_driver is None:
            continue
        for t in L.targets:
            if t not in df.columns or df[t].dtype.kind not in "iuf":
                continue
            sd = df[t].to_numpy(float).std() + 1e-9
            df[t] = df[t] + L.strength * sd * z_driver
    return df


def add_noise(df: pd.DataFrame, n_noise: int, rng: np.random.Generator) -> pd.DataFrame:
    if n_noise <= 0:
        return df
    df = df.copy()
    for i in range(1, n_noise + 1):
        df[f"noise_{i:03d}"] = rng.standard_normal(len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FIX A — Adjust income / tenure by employment status (pre-causal)
# Applied before causal links so downstream correlations (income → loan_amount)
# already reflect realistic income levels per employment group.
# ──────────────────────────────────────────────────────────────────────────────
def _adjust_by_employment(df: pd.DataFrame) -> pd.DataFrame:
    """Scale net_income by employment status; zero out tenure for non-workers."""
    if "employment_status" not in df.columns:
        return df
    df = df.copy()
    if "net_income" in df.columns:
        for status, factor in _EMPLOYMENT_INCOME_SCALE.items():
            mask = df["employment_status"] == status
            if mask.any():
                df.loc[mask, "net_income"] = df.loc[mask, "net_income"] * factor
    if "tenure_months" in df.columns:
        for status in _EMPLOYMENT_TENURE_ZERO:
            mask = df["employment_status"] == status
            if mask.any():
                df.loc[mask, "tenure_months"] = 0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FIX B — Derive dti from actual debt/income; round integer fields;
#          condition LTV on collateral flag.
# Called after causal links + clamping so derived values are consistent.
# ──────────────────────────────────────────────────────────────────────────────
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process: round integer fields, derive dti, condition ltv on collateral."""
    df = df.copy()

    # 1. Round fields that must be whole numbers in reality
    for col in INT_FIELDS:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    # 2. DTI = total current debt / net income, clipped to [0, 2]
    #    This replaces the independently sampled dti from the old catalog.
    if "current_debt" in df.columns and "net_income" in df.columns:
        income = df["net_income"].to_numpy(float)
        debt   = df["current_debt"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dti = np.where(income > 0, debt / income, 0.0)
        df["dti"] = np.clip(dti, 0.0, 2.0)

    # 3. LTV is only meaningful for collateralized loans (FIX C)
    if "ltv" in df.columns and "collateral" in df.columns:
        df.loc[df["collateral"] == 0, "ltv"] = np.nan

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Base feature pipeline
# ──────────────────────────────────────────────────────────────────────────────
def make_features(cfg: Cfg, n=None, seed_offset=0) -> tuple[pd.DataFrame, np.random.Generator]:
    rng   = np.random.default_rng(cfg.run.seed + seed_offset)
    specs = [s for s in CATALOG if s.name in set(cfg.features.selected)]
    df    = tokenise(specs, n or cfg.run.n, rng)
    df    = _adjust_by_employment(df)          # FIX A: employment → income/tenure
    df    = apply_causal(df, cfg.causal)
    df    = clamp_to_bounds(df, specs)
    df    = derive_features(df)                # FIX B+C: dti, integers, ltv
    df    = add_noise(df, cfg.features.n_noise, rng)
    return df, rng


# ──────────────────────────────────────────────────────────────────────────────
# PD model
# ──────────────────────────────────────────────────────────────────────────────
def pd_score(
    df: pd.DataFrame,
    weights: list[W],
    intercept: float,
    rng: np.random.Generator | None = None,
    noise: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute logit scores and feature contributions.

    FIX 3 — noise parameter: if provided, use it directly instead of drawing
    from rng. This lets gen_pd reuse the same noise realisation for both the
    real and the counterfactual dataset, so that |ΔPD| measures only the
    causal effect of flipping the sensitive attribute and not random noise.
    """
    contribs = pd.DataFrame(index=df.index)
    contribs["intercept"] = intercept
    for w in weights:
        if w.weight == 0:
            continue
        zv = _z(df, w.feature)
        if zv is not None:
            contribs[w.feature] = w.weight * zv
        elif w.feature in df.columns:
            ref = df[w.feature].mode().iloc[0]
            contribs[w.feature] = w.weight * (df[w.feature] != ref).astype(float).to_numpy()
    if noise is not None:
        contribs["noise"] = noise
    elif rng is not None:
        contribs["noise"] = rng.normal(0, 0.5, len(df))
    else:
        contribs["noise"] = 0.0
    return contribs.sum(axis=1), contribs


def _disadvantaged_indicator(sens: pd.Series, reference_cat: str = "") -> pd.Series:
    """Return 1 for the disadvantaged group, 0 for the reference (advantaged) group."""
    if sens.dtype.kind in "iuf":
        unique_vals = set(pd.unique(sens.dropna()))
        if unique_vals <= {0, 1}:
            return sens.astype(float)
        else:
            median = float(sens.median())
            return (sens.astype(float) < median).astype(float)
    else:
        if reference_cat and reference_cat in sens.values:
            ref = reference_cat
        else:
            ref = sens.mode().iloc[0]
        return (sens != ref).astype(float)


def add_direct_bias(
    logits: pd.Series,
    sens: pd.Series,
    bias: float,
    reference_cat: str = "",
) -> pd.Series:
    """Add bias × disadvantaged_indicator to the logit."""
    if bias == 0:
        return logits
    ind = _disadvantaged_indicator(sens, reference_cat)
    return logits + bias * ind


def to_pd(logits) -> pd.Series:
    return pd.Series(1 / (1 + np.exp(-np.asarray(logits, float))),
                     index=getattr(logits, "index", None))


# ──────────────────────────────────────────────────────────────────────────────
# Counterfactual
# ──────────────────────────────────────────────────────────────────────────────
def _flip(s: pd.Series) -> pd.Series:
    """Flip a feature for the case where no reference category is set."""
    if s.dtype.kind in "iuf":
        if set(pd.unique(s)) <= {0, 1}:
            return 1 - s
        median = s.median()
        return pd.Series(2 * median - s.to_numpy(), index=s.index, dtype=s.dtype)
    cats = list(pd.unique(s))
    mp   = {c: cats[(i + 1) % len(cats)] for i, c in enumerate(cats)}
    return s.map(mp)


def counterfactual(
    df: pd.DataFrame,
    sens: str,
    paths: list[Path],
    reference_cat: str = "",
) -> pd.DataFrame:
    """Build counterfactual dataset by flipping the sensitive attribute."""
    cf = df.copy()

    if reference_cat and reference_cat in df[sens].values:
        cf[sens] = reference_cat
        delta = (df[sens] != reference_cat).astype(float).to_numpy()
    else:
        cf[sens] = _flip(df[sens])
        raw = df[sens]
        if raw.dtype.kind in "iuf":
            if set(pd.unique(raw)) <= {0, 1}:
                delta = (cf[sens].astype(float) - raw.astype(float)).to_numpy()
            else:
                sd = raw.astype(float).std() + 1e-9
                delta = ((cf[sens].astype(float) - raw.astype(float)) / sd).to_numpy()
        else:
            cats  = list(pd.unique(raw))
            enc   = {c: i for i, c in enumerate(cats)}
            delta = (cf[sens].map(enc) - raw.map(enc)).astype(float).to_numpy()

    for p in paths:
        if p.feature not in cf.columns or p.strength == 0:
            continue
        if df[p.feature].dtype.kind not in "iuf":
            continue
        sd = df[p.feature].to_numpy(float).std() + 1e-9
        cf[p.feature] = cf[p.feature].to_numpy(float) + p.strength * sd * delta

    return cf


# ──────────────────────────────────────────────────────────────────────────────
# LGD — FIX 4: workout_months now properly scales the recovery discount
# ──────────────────────────────────────────────────────────────────────────────
def lgd(
    df: pd.DataFrame,
    base: float,
    weights: list[W],
    workout_months: int,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Recovery model:
      logit(μ_LGD) = logit(base_LGD) + Σ wⱼ·zⱼ
      μ_adj = σ(logit(μ_LGD)) × (1 − 0.5 × recovery_rate)
      LGD_i ~ Beta(20·μ_adj, 20·(1 − μ_adj))

    recovery_rate = 1 − exp(−2 · workout_months / 24)
    """
    s = np.full(len(df), _logit(np.clip(base, 1e-4, 1 - 1e-4)), float)
    for w in weights:
        if not w.weight:
            continue
        zv = _z(df, w.feature)
        if zv is not None:
            s += w.weight * zv

    point = expit(s)
    recovery_rate = 1.0 - np.exp(-2.0 * workout_months / 24.0)
    point = point * (1.0 - 0.5 * recovery_rate)

    a, b = point * 20.0, (1 - point) * 20.0
    return pd.Series(
        rng.beta(np.clip(a, 1e-3, None), np.clip(b, 1e-3, None)),
        index=df.index, name="lgd",
    )


# ──────────────────────────────────────────────────────────────────────────────
# EAD
# ──────────────────────────────────────────────────────────────────────────────
def ead(
    df: pd.DataFrame,
    base_util: float,
    weights: list[W],
    ccf: float,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    s = np.full(len(df), _logit(np.clip(base_util, 1e-4, 1 - 1e-4)), float)
    for w in weights:
        if not w.weight:
            continue
        zv = _z(df, w.feature)
        if zv is not None:
            s += w.weight * zv

    point = expit(s)
    a, b  = point * 15.0, (1 - point) * 15.0
    util  = rng.beta(np.clip(a, 1e-3, None), np.clip(b, 1e-3, None))
    lim   = (df["loan_amount"].to_numpy(float)
             if "loan_amount" in df.columns else np.ones(len(df)))
    e = util * lim + ccf * (1 - util) * lim
    return (
        pd.Series(e,    index=df.index, name="ead"),
        pd.Series(util, index=df.index, name="credit_utilisation"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Threshold / shifts
# ──────────────────────────────────────────────────────────────────────────────
def credit_decision(pd_values: pd.Series, threshold: float) -> pd.Series:
    """1 = loan GRANTED (PD ≤ threshold), 0 = loan DENIED."""
    return (pd_values <= threshold).astype(int)


def covariate_shift(df: pd.DataFrame, shifts: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for c, k in shifts.items():
        if c not in df.columns or k == 0:
            continue
        if df[c].dtype.kind not in "iuf":
            continue
        sd    = df[c].to_numpy(float).std() + 1e-9
        df[c] = df[c] + k * sd
    return df


def concept_drift_lgd(lgd_series: pd.Series, drift: float) -> pd.Series:
    if drift == 0:
        return lgd_series
    eps = 1e-6
    x   = np.clip(lgd_series, eps, 1 - eps)
    log = np.log(x / (1 - x)) + drift
    return pd.Series(1 / (1 + np.exp(-log)),
                     index=lgd_series.index, name=lgd_series.name)
