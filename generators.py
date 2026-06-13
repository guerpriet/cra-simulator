"""Three end-to-end generators."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit
from config import Cfg
from synth import (
    make_features, pd_score, add_direct_bias, to_pd,
    counterfactual, credit_decision,
    lgd as compute_lgd, ead as compute_ead,
    covariate_shift, concept_drift_lgd,
)


# ──────────────────────────────────────────────────────────────────────────────
# Use case 1: PD + fairness
# ──────────────────────────────────────────────────────────────────────────────
def gen_pd(cfg: Cfg) -> dict:
    df, rng = make_features(cfg)

    # FIX 3 — draw noise ONCE and reuse it for both real and CF pd_score calls.
    # This ensures |ΔPD| measures only the causal effect of flipping the
    # sensitive attribute, not random noise differences between the two runs.
    shared_noise = rng.normal(0, 0.5, len(df))

    logits, contribs = pd_score(
        df, cfg.pd.weights, intercept=cfg.pd.intercept, noise=shared_noise
    )

    # Apply bias for each sensitive attribute independently
    for attr in cfg.fair.attrs:
        if attr.name in df.columns:
            logits = add_direct_bias(logits, df[attr.name], attr.bias, attr.reference_cat)

    pd_f = to_pd(logits)

    # FIX 1 — Bernoulli default draw: each borrower defaults with probability PD_i.
    # This replaces the old deterministic threshold, which made defaults perfectly
    # predictable from features (AUC = 1 by construction).
    default = pd.Series(
        (rng.random(len(df)) < pd_f.to_numpy()).astype(int),
        index=df.index, name="default",
    )

    out = df.copy()
    out["pd"]      = pd_f
    out["default"] = default
    # Credit decision: 1 = loan granted (PD ≤ threshold), 0 = denied
    out["decision"] = credit_decision(pd_f, cfg.run.threshold)

    # Counterfactual evaluation — one pass per sensitive attribute,
    # reusing the same noise so CF and real datasets are comparable.
    cf_col_names: list[str] = []
    if cfg.fair.counterfactual and cfg.fair.attrs:
        attr_diffs: list[pd.Series] = []
        for attr in cfg.fair.attrs:
            if attr.name not in df.columns:
                continue
            df_cf = counterfactual(df, attr.name, attr.paths, attr.reference_cat)
            logits_cf, _ = pd_score(
                df_cf, cfg.pd.weights, intercept=cfg.pd.intercept, noise=shared_noise
            )
            for a2 in cfg.fair.attrs:
                if a2.name in df_cf.columns:
                    logits_cf = add_direct_bias(
                        logits_cf, df_cf[a2.name], a2.bias, a2.reference_cat
                    )
            diff = (pd_f - to_pd(logits_cf)).abs().rename(f"cf_pd_diff_{attr.name}")
            out[diff.name] = diff
            cf_col_names.append(diff.name)
            attr_diffs.append(diff)
        out["cf_pd_diff"] = sum(attr_diffs) / len(attr_diffs)
    else:
        out["cf_pd_diff"] = 0.0

    meta = dict(
        default_rate=float(default.mean()),
        threshold=cfg.run.threshold,
        counterfactual=cfg.fair.counterfactual,
        sensitive_attrs=[a.name for a in cfg.fair.attrs],
        biases={a.name: a.bias for a in cfg.fair.attrs},
        cf_cols=cf_col_names,
    )
    return dict(data=out, contribs=contribs, meta=meta)


# ──────────────────────────────────────────────────────────────────────────────
# Use case 2: Reject inference
# ──────────────────────────────────────────────────────────────────────────────
def gen_reject(cfg: Cfg) -> dict:
    base = gen_pd(cfg)
    df   = base["data"]
    rng  = np.random.default_rng(cfg.run.seed + 1)

    parts = []
    for f in cfg.reject.selection_features:
        zc = f + "_z"
        if zc in df.columns:
            parts.append(df[zc].to_numpy())
        elif f in df.columns and df[f].dtype.kind in "iuf":
            x = df[f].astype(float).to_numpy()
            sd = x.std() + 1e-9
            parts.append((x - x.mean()) / sd)

    s = 2.0 * np.sum(parts, axis=0) / max(1, len(parts)) if parts else np.zeros(len(df))
    if cfg.reject.mechanism == "MNAR":
        s = s - 2.0 * (df["pd"].to_numpy() - df["pd"].mean())

    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if expit(s + mid).mean() > cfg.reject.accept_rate: hi = mid
        else:                                              lo = mid
    icpt  = (lo + hi) / 2
    p_acc = expit(s + icpt)
    acc   = (rng.random(len(df)) < p_acc).astype(int)

    out = df.copy()
    out["p_accept"]         = p_acc
    out["accepted"]         = acc
    out["default_observed"] = np.where(acc == 1, out["default"], np.nan)

    meta = dict(base["meta"])
    meta.update(
        accept_rate=float(acc.mean()),
        mechanism=cfg.reject.mechanism,
        selection_features=list(cfg.reject.selection_features),
    )
    return dict(data=out, contribs=base["contribs"], meta=meta)


# ──────────────────────────────────────────────────────────────────────────────
# Use case 3: LGD / EAD under distribution shift
# ──────────────────────────────────────────────────────────────────────────────
def _one_domain(cfg: Cfg, n: int, seed_offset: int) -> pd.DataFrame:
    df, rng = make_features(cfg, n=n, seed_offset=seed_offset)
    noise   = rng.normal(0, 0.5, len(df))
    logits, _ = pd_score(df, cfg.pd.weights, intercept=cfg.pd.intercept, noise=noise)
    pd_f    = to_pd(logits)
    df["pd"]       = pd_f
    # FIX 1: Bernoulli default
    df["default"]  = (rng.random(len(df)) < pd_f.to_numpy()).astype(int)
    df["decision"] = credit_decision(pd_f, cfg.run.threshold)
    df["lgd"]      = compute_lgd(df, cfg.lgd.base, cfg.lgd.weights,
                                  cfg.lgd.workout_months, rng)
    e, u = compute_ead(df, cfg.ead.base_util, cfg.ead.weights, cfg.ead.ccf, rng)
    # FIX 7: renamed from "utilisation" to "credit_utilisation"
    df["ead"], df["credit_utilisation"] = e, u
    return df


def gen_shift(cfg: Cfg) -> dict:
    src = _one_domain(cfg, cfg.run.n,          seed_offset=0)
    tgt = _one_domain(cfg, cfg.shift.n_target, seed_offset=10_000)
    tgt = covariate_shift(tgt, cfg.shift.covariate_shifts)
    tgt["lgd"] = concept_drift_lgd(tgt["lgd"], cfg.shift.lgd_drift)
    for c in cfg.shift.drop_in_target:
        if c in tgt.columns: tgt = tgt.drop(columns=c)
    src["domain"] = "source"; tgt["domain"] = "target"
    stacked = pd.concat([src, tgt], ignore_index=True, sort=False)
    meta = dict(
        n_source=len(src), n_target=len(tgt),
        covariate_shifts=dict(cfg.shift.covariate_shifts),
        lgd_drift=cfg.shift.lgd_drift,
        drop_in_target=list(cfg.shift.drop_in_target),
        workout_months=cfg.lgd.workout_months,
        ccf=cfg.ead.ccf,
    )
    return dict(source=src, target=tgt, stacked=stacked, meta=meta)
