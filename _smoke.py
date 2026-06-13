"""Headless smoke-test of the three pipelines (updated for v06)."""
from __future__ import annotations
import sys, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import (Cfg, Run, FeatureSet, W, PD, Fair, SensitiveCfg,
                    Path as CPath, Reject, LGD, EAD, Shift, CausalLink,
                    DEFAULT_CAUSAL_LINKS)
from generators import gen_pd, gen_reject, gen_shift
import copy


def demo() -> Cfg:
    feats = ["age", "gender", "net_income", "current_debt", "debt_history",
             "bureau_score", "loan_amount", "term_months", "interest_rate_pct",
             "collateral", "ltv"]
    return Cfg(
        run=Run(n=2000, seed=123, threshold=0.4),
        features=FeatureSet(selected=feats, n_noise=3),
        pd=PD(weights=[W("bureau_score", -1.2), W("net_income", -0.5),
                       W("current_debt", 0.7), W("loan_amount", 0.3),
                       W("interest_rate_pct", 0.4)]),
        fair=Fair(
            attrs=[SensitiveCfg(
                name="gender",
                bias=0.4,
                paths=[CPath("net_income", -0.3), CPath("debt_history", -0.2)],
            )],
            counterfactual=True,
        ),
        reject=Reject(accept_rate=0.6, mechanism="MNAR",
                      selection_features=["bureau_score", "net_income"]),
        lgd=LGD(base=0.45, workout_months=24,
                weights=[W("ltv", 0.8), W("net_income", -0.4)]),
        ead=EAD(base_util=0.4, ccf=0.55,
                weights=[W("bureau_score", -0.3)]),
        shift=Shift(n_target=1000,
                    covariate_shifts={"net_income": -0.5, "interest_rate_pct": 0.4},
                    lgd_drift=0.3, drop_in_target=["debt_history"]),
        causal=copy.deepcopy(DEFAULT_CAUSAL_LINKS),
    )


def main() -> int:
    cfg = demo()
    r1 = gen_pd(cfg)
    df = r1["data"]
    print(f"PD            rows={len(df)} default={r1['meta']['default_rate']:.3f} "
          f"cf_diff={df['cf_pd_diff'].mean():.4f}")
    assert df["pd"].between(0, 1).all(), "PD out of [0,1]"

    # Check integer fields
    for col in ["age", "term_months"]:
        if col in df.columns:
            assert df[col].dtype in ["int32", "int64", "int8", "int16"], f"{col} not integer: {df[col].dtype}"
    print(f"  age dtype={df['age'].dtype}, age sample={df['age'].head(3).tolist()}")

    # Check DTI is derived (correlates with debt/income)
    if "dti" in df.columns and "current_debt" in df.columns and "net_income" in df.columns:
        computed = (df["current_debt"] / df["net_income"]).clip(0, 2)
        corr = df["dti"].corr(computed)
        assert corr > 0.99, f"DTI not derived from debt/income: corr={corr:.3f}"
        print(f"  dti correlation with current_debt/net_income: {corr:.4f} ✓")

    # Check LTV is NaN for non-collateralized loans
    if "ltv" in df.columns and "collateral" in df.columns:
        non_coll = df[df["collateral"] == 0]["ltv"]
        assert non_coll.isna().all(), f"LTV not NaN for uncollateralized loans: {non_coll.notna().sum()} have values"
        print(f"  ltv NaN for all collateral=0 rows ✓")

    # Check income differs by employment
    if "employment_status" in df.columns and "net_income" in df.columns:
        inc = df.groupby("employment_status")["net_income"].mean()
        if "unemployed" in inc.index and "employed" in inc.index:
            assert inc["unemployed"] < inc["employed"] * 0.8, "Unemployed income not lower"
            print(f"  income: unemployed={inc.get('unemployed',0):.0f} < employed={inc.get('employed',0):.0f} ✓")

    # Check bureau_score has more spread
    bs_std = df["bureau_score"].std() if "bureau_score" in df.columns else None
    if bs_std is not None:
        print(f"  bureau_score std={bs_std:.1f} (was ~3.4 in v05, target >7)")

    r2 = gen_reject(cfg)
    print(f"Reject        accept={r2['meta']['accept_rate']:.3f} "
          f"acc_dr={r2['data'][r2['data']['accepted']==1]['default'].mean():.3f} "
          f"rej_dr={r2['data'][r2['data']['accepted']==0]['default'].mean():.3f}")
    assert 0.55 < r2["meta"]["accept_rate"] < 0.65

    r3 = gen_shift(cfg)
    print(f"Shift         lgd_src={r3['source']['lgd'].mean():.3f} "
          f"lgd_tgt={r3['target']['lgd'].mean():.3f} "
          f"src_cols={len(r3['source'].columns)} tgt_cols={len(r3['target'].columns)}")
    assert "debt_history" in r3["source"].columns
    assert "debt_history" not in r3["target"].columns
    print("OK — all assertions passed")
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(1)
