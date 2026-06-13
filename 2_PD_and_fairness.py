"""PD model with fairness mechanics (Robertson 2024, Kozodoi 2022/2025, Lessmann et al.)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import streamlit as st
from scipy.special import expit
from state import get_cfg, store
from config import W, Path as CPath, SensitiveCfg
from catalog import CATALOG
from generators import gen_pd

st.set_page_config(page_title="PD + fairness", layout="wide")
cfg = get_cfg()
st.title("PD model and fairness")

# ── Formula ───────────────────────────────────────────────────────────────────
with st.expander("📐  PD model — formula and legend", expanded=True):
    col_form, col_leg = st.columns([3, 2])
    with col_form:
        st.markdown("#### Probability of Default (PD)")
        st.latex(r"""
\text{logit}(\widehat{PD}_i)
  = \underbrace{\beta_0}_{\text{intercept}}
  + \underbrace{\sum_{j} w_j \cdot z_{ij}}_{\text{feature scores}}
  + \underbrace{\sum_{k} b_k \cdot \mathbf{1}[\text{disadv.}_{k,i}]}_{\text{bias terms}}
  + \underbrace{\varepsilon_i}_{\text{noise } \mathcal{N}(0,\,0.5)}
""")
        st.latex(r"""
\widehat{PD}_i = \sigma(\text{logit}_i),
\qquad
D_i \sim \text{Bernoulli}(\widehat{PD}_i)
""")
    with col_leg:
        st.markdown("#### Legend")
        st.markdown(r"""
| Symbol | Meaning |
|---|---|
| $\beta_0$ | Intercept — baseline log-odds; $\sigma(\beta_0)$ = base default rate |
| $w_j$ | Weight for feature $j$ |
| $z_{ij}$ | Standardised feature: $(x_{ij}-\bar x_j)/\sigma_j$ |
| $b_k$ | Direct bias for sensitive attribute $k$ |
| $\mathbf{1}[\text{disadv.}_{k,i}]$ | 1 if borrower $i$ is in the disadvantaged group for attr $k$ |
| $\sigma(\cdot)$ | Logistic (sigmoid) function |
| $D_i$ | Actual default — Bernoulli draw from $\widehat{PD}_i$ |

**Disadvantaged group definition** per attribute type:
- Binary {0,1}: value = 1
- Categorical: not the reference category (set per attribute below)
- Numeric continuous: below sample median
""")

st.divider()

if not cfg.features.selected:
    st.warning("Pick features on the Features page first.")
    st.stop()

# ── Intercept ─────────────────────────────────────────────────────────────────
st.subheader("Baseline  $\\beta_0$")
# Show the implied base default rate next to the slider
baseline_pd = float(expit(cfg.pd.intercept)) * 100
st.caption(
    f"Current intercept → baseline default rate ≈ **{baseline_pd:.1f} %** "
    "(before feature weights, noise, and bias terms)."
)
if "pd_intercept" not in st.session_state:
    st.session_state["pd_intercept"] = float(cfg.pd.intercept)
cfg.pd.intercept = st.slider(
    "Intercept  $\\beta_0$  (log-odds)",
    -6.0, 0.0, step=0.1, key="pd_intercept",
    help="σ(−4.6) ≈ 1 %  |  σ(−3.0) ≈ 5 %  |  σ(−2.2) ≈ 10 %  |  σ(−1.4) ≈ 20 %",
)
# Update caption live
cfg.pd.intercept = float(cfg.pd.intercept)
st.caption(
    f"→ σ({cfg.pd.intercept:.1f}) = **{float(expit(cfg.pd.intercept))*100:.1f} %** base default rate"
)

st.divider()

# ── PD weights ────────────────────────────────────────────────────────────────
st.subheader("PD weights  $w_j$")
st.caption("Positive → raises PD (riskier); negative → lowers PD (safer).")

m = {w.feature: w.weight for w in cfg.pd.weights}
for f in cfg.features.selected:
    k = f"pdw_{f}"
    if k not in st.session_state:
        st.session_state[k] = float(m.get(f, 0.0))

new: list[W] = []
cols = st.columns(3)
for i, f in enumerate(cfg.features.selected):
    with cols[i % 3]:
        v = st.slider(f, -3.0, 3.0, step=0.05, key=f"pdw_{f}")
        new.append(W(feature=f, weight=float(v)))
cfg.pd.weights = new

st.divider()

# ── Sensitive attributes ──────────────────────────────────────────────────────
st.subheader("Sensitive attributes and direct bias  $b_k$")
st.caption(
    "Select one or more sensitive attributes. "
    "Each gets its **own independent** bias $b_k$ and counterfactual paths."
)

attr_map: dict[str, SensitiveCfg] = {a.name: a for a in cfg.fair.attrs}
current_names = [a.name for a in cfg.fair.attrs if a.name in cfg.features.selected]

selected_names: list[str] = st.multiselect(
    "Sensitive attributes  $A_k$",
    options=cfg.features.selected,
    default=current_names,
    key="fair_attr_select",
)

cfg.fair.counterfactual = st.toggle(
    "Enable counterfactual evaluation  (Robertson et al., 2024)",
    value=cfg.fair.counterfactual,
    key="fair_cf_toggle",
)
if cfg.fair.counterfactual:
    st.caption(
        "For each sensitive attribute: flip $A_k$ (propagating through its causal paths), "
        "then measure $|\\widehat{PD}^{\\text{real}} - \\widehat{PD}^{\\text{cf}}|$. "
        "Real and CF datasets share the same noise realization so that only the "
        "causal effect of flipping $A_k$ is measured."
    )

# Build catalog lookup for category info
cat_spec = {s.name: s for s in CATALOG}

new_attrs: list[SensitiveCfg] = []
for attr_name in selected_names:
    prev = attr_map.get(attr_name, SensitiveCfg(name=attr_name))

    with st.expander(f"⚖️  **{attr_name}**", expanded=True):

        # ── Reference / advantaged group selector ──────────────────────────
        spec = cat_spec.get(attr_name)
        reference_cat = prev.reference_cat

        if spec and spec.dtype == "cat":
            cats = spec.params.get("categories", [])
            # Set default index: previously saved ref_cat, else index 0
            ref_idx = cats.index(prev.reference_cat) if prev.reference_cat in cats else 0
            ref_key = f"fair_refcat_{attr_name}"
            if ref_key not in st.session_state:
                st.session_state[ref_key] = prev.reference_cat if prev.reference_cat in cats else cats[ref_idx]
            reference_cat = st.selectbox(
                "Reference (advantaged) group",
                cats, ref_idx,
                key=ref_key,
                help="'Disadvantaged' = everyone NOT in this category. "
                     "The bias b is added to their logit.",
            )
        elif spec and spec.dtype == "bin":
            st.caption("Binary attribute: **0 = advantaged** (reference), **1 = disadvantaged**.")
            reference_cat = ""
        else:
            st.caption("Numeric attribute: **below-median = disadvantaged**. "
                       "Use b > 0 to penalise lower values, b < 0 to penalise higher values.")
            reference_cat = ""

        # ── Bias slider ────────────────────────────────────────────────────
        bias_key = f"fair_bias_{attr_name}"
        if bias_key not in st.session_state:
            st.session_state[bias_key] = float(prev.bias)
        bias = st.slider(
            "Direct bias  $b$  (positive → disadvantaged group gets higher PD)",
            -3.0, 3.0, step=0.05, key=bias_key,
        )

        # ── Counterfactual paths ───────────────────────────────────────────
        paths: list[CPath] = []
        if cfg.fair.counterfactual:
            pm = {p.feature: p for p in prev.paths}
            for f in cfg.features.selected:
                if f == attr_name:
                    continue
                cf_key = f"cf_{attr_name}_{f}"
                if cf_key not in st.session_state:
                    st.session_state[cf_key] = float(pm.get(f, CPath(feature=f)).strength)
                s = st.slider(
                    f"Path strength  {f} ← {attr_name}",
                    -2.0, 2.0, step=0.05, key=cf_key,
                )
                if s != 0.0:
                    paths.append(CPath(feature=f, strength=float(s)))

    new_attrs.append(SensitiveCfg(
        name=attr_name, bias=bias, paths=paths, reference_cat=reference_cat
    ))

cfg.fair.attrs = new_attrs

if not selected_names:
    st.info("No sensitive attributes selected — bias terms are omitted from the PD model.")

st.divider()

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("Generate dataset", type="primary"):
    store("ds_pd", gen_pd(cfg))

r = st.session_state.get("ds_pd")
if r:
    df, meta = r["data"], r["meta"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Default rate (Bernoulli)", f"{meta['default_rate']*100:.1f}%")
    c3.metric("Mean PD", f"{df['pd'].mean()*100:.1f}%")

    cf_cols: list[str] = meta.get("cf_cols", [])
    if cf_cols:
        cols_cf = st.columns(len(cf_cols))
        for col_ui, cf_col in zip(cols_cf, cf_cols):
            aname = cf_col.replace("cf_pd_diff_", "")
            col_ui.metric(f"CF unfairness — {aname}  (mean |ΔPD|)", f"{df[cf_col].mean():.3f}")

    st.dataframe(df.head(20), use_container_width=True)
