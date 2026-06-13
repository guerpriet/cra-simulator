"""LGD/EAD under distribution shift (Gerlin et al. 2026, Lessmann et al.)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import streamlit as st
from state import get_cfg, store
from config import W
from generators import gen_shift

st.set_page_config(page_title="LGD/EAD + shift", layout="wide")
cfg = get_cfg()
st.title("LGD / EAD and distribution shift")

# ── Formula boxes ─────────────────────────────────────────────────────────────
with st.expander("📐  LGD model — formula and legend", expanded=True):
    col_form, col_leg = st.columns([3, 2])

    with col_form:
        st.markdown("#### Loss Given Default (LGD)")
        st.latex(r"""
\text{logit}(\mu_i^{\text{LGD}})
  = \underbrace{\text{logit}(\text{base\_LGD})}_{\text{baseline log-odds}}
  + \underbrace{\sum_{j} w_j \cdot z_{ij}}_{\text{feature scores}}
""")
        st.latex(r"""
\mu_i^{\text{adj}}
  = \sigma\!\left(\text{logit}(\mu_i^{\text{LGD}})\right)
    \times \bigl(1 - 0.5 \cdot \underbrace{(1 - e^{-2})}_{\approx\,0.865}\bigr)
""")
        st.latex(r"""
\alpha_i = 20\,\mu_i^{\text{adj}},\quad
\beta_i  = 20\,(1 - \mu_i^{\text{adj}})
""")
        st.latex(r"""
\widehat{\text{LGD}}_i \;\sim\; \text{Beta}(\alpha_i,\,\beta_i)
""")

    with col_leg:
        st.markdown("#### Legend")
        st.markdown("""
| Symbol | Meaning |
|---|---|
| $\\widehat{\\text{LGD}}_i$ | Loss Given Default for loan $i$ ∈ [0, 1] |
| $\\text{base\\_LGD}$ | Baseline LGD before feature adjustments |
| $w_j$ | Weight for feature $j$ (set via sliders) |
| $z_{ij}$ | Standardised feature value: $(x_{ij}-\\bar x_j)/\\sigma_j$ |
| $\\sigma(\\cdot)$ | Logistic (sigmoid) function |
| $1 - e^{-2} ≈ 0.865$ | Recovery discount from exponential workout curve over the workout horizon |
| $\\alpha_i,\\, \\beta_i$ | Beta distribution shape parameters (concentration = 20) |
| $\\text{Beta}(\\alpha,\\beta)$ | Beta distribution — produces realised LGD in [0, 1] |

*Based on:* **Gerlin, Peng, Chen & Lessmann (2026)**
— *Transfer Learning for Loan Recovery Prediction under Distribution Shifts.*
""")

st.divider()

with st.expander("📐  EAD model — formula and legend", expanded=True):
    col_form, col_leg = st.columns([3, 2])

    with col_form:
        st.markdown("#### Exposure at Default (EAD)")
        st.latex(r"""
\text{logit}(\mu_i^{\text{util}})
  = \text{logit}(\text{base\_util})
  + \sum_{j} w_j \cdot z_{ij}
""")
        st.latex(r"""
\widehat{\text{util}}_i \;\sim\;
  \text{Beta}\!\left(15\,\mu_i^{\text{util}},\;15\,(1-\mu_i^{\text{util}})\right)
""")
        st.latex(r"""
\widehat{\text{EAD}}_i
  = \underbrace{\widehat{\text{util}}_i \cdot L_i}_{\text{drawn amount}}
  + \underbrace{\text{CCF} \cdot (1 - \widehat{\text{util}}_i) \cdot L_i}_{\text{undrawn commitment}}
""")
        st.latex(r"""
  = L_i \cdot \bigl[\widehat{\text{util}}_i + \text{CCF}\cdot(1-\widehat{\text{util}}_i)\bigr]
""")

    with col_leg:
        st.markdown("#### Legend")
        st.markdown("""
| Symbol | Meaning |
|---|---|
| $\\widehat{\\text{EAD}}_i$ | Exposure at Default for loan $i$ |
| $\\text{base\\_util}$ | Baseline utilisation rate (before feature adjustments) |
| $\\widehat{\\text{util}}_i$ | Realised utilisation rate ∈ [0, 1] |
| $w_j$ | Weight for feature $j$ (set via sliders) |
| $z_{ij}$ | Standardised feature value |
| $L_i$ | Committed credit limit (= `loan_amount`) |
| $\\text{CCF}$ | Credit Conversion Factor: fraction of undrawn limit drawn at default |
| $\\text{Beta}(\\alpha,\\beta)$ | Beta distribution — produces bounded utilisation in [0, 1] |

**Regulatory context:** EAD = amount the bank is owed at moment of default.
CCF converts off-balance-sheet commitments to an on-balance-sheet equivalent
(*Basel III framework*).
""")

st.divider()

if not cfg.features.selected:
    st.warning("Pick features first.")
    st.stop()

# ── LGD ───────────────────────────────────────────────────────────────────────
st.subheader("LGD settings")
c1, c2 = st.columns(2)
with c1: cfg.lgd.base = st.slider("Base LGD  (base_LGD)", 0.01, 0.99, float(cfg.lgd.base), 0.01)
with c2: cfg.lgd.workout_months = st.slider("Workout period (months)", 1, 120,
                                            int(cfg.lgd.workout_months), 1)
st.caption("Feature weights  $w_j$ for LGD:")
m = {w.feature: w.weight for w in cfg.lgd.weights}
new_lgd: list[W] = []
cols = st.columns(3)
for i, f in enumerate(cfg.features.selected):
    with cols[i % 3]:
        v = st.slider(f, -3.0, 3.0, float(m.get(f, 0.0)), 0.05, key=f"lw_{f}")
        if v != 0: new_lgd.append(W(feature=f, weight=float(v)))
cfg.lgd.weights = new_lgd

# ── EAD ───────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("EAD settings")
c1, c2 = st.columns(2)
with c1: cfg.ead.base_util = st.slider("Base utilisation  (base_util)", 0.01, 0.99,
                                        float(cfg.ead.base_util), 0.01)
with c2: cfg.ead.ccf = st.slider("Credit Conversion Factor  (CCF)", 0.0, 1.0,
                                  float(cfg.ead.ccf), 0.01)
st.caption("Feature weights  $w_j$ for EAD utilisation:")
m = {w.feature: w.weight for w in cfg.ead.weights}
new_ead: list[W] = []
cols = st.columns(3)
for i, f in enumerate(cfg.features.selected):
    with cols[i % 3]:
        v = st.slider(f, -3.0, 3.0, float(m.get(f, 0.0)), 0.05, key=f"ew_{f}")
        if v != 0: new_ead.append(W(feature=f, weight=float(v)))
cfg.ead.weights = new_ead

# ── Distribution shift ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Distribution shift — target domain")
st.caption(
    "Simulate a *source → target* transfer scenario "
    "(Gerlin, Peng, Chen & Lessmann, 2026). "
    "Covariate shift changes feature means; concept drift changes the LGD mapping."
)

with st.expander("ℹ️  Shift types explained", expanded=False):
    st.markdown("""
| Shift type | What changes | What stays fixed |
|---|---|---|
| **Covariate shift** | Marginal distribution P(X) — feature means shift | Conditional P(LGD\|X) |
| **Concept drift** (label shift) | Conditional P(LGD\|X) — logit shifted by drift parameter | P(X) |
| **Heterogeneous schema** | Some features are dropped in the target domain | — |

> *Reference:* Gerlin, Peng, Chen & Lessmann (2026); Pan & Yang (2010).
""")

cfg.shift.n_target = st.number_input("Target sample size", 100, 200_000,
                                     int(cfg.shift.n_target), 500)
cfg.shift.lgd_drift = st.slider(
    "LGD concept drift  (logit shift added to target LGD)",
    -2.0, 2.0, float(cfg.shift.lgd_drift), 0.05
)

st.markdown("**Covariate shifts** — mean shift in SD units per feature:")
shifts = dict(cfg.shift.covariate_shifts)
cols = st.columns(3)
for i, f in enumerate(cfg.features.selected):
    with cols[i % 3]:
        v = st.slider(f, -2.0, 2.0, float(shifts.get(f, 0.0)), 0.05, key=f"sh_{f}")
        if v != 0: shifts[f] = v
        elif f in shifts: del shifts[f]
cfg.shift.covariate_shifts = shifts

cfg.shift.drop_in_target = st.multiselect(
    "Drop features in target domain  (heterogeneous schema)",
    cfg.features.selected,
    default=cfg.shift.drop_in_target,
    help="Simulates the case where the target portfolio does not record certain features."
)

st.divider()
if st.button("Generate datasets", type="primary"):
    store("ds_shift", gen_shift(cfg))

r = st.session_state.get("ds_shift")
if r:
    src, tgt, meta = r["source"], r["target"], r["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n source", f"{len(src)}")
    c2.metric("n target", f"{len(tgt)}")
    c3.metric("LGD source (mean)", f"{src['lgd'].mean():.3f}")
    c4.metric("LGD target (mean)", f"{tgt['lgd'].mean():.3f}")
    st.markdown("**Source dataset** (first 15 rows)")
    st.dataframe(src.head(15), use_container_width=True)
    st.markdown("**Target dataset** (first 15 rows)")
    st.dataframe(tgt.head(15), use_container_width=True)
