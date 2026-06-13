"""Landing page and global sidebar."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import streamlit as st
from state import get_cfg

st.set_page_config(page_title="CRA-Sim", layout="wide")
cfg = get_cfg()

with st.sidebar:
    st.header("Global parameters")
    cfg.run.n    = st.number_input("Sample size", 100, 200_000, value=cfg.run.n, step=500)
    cfg.run.seed = st.number_input("Random seed", 0, 2**31 - 1, value=cfg.run.seed, step=1)
    cfg.run.threshold = st.slider(
        "Credit decision threshold  τ  (grant loan if PD ≤ τ)",
        0.01, 0.99, float(cfg.run.threshold), 0.01,
        help="Used to populate the 'decision' column (1 = granted, 0 = denied) "
             "and as the selection threshold in reject inference. "
             "Actual defaults are drawn from Bernoulli(PD) — not from this threshold.",
    )

st.title("Credit Risk Analytics (CRA) Simulator")
st.markdown("""
Generates synthetic credit-risk datasets for research on PD, LGD, EAD, fairness,
and distribution shift. All weights and parameters are set via the GUI.

| Page | Key references | Focus |
|---|---|---|
| **1. Features** | Robertson et al. (2024) | Feature synthesis, distribution types, causal links, noise |
| **2. PD + fairness** | Robertson et al. (2024); Kozodoi, Lessmann et al. (2022/2025) | Counterfactual paths, direct bias |
| **3. Reject inference** | Kozodoi, Lessmann et al. (2025) | Sample selection, MAR vs. MNAR, sampling bias |
| **4. LGD / EAD + shift** | Gerlin, Peng, Chen & Lessmann (2026) | Recovery curve, concept drift, heterogeneous schemas |
| **5. Export** | — | CSV download + JSON config |

**Workflow:** open the pages on the left, top to bottom. State carries over.
""")

st.divider()
st.subheader("Quick reference: distribution types")
st.caption("All numeric features can be sampled from one of the following distributions (set on the Features page).")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
- **Normal** `N(mean, sd)` — symmetric bell curve
- **Uniform** `U(min, max)` — equal probability in a range
- **Log-normal** `LogN(μ, σ)` — positive, right-skewed (income, loan amounts)
- **Exponential** `Exp(scale) + offset` — exponentially decreasing (count-like variables)
""")
with col2:
    st.markdown("""
- **Beta** `Beta(α, β) → [lo, hi]` — flexible bounded distribution (rates, scores)
- **Right-skewed** `Gamma(k, θ) + offset` — long right tail (debts, claim sizes)
- **Left-skewed** `–Gamma(k, θ) + offset` — long left tail
""")

st.divider()
st.subheader("Default generation model")
st.markdown(r"""
Actual defaults are drawn as **Bernoulli realisations** of the predicted PD:

$$D_i \sim \text{Bernoulli}\!\left(\widehat{PD}_i\right)$$

This means a borrower with $\widehat{PD} = 0.12$ has a 12 % *chance* of defaulting,
not a deterministic outcome. The credit decision threshold $\tau$ (sidebar) determines
which loans are granted but does **not** determine who defaults.
""")
