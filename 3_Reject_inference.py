"""Reject inference / sampling bias (Kozodoi 2025)."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import streamlit as st
import pandas as pd
from state import get_cfg, store
from generators import gen_reject

st.set_page_config(page_title="Reject inference", layout="wide")
cfg = get_cfg()
st.title("Reject inference and sampling bias")

st.markdown(r"""
Default outcomes are observed only for accepted applicants. The acceptance
probability depends on features (and, in MNAR mode, on the latent true risk).
""")

if not cfg.features.selected or not cfg.pd.weights:
    st.warning("Features + PD weights are required first.")
    st.stop()

cfg.reject.selection_features = st.multiselect(
    "Features feeding the selection model", cfg.features.selected,
    default=cfg.reject.selection_features or
            [f for f in ("bureau_score", "net_income") if f in cfg.features.selected])

cfg.reject.mechanism = st.radio("Mechanism", ["MAR", "MNAR"],
                                horizontal=True,
                                index=0 if cfg.reject.mechanism == "MAR" else 1)
cfg.reject.accept_rate = st.slider("Target accept rate", 0.05, 0.95,
                                   float(cfg.reject.accept_rate), 0.01)

st.divider()
if st.button("Generate dataset", type="primary"):
    store("ds_rej", gen_reject(cfg))

r = st.session_state.get("ds_rej")
if r:
    df, meta = r["data"], r["meta"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Population", f"{len(df)}")
    c2.metric("Accept rate", f"{meta['accept_rate']*100:.1f}%")
    c3.metric("Mechanism", meta["mechanism"])

    acc = df[df["accepted"] == 1]; rej = df[df["accepted"] == 0]
    st.table(pd.DataFrame({
        "Group": ["Population", "Accepts (observed)", "Rejects (truth, hidden)"],
        "n": [len(df), len(acc), len(rej)],
        "Default rate": [f"{df['default'].mean()*100:.1f}%",
                         f"{(acc['default'].mean()*100 if len(acc) else 0):.1f}%",
                         f"{(rej['default'].mean()*100 if len(rej) else 0):.1f}%"],
    }))
    st.dataframe(df.head(20), use_container_width=True)
