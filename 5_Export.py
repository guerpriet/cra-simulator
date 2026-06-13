"""Export — one merged CSV with all generated columns, plus individual files and config."""
from __future__ import annotations
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

import pandas as pd
import streamlit as st
from state import get_cfg

st.set_page_config(page_title="Export", layout="wide")
cfg = get_cfg()
st.title("Export")

# ── Collect all generated datasets from session state ────────────────────────
datasets: dict[str, pd.DataFrame] = {}

if "ds_pd" in st.session_state:
    datasets["PD_fairness"] = st.session_state["ds_pd"]["data"]

if "ds_rej" in st.session_state:
    datasets["Reject_inference"] = st.session_state["ds_rej"]["data"]

if "ds_shift" in st.session_state:
    r = st.session_state["ds_shift"]
    datasets["Shift_source"] = r["source"]
    datasets["Shift_target"] = r["target"]

# ── Merged dataset ─────────────────────────────────────────────────────────
st.subheader("⬇️  Merged dataset (recommended)")
st.caption(
    "All generated datasets are joined column-by-column on a shared base. "
    "New columns from each page (e.g. `lgd`, `ead`, `pd_score`, `default`) are added "
    "alongside the feature columns. Use this file with `analyze_cra.py`."
)

if not datasets:
    st.info("No datasets generated yet — go through pages 2–4 first, then come back here.")
else:
    # Start with the PD dataset as the base (has the feature columns + pd_score + default)
    # If not available, use whichever came first.
    base_key = "PD_fairness" if "PD_fairness" in datasets else list(datasets.keys())[0]
    merged = datasets[base_key].copy()

    # Add columns from other datasets that are not yet present
    for key, df in datasets.items():
        if key == base_key:
            continue
        for col in df.columns:
            if col not in merged.columns:
                if len(df) == len(merged):
                    merged[col] = df[col].values
                else:
                    # Different row count — skip merging, will be offered separately
                    pass

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(merged):,}")
    c2.metric("Columns", f"{len(merged.columns)}")
    c3.metric("Datasets merged", f"{len(datasets)}")

    # Column overview
    with st.expander("Column overview", expanded=False):
        col_info = pd.DataFrame({
            "column": merged.columns,
            "dtype":  [str(merged[c].dtype) for c in merged.columns],
            "non-null": [merged[c].notna().sum() for c in merged.columns],
            "sample": [str(merged[c].iloc[0])[:40] if len(merged) > 0 else "" for c in merged.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    st.dataframe(merged.head(10), use_container_width=True)

    csv_merged = merged.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download  dataset.csv  (all columns merged)",
        csv_merged,
        file_name="dataset.csv",
        mime="text/csv",
        type="primary",
    )

    # ── If row counts differ, offer stacked version too ──────────────────────
    if "ds_shift" in st.session_state:
        r = st.session_state["ds_shift"]
        src, tgt = r["source"], r["target"]
        if len(src) != len(tgt):
            st.info(
                "ℹ️  Source and target datasets have different row counts, "
                "so their rows cannot be merged side-by-side. "
                "A **stacked** version (source rows on top, target rows below) "
                "is available below."
            )
            stacked = pd.concat(
                [src.assign(domain="source"), tgt.assign(domain="target")],
                ignore_index=True
            )
            st.download_button(
                "⬇️  Download  dataset_stacked.csv  (source + target rows)",
                stacked.to_csv(index=False).encode(),
                file_name="dataset_stacked.csv",
                mime="text/csv",
            )

    st.divider()

    # ── Individual files ──────────────────────────────────────────────────────
    st.subheader("Individual datasets")
    st.caption("Download each page's output separately if needed.")
    for name, df in datasets.items():
        with st.expander(f"{name}  —  {len(df):,} rows × {len(df.columns)} cols"):
            st.dataframe(df.head(5), use_container_width=True)
            st.download_button(
                f"⬇️  {name}.csv",
                df.to_csv(index=False).encode(),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"ind_{name}",
            )

st.divider()

# ── Config JSON ───────────────────────────────────────────────────────────────
st.subheader("Configuration")
st.caption("Save this JSON to reproduce the exact same dataset later (same seed + same settings).")
j = json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str)
st.download_button(
    "⬇️  crasim_config.json",
    j,
    file_name="crasim_config.json",
    mime="application/json",
)
with st.expander("Show config JSON"):
    st.code(j, language="json")
