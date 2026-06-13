"""Feature selection, distribution parameters, noise, causal links,
and a full catalog manager (add / edit / delete → persisted to catalog.json)."""
from __future__ import annotations
import sys, json as _json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pandas as pd
import streamlit as st

import catalog as _cat
from catalog import Spec, by_category, get, save, reload, categories
from state import get_cfg
from config import CausalLink
from synth import DIST_OPTIONS, DIST_LABELS, DIST_DEFAULT_PARAMS

st.set_page_config(page_title="Features", layout="wide")
cfg = get_cfg()
st.title("Features")


# ════════════════════════════════════════════════════════════════════════════
# 1.  CATALOG MANAGER
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Feature catalog")
st.caption(
    "All features come from **catalog.json**. "
    "Add your own, tweak parameters inline, or delete rows — then press **Save catalog**."
)

def _catalog_to_df() -> pd.DataFrame:
    rows = [
        {
            "name":       s.name,
            "category":   s.category,
            "dtype":      s.dtype,
            "dist":       s.params.get("dist", "normal") if s.dtype == "num" else "—",
            "meaningful": s.meaningful,
            "params":     _json.dumps(s.params),
            "🗑 delete":  False,
        }
        for s in _cat.CATALOG
    ]
    return pd.DataFrame(rows)


edited_df = st.data_editor(
    _catalog_to_df(),
    column_config={
        "name":       st.column_config.TextColumn("Name",          disabled=True),
        "category":   st.column_config.TextColumn("Category"),
        "dtype":      st.column_config.SelectboxColumn("Type",     options=["num", "cat", "bin"]),
        "dist":       st.column_config.SelectboxColumn("Distribution (num only)",
                          options=DIST_OPTIONS + ["—"]),
        "meaningful": st.column_config.CheckboxColumn("Meaningful"),
        "params":     st.column_config.TextColumn("Params (JSON)", width="large"),
        "🗑 delete":  st.column_config.CheckboxColumn("🗑 Delete"),
    },
    use_container_width=True,
    num_rows="fixed",
    key="cat_editor",
)

# ── "Add new feature" form ────────────────────────────────────────────────────
with st.expander("➕ Add new feature", expanded=False):
    with st.form("add_feat_form", clear_on_submit=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns([2, 2, 1, 1])
        new_name       = r1c1.text_input("Feature name *")
        cat_opts       = categories() + ["— new category —"]
        cat_sel        = r1c2.selectbox("Category", cat_opts)
        new_dtype      = r1c3.selectbox("Type", ["num", "cat", "bin"])
        new_meaningful = r1c4.checkbox("Meaningful", value=True)

        if cat_sel == "— new category —":
            new_cat = st.text_input("New category name *")
        else:
            new_cat = cat_sel

        st.markdown("**Parameters**")

        if new_dtype == "num":
            # Distribution type selector
            dist_label_list = [DIST_LABELS[d] for d in DIST_OPTIONS]
            dist_choice_label = st.selectbox(
                "Distribution type",
                dist_label_list,
                index=0,
                help="Choose how values for this feature are sampled.",
            )
            selected_dist = DIST_OPTIONS[dist_label_list.index(dist_choice_label)]

            st.caption(f"**{DIST_LABELS[selected_dist]}**")

            # Show relevant parameter inputs based on distribution
            if selected_dist == "normal":
                c1, c2, c3, c4 = st.columns(4)
                p_mean = c1.number_input("mean",  value=0.0)
                p_sd   = c2.number_input("sd",    value=1.0, min_value=0.001)
                p_min  = c3.number_input("clip min (optional)", value=-999.0)
                p_max  = c4.number_input("clip max (optional)", value=999.0)
                new_params: dict = {"dist": "normal", "mean": p_mean, "sd": p_sd,
                                    "min": p_min, "max": p_max}

            elif selected_dist == "uniform":
                c1, c2 = st.columns(2)
                p_min = c1.number_input("min",  value=0.0)
                p_max = c2.number_input("max",  value=1.0)
                new_params = {"dist": "uniform", "min": p_min, "max": p_max}

            elif selected_dist == "lognormal":
                c1, c2, c3, c4 = st.columns(4)
                p_mu    = c1.number_input("mu (log-mean)",  value=0.0)
                p_sigma = c2.number_input("sigma (log-sd)", value=0.5, min_value=0.001)
                p_min   = c3.number_input("clip min (optional)", value=0.0)
                p_max   = c4.number_input("clip max (optional)", value=999.0)
                new_params = {"dist": "lognormal", "mu": p_mu, "sigma": p_sigma,
                              "min": p_min, "max": p_max}

            elif selected_dist == "exponential":
                c1, c2, c3 = st.columns(3)
                p_scale  = c1.number_input("scale (mean of dist)", value=1.0, min_value=0.001)
                p_offset = c2.number_input("offset (shift)",       value=0.0)
                p_max    = c3.number_input("clip max (optional)",  value=999.0)
                new_params = {"dist": "exponential", "scale": p_scale,
                              "offset": p_offset, "max": p_max}

            elif selected_dist == "beta":
                c1, c2, c3, c4 = st.columns(4)
                p_alpha = c1.number_input("α (alpha)",      value=2.0, min_value=0.01)
                p_beta_ = c2.number_input("β (beta)",       value=2.0, min_value=0.01)
                p_lo    = c3.number_input("scale min (lo)", value=0.0)
                p_hi    = c4.number_input("scale max (hi)", value=1.0)
                new_params = {"dist": "beta", "alpha": p_alpha, "beta_": p_beta_,
                              "lo": p_lo, "hi": p_hi}

            elif selected_dist in ("right_skewed", "left_skewed"):
                tail_label = "Right-skewed (Gamma)" if selected_dist == "right_skewed" \
                             else "Left-skewed (Reflected Gamma)"
                st.caption(tail_label)
                c1, c2, c3 = st.columns(3)
                p_shape  = c1.number_input("shape (k ≥ 1 for mode > 0)", value=2.0, min_value=0.01)
                p_scale  = c2.number_input("scale (θ)",                   value=1.0, min_value=0.001)
                p_offset = c3.number_input("offset (shift)",              value=0.0)
                new_params = {"dist": selected_dist, "shape": p_shape,
                              "scale": p_scale, "offset": p_offset}

            else:
                new_params = DIST_DEFAULT_PARAMS.get(selected_dist, {"dist": selected_dist})

        elif new_dtype == "cat":
            cats_raw  = st.text_input("Categories (comma-separated) *", value="a, b, c")
            probs_raw = st.text_input("Probabilities (comma-separated, optional — must sum to 1)")
            cats_list = [c.strip() for c in cats_raw.split(",") if c.strip()]
            if probs_raw.strip():
                try:
                    probs = [float(x) for x in probs_raw.split(",")]
                    new_params = {"categories": cats_list, "probs": probs}
                except ValueError:
                    st.warning("Probabilities could not be parsed; they will be ignored.")
                    new_params = {"categories": cats_list}
            else:
                new_params = {"categories": cats_list}

        else:  # bin
            p_p = st.number_input("Probability p (0–1)", value=0.5,
                                   min_value=0.0, max_value=1.0, step=0.01)
            new_params = {"p": p_p}

        add_clicked = st.form_submit_button("Add feature", type="primary")

    if add_clicked:
        err = None
        if not new_name.strip():
            err = "Feature name is required."
        elif new_name.strip() in {s.name for s in _cat.CATALOG}:
            err = f"A feature named **{new_name}** already exists."
        elif not new_cat.strip():
            err = "Category name is required."
        if err:
            st.error(err)
        else:
            _cat.CATALOG.append(Spec(
                name=new_name.strip(),
                category=new_cat.strip(),
                dtype=new_dtype,
                params=new_params,
                meaningful=new_meaningful,
            ))
            save()
            st.success(f"✅ Feature **{new_name}** added and catalog saved.")
            st.rerun()

# ── Save / discard buttons ────────────────────────────────────────────────
col_save, col_reset = st.columns([1, 5])

with col_save:
    if st.button("💾 Save catalog", type="primary"):
        errors: list[str] = []
        new_specs: list[Spec] = []

        for _, row in edited_df.iterrows():
            if row["🗑 delete"]:
                continue
            try:
                params = _json.loads(row["params"])
            except _json.JSONDecodeError:
                errors.append(f"**{row['name']}**: params is not valid JSON — kept as-is.")
                try:
                    params = get(row["name"]).params
                except KeyError:
                    params = {}

            # If user changed the dist column in the table, update params["dist"]
            dist_val = row.get("dist", "—")
            if row["dtype"] == "num" and dist_val != "—" and dist_val in DIST_OPTIONS:
                params["dist"] = dist_val

            new_specs.append(Spec(
                name=str(row["name"]),
                category=str(row["category"]),
                dtype=row["dtype"],
                params=params,
                meaningful=bool(row["meaningful"]),
            ))

        if errors:
            for e in errors:
                st.warning(e)

        _cat.CATALOG.clear()
        _cat.CATALOG.extend(new_specs)
        save()

        valid = {s.name for s in _cat.CATALOG}
        cfg.features.selected = [f for f in cfg.features.selected if f in valid]

        st.success(f"✅ Catalog saved — {len(new_specs)} features.")
        st.rerun()

with col_reset:
    if st.button("↺ Discard edits"):
        reload()
        st.rerun()

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE SELECTION
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Select features for this run")
st.caption(f"{len(_cat.CATALOG)} features in catalog. Tick the ones to include in the dataset.")

sel = set(cfg.features.selected)
cols = st.columns(2)
for i, (cat, specs) in enumerate(by_category().items()):
    with cols[i % 2]:
        st.markdown(f"**{cat}**")
        for s in specs:
            label = s.name + ("  *(meaningless token)*" if not s.meaningful else "")
            on = st.checkbox(label, value=(s.name in sel), key=f"f_{s.name}")
            if on:
                sel.add(s.name)
            else:
                sel.discard(s.name)
cfg.features.selected = sorted(sel)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# 3.  DISTRIBUTION PARAMETERS (numeric features only)
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Distribution parameters (numeric features)")
st.caption(
    "Edit mean / sd for Normal features here. "
    "For other distributions, edit params directly in the catalog table above."
)

num_rows = [
    {
        "feature":  n,
        "dist":     get(n).params.get("dist", "normal"),
        "mean":     float(get(n).params.get("mean", 0.0)) if get(n).params.get("dist", "normal") == "normal" else float("nan"),
        "sd":       float(get(n).params.get("sd",   1.0))  if get(n).params.get("dist", "normal") == "normal" else float("nan"),
    }
    for n in cfg.features.selected
    if get(n).dtype == "num"
]

if num_rows:
    edited_params = st.data_editor(
        pd.DataFrame(num_rows),
        column_config={
            "feature": st.column_config.TextColumn("Feature",     disabled=True),
            "dist":    st.column_config.TextColumn("Distribution", disabled=True),
            "mean":    st.column_config.NumberColumn("mean (Normal only)", format="%.3f"),
            "sd":      st.column_config.NumberColumn("sd   (Normal only)", format="%.3f"),
        },
        disabled=["feature", "dist"],
        num_rows="fixed",
        use_container_width=True,
        key="param_editor",
    )
    for _, r in edited_params.iterrows():
        spec = get(r["feature"])
        if spec.params.get("dist", "normal") == "normal":
            import math
            if not math.isnan(float(r["mean"])):
                spec.params["mean"] = float(r["mean"])
            if not math.isnan(float(r["sd"])):
                spec.params["sd"] = float(r["sd"])
else:
    st.info("No numeric features selected.")

# Legend for distributions
with st.expander("ℹ️  Distribution types reference", expanded=False):
    st.markdown("""
| Distribution | Parameters | Typical use |
|---|---|---|
| **Normal** | mean, sd | Age, income, debt |
| **Uniform** | min, max | Equally likely values in a range |
| **Log-normal** | mu, sigma | Loan amounts, income (right-skewed, positive) |
| **Exponential** | scale, offset | Time between events, waiting times |
| **Beta** | α, β, lo, hi | Rates and probabilities bounded in [lo, hi] |
| **Right-skewed** (Gamma) | shape, scale, offset | Positive quantities with long right tail |
| **Left-skewed** (Reflected Gamma) | shape, scale, offset | Values with long left tail |

> **Tip:** parameters are stored in `catalog.json` under each feature's `params` key,
> so the JSON file always reflects what you see in the GUI.
""")

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# 4.  NOISE COLUMNS
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Noise columns")
cfg.features.n_noise = st.slider(
    "Number of pure-noise columns appended to every dataset",
    0, 50, value=cfg.features.n_noise, step=1,
)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# 5.  CAUSAL LINKS
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Causal links")
st.caption("Driver shifts target columns by  strength × σ(target) × z(driver).")

if st.button("Suggest defaults"):
    from config import DEFAULT_CAUSAL_LINKS
    import copy
    cfg.causal = copy.deepcopy(DEFAULT_CAUSAL_LINKS)
    st.success("Default causal links loaded. These reflect key economic relationships: "
               "missed payments → bureau score, risk-based pricing, income → loan size.")
if st.button("Clear all"):
    cfg.causal = []

new_causal: list[CausalLink] = []
for i, L in enumerate(cfg.causal):
    with st.container(border=True):
        a, b, c = st.columns([2, 4, 1])
        opts = cfg.features.selected or ([L.driver] if L.driver else [""])
        idx  = opts.index(L.driver) if L.driver in opts else 0
        with a: drv  = st.selectbox("Driver", opts, idx, key=f"d_{i}")
        with b: tgts = st.multiselect("Targets", cfg.features.selected,
                                       default=[t for t in L.targets if t in cfg.features.selected],
                                       key=f"t_{i}")
        with c: stg  = st.number_input("Strength", -3.0, 3.0, float(L.strength), 0.1, key=f"s_{i}")
        if not st.checkbox("delete", key=f"x_{i}"):
            new_causal.append(CausalLink(driver=drv, targets=tgts, strength=float(stg)))

if st.button("Add link"):
    new_causal.append(CausalLink(
        driver=cfg.features.selected[0] if cfg.features.selected else ""
    ))
cfg.causal = new_causal
