"""Streamlit session-state wrappers."""
from __future__ import annotations
import copy
import streamlit as st
from config import Cfg, DEFAULT_CAUSAL_LINKS

_KEY = "cfg"

def get_cfg() -> Cfg:
    if _KEY not in st.session_state:
        cfg = Cfg()
        # Populate with economically motivated default causal links
        cfg.causal = copy.deepcopy(DEFAULT_CAUSAL_LINKS)
        st.session_state[_KEY] = cfg
    return st.session_state[_KEY]

def store(k, v): st.session_state[k] = v
def fetch(k, d=None): return st.session_state.get(k, d)
