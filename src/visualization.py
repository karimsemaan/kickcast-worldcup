"""
visualization.py — Visualization utilities for World Cup simulation results.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


CONFED_COLORS = {
    "UEFA": "#003f88",
    "CONMEBOL": "#2a9d8f",
    "CONCACAF": "#e63946",
    "CAF": "#e9c46a",
    "AFC": "#f4a261",
    "OFC": "#8ecae6",
}


def get_confed(team, groups_df):
    """Get confederation for a team."""
    row = groups_df[groups_df["team"] == team]
    if len(row) > 0:
        return row.iloc[0]["confederation"]
    return "Unknown"
