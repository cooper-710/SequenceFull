from __future__ import annotations
from pathlib import Path
from typing import Optional
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TrackMan-ish palette (kept flexible; we prune to data later)
PITCH_COLOR_MAP = {
    "FF": "#FF0000", "FT": "#8B0000", "SI": "#FFA500", "FC": "#808080",
    "SL": "#0000FF", "CU": "#800080", "CH": "#008000", "FS": "#00FFFF",
    "KC": "#4B0082", "KN": "#D3D3D3", "EP": "#FFC0CB", "SV": "#4682B4",
    "ST": "#008080", "FO": "#00008B", "SC": "#FF00FF", "CS": "#800000",
    "UN": "#B0C4DE", "PO": "#A52A2A", "IN": "#F0E68C"
}

# Pitch type abbreviation to full name mapping for legend labels
PITCH_NAME_MAP = {
    "FF": "Four-Seam", "FT": "Two-Seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle Curve",
    "CH": "Changeup", "FS": "Splitter", "SV": "Slurve", "CS": "Slow Curve",
    "FO": "Forkball", "SC": "Screwball", "EP": "Eephus", "KN": "Knuckleball"
}

# Normalize weird labels before plotting
def _normalize_pitch_labels(s: pd.Series) -> pd.Series:
    mapping = {
        "FA": "FF",   # generic fastball → Four-Seam
        "FO": "FF",   # forkball tagged fastball → FF
        "SV": "SL",   # slurve variants consolidate to SL
        # "ST": "SL",   # Keep Sweeper separate from Slider
        "KC": "CU",   # knuckle-curve → CU bucket
        "CS": "CU",
        "UN": "FF"    # unknown → safest to bucket with FF
    }
    s = s.astype(str).str.upper().map(lambda x: mapping.get(x, x))
    # Drop truly non-pitch rows if they ever sneak in
    s = s.where(~s.isin({"PO", "IN"}), other=pd.NA)
    return s

def _prep(df_p: pd.DataFrame, normalize_by_throws: bool = False) -> pd.DataFrame:
    required = {"pitch_type", "pfx_x", "pfx_z", "p_throws"}
    if not required.issubset(df_p.columns):
        return pd.DataFrame(columns=list(required))

    d = df_p.copy()
    d["pitch_type"] = _normalize_pitch_labels(d["pitch_type"])
    d = d.dropna(subset=["pitch_type", "pfx_x", "pfx_z"])

    if d.empty:
        return d

    if normalize_by_throws:
        sign = d["p_throws"].map({"R": -1, "L": 1}).fillna(-1)
    else:
        sign = -1

    d["pfx_x_inches"] = d["pfx_x"] * 12 * sign
    d["pfx_z_inches"] = d["pfx_z"] * 12

    return d

def _slug(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip())

def render_movement_plot_png(
    df_pitcher: pd.DataFrame,
    pitcher_name: str,
    out_dir: str = "build/figures",
    include_density: bool = True,
    normalize_by_throws: bool = False,
) -> Optional[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d = _prep(df_pitcher, normalize_by_throws=normalize_by_throws)
    if d.empty:
        # still emit a placeholder so downstream never breaks
        out_path = Path(out_dir) / f"{_slug(pitcher_name)}_movement.png"
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        ax.text(0.5, 0.5, f"No movement data for {pitcher_name}",
                ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(out_path)

    # Drop tiny groups to avoid covariance/KDE failures, but keep plot robust
    counts = d["pitch_type"].value_counts()
    keep_types = counts[counts >= 3].index
    d = d[d["pitch_type"].isin(keep_types)]
    if d.empty:
        return render_movement_plot_png(pd.DataFrame([], columns=d.columns),
                                        pitcher_name, out_dir,
                                        include_density, normalize_by_throws)

    usage_order = d["pitch_type"].value_counts().index.tolist()
    pitch_palette = {k: v for k, v in PITCH_COLOR_MAP.items() if k in usage_order}

    # Modern professional styling
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#fafafa'
    plt.rcParams['axes.edgecolor'] = '#d1d5db'
    plt.rcParams['grid.color'] = '#e5e7eb'
    plt.rcParams['text.color'] = '#0a0a0a'
    plt.rcParams['xtick.color'] = '#4b5563'
    plt.rcParams['ytick.color'] = '#4b5563'
    plt.rcParams['font.family'] = 'Inter, system-ui, sans-serif'
    
    sns.set_style("whitegrid", {'grid.color': '#e5e7eb', 'grid.alpha': 0.5})
    fig, ax = plt.subplots(figsize=(12, 10))

    # Remove density - user requested no density
    # if include_density and len(d) >= 10:
    #     try:
    #         sns.kdeplot(...)
    #     except Exception:
    #         pass

    # Enhanced scatter plot with better styling
    sns.scatterplot(
        data=d, x="pfx_x_inches", y="pfx_z_inches",
        hue="pitch_type", hue_order=usage_order,
        palette=pitch_palette, s=120,
        edgecolor="white", linewidth=1.5, alpha=0.75, ax=ax
    )

    # Enhanced grid lines
    ax.axhline(0, color="#d1d5db", linestyle="-", lw=1.5, alpha=0.6, zorder=0)
    ax.axvline(0, color="#d1d5db", linestyle="-", lw=1.5, alpha=0.6, zorder=0)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal", adjustable="box")
    
    # Enhanced labels
    ax.set_xlabel("Horizontal Break (inches)", fontsize=13, weight='600', color='#374151', labelpad=10)
    ax.set_ylabel("Induced Vertical Break (inches)", fontsize=13, weight='600', color='#374151', labelpad=10)
    
    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='#e5e7eb')
    ax.set_axisbelow(True)
    
    # Modernize axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d1d5db')
    ax.spines['bottom'].set_color('#d1d5db')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#4b5563')
    
    # Enhanced legend with full pitch names
    leg = ax.legend(title="Pitch Type", bbox_to_anchor=(1.02, 1), loc='upper left',
                    frameon=True, framealpha=1.0, edgecolor='#d1d5db', facecolor='white',
                    shadow=False, title_fontsize=12, fontsize=10)
    if leg:
        leg.get_frame().set_linewidth(1.5)
        leg.get_title().set_fontweight('600')
        # Update legend labels to show full pitch names instead of abbreviations
        for text in leg.get_texts():
            label = text.get_text()
            # Map abbreviation to full name if available, otherwise keep abbreviation
            full_name = PITCH_NAME_MAP.get(label, label)
            text.set_text(full_name)
    
    plt.tight_layout()

    out_path = Path(out_dir) / f"{_slug(pitcher_name)}_movement.png"
    try:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return str(out_path)
