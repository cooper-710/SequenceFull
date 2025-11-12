# src/build_bar_by_count.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from scrape_savant import fetch_pitcher_statcast

PITCH_COLOR_MAP = {
    'FF': '#FF0000', 'FT': '#8B0000', 'SI': '#FFA500', 'FC': '#808080', 'SL': '#0000FF',
    'CU': '#800080', 'SV': '#4682B4', 'ST': '#008080', 'CH': '#008000', 'FS': '#00FFFF',
    'KC': '#4B0082', 'KN': '#D3D3D3', 'EP': '#FFC0CB', 'FO': '#00008B', 'SC': '#FF00FF',
    'PO': '#A52A2A', 'IN': '#F0E68C', 'UN': '#B0C4DE', 'CS': '#800000'
}
PITCH_ORDER = list(PITCH_COLOR_MAP.keys())

COUNTS = [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(1,2),(2,2),(3,0),(3,1),(3,2),(0,2)]
COUNT_LABELS = [f"{b}-{s}" for (b,s) in COUNTS]

def _maybe_add_logo(ax, logo_path: Optional[str]) -> None:
    if not logo_path:
        return
    try:
        img = mpimg.imread(logo_path)
        imagebox = OffsetImage(img, zoom=0.07, interpolation='bilinear')
        ab = AnnotationBbox(imagebox, (0.05, 1.20), xycoords='axes fraction',
                            frameon=False, box_alignment=(0, 1))
        ax.add_artist(ab)
    except Exception:
        pass

def build_pitch_mix_by_count_for_pitcher(
    pitcher_id: int,
    pitcher_name: str,
    start_date: str,
    end_date: str,
    out_dir: str = "build/figures",
    logo_path: Optional[str] = None,
    stand: Optional[str] = None,   # 'R' or 'L' for batter handedness
) -> Optional[str]:
    df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
    if df is None or df.empty:
        return None

    if stand in ('R', 'L'):
        df = df[df['stand'] == stand]
    df = df.dropna(subset=['pitch_type', 'balls', 'strikes'])
    if df.empty:
        return None

    df = df.assign(count = df['balls'].astype(int).astype(str) + '-' + df['strikes'].astype(int).astype(str))
    df = df[df['count'].isin(COUNT_LABELS)]
    if df.empty:
        return None

    # Need at least 10 pitches to create a meaningful visualization
    if len(df) < 10:
        return None

    grp = (df.groupby(['count','pitch_type'])
             .size()
             .reset_index(name='pitch_count'))
    totals = grp.groupby('count')['pitch_count'].transform('sum')
    grp['percentage'] = (grp['pitch_count'] / totals) * 100.0

    grp['count'] = pd.Categorical(grp['count'], categories=COUNT_LABELS, ordered=True)
    grp = grp.sort_values(['count', 'pitch_type'])

    unique_pitches = grp['pitch_type'].unique().tolist()
    palette = {pt: PITCH_COLOR_MAP.get(pt, '#bfbfbf') for pt in unique_pitches}
    hue_order = [p for p in PITCH_ORDER if p in unique_pitches]

    # Modern styling
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#fafafa'
    plt.rcParams['axes.edgecolor'] = '#d1d5db'
    plt.rcParams['grid.color'] = '#e5e7eb'
    plt.rcParams['text.color'] = '#0a0a0a'
    plt.rcParams['xtick.color'] = '#4b5563'
    plt.rcParams['ytick.color'] = '#4b5563'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = '#d1d5db'
    plt.rcParams['font.family'] = 'Inter, system-ui, sans-serif'

    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.set_style("whitegrid", {'grid.color': '#e5e7eb', 'grid.alpha': 0.5})

    # Create stacked bar chart with modern styling
    bars = sns.barplot(
        data=grp,
        x='count',
        y='percentage',
        hue='pitch_type',
        palette=palette,
        hue_order=hue_order,
        ax=ax,
        edgecolor='white',
        linewidth=1.5,
        width=0.85
    )

    # Add subtle dividers between counts
    xticks = ax.get_xticks()
    for i in range(len(xticks) - 1):
        ax.axvline((xticks[i] + xticks[i+1]) / 2, color='#e5e7eb', linewidth=1, linestyle='-', zorder=0, alpha=0.6)

    _maybe_add_logo(ax, logo_path)

    ax.set_ylabel("Pitch Usage (%)", fontsize=13, weight='600', color='#374151', labelpad=10)
    ax.set_xlabel("Count", fontsize=13, weight='600', color='#374151', labelpad=10)
    ax.set_ylim(0, 100)
    
    # Enhanced grid
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Modernize axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d1d5db')
    ax.spines['bottom'].set_color('#d1d5db')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#4b5563')

    # Enhanced legend
    leg = ax.legend(
        title="Pitch Type",
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True,
        framealpha=1.0,
        edgecolor='#d1d5db',
        facecolor='white',
        shadow=False
    )
    if leg:
        leg.get_frame().set_linewidth(1.5)
        leg.get_title().set_fontweight('600')

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    suffix = f"_vs_{'RHH' if stand=='R' else ('LHH' if stand=='L' else 'ALL')}"
    out = Path(out_dir) / f"{pitcher_name.replace(' ','_')}_mix_by_count{suffix}.png"
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return str(out)
