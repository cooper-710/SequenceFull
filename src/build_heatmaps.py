from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from scrape_savant import fetch_pitcher_statcast

COUNTS = [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(1,2),(2,2),(3,0),(3,1),(3,2),(0,2)]

PITCH_LABELS = {
    "FF":"Four-Seam Fastball","FT":"Two-Seam Fastball","SI":"Sinker","FC":"Cutter","SL":"Slider","ST":"Sweeper",
    "CU":"Curveball","KC":"Knuckle Curve","CS":"Slow Curve","CH":"Changeup","FS":"Splitter","KN":"Knuckleball",
    "SV":"Slurve","FO":"Forkball","EP":"Eephus","SC":"Screwball","PO":"Pitch Out","IN":"Intentional Ball","UN":"Unknown",
}

def _draw_zone(ax):
    ax.set_xlim(-2, 2); ax.set_ylim(0, 5); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    # Enhanced strike zone with better styling
    w,h=0.57,0.67; xs,ys=-0.855,3.5
    for r in range(3):
        for c in range(3):
            ax.add_patch(patches.Rectangle(
                (xs+c*w, ys-(r+1)*h), w, h, 
                linewidth=1.5, 
                edgecolor='#4b5563', 
                facecolor='none',
                alpha=0.6,
                linestyle='-'
            ))
    # Enhanced home plate
    plate=[[-0.7085,0.7085],[0.7085,0.7085],[0.35425,0.2085],[0,0],[-0.35425,0.2085]]
    ax.add_patch(patches.Polygon(
        plate, closed=True, 
        edgecolor='#1a1a1a', 
        facecolor='white', 
        linewidth=2.0, 
        zorder=2,
        alpha=0.9
    ))

def _labels_from(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty or 'pitch_type' not in df.columns: return []
    order = df['pitch_type'].dropna().astype(str).value_counts().index.tolist()
    return [PITCH_LABELS.get(k, k) for k in order]

def build_heatmaps_for_pitcher(
    pitcher_id: int,
    pitcher_name: str,
    start_date: str,
    end_date: str,
    out_dir: str = "build/figures",
    stand: Optional[str] = None,  # 'R' or 'L' to filter by batter handedness
) -> Optional[str]:
    d = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
    if d is None or d.empty: return None
    d = d[d['plate_x'].notna() & d['plate_z'].notna() & d['balls'].notna() & d['strikes'].notna() & d['pitch_type'].notna()]
    if stand in ('R','L'):
        d = d[d['stand'] == stand]
    if d.empty: return None
    
    # Need at least 10 pitches to create a meaningful heatmap
    if len(d) < 10: return None

    ptypes = d['pitch_type'].value_counts(normalize=True).sort_values(ascending=False).index.tolist()

    # Compact per-cell size; overall pixel width capped via dpi calculation
    cell_w, cell_h = 1.45, 1.35
    nrows, ncols = len(ptypes), len(COUNTS)
    fig_w, fig_h = ncols*cell_w, nrows*cell_h
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    if nrows==1: axes=[axes]
    if ncols==1: axes=[[ax] for ax in axes]

    # Modern professional styling for heatmaps
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'Inter, system-ui, sans-serif'
    
    # Create custom blue-green-red colormap
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']  # Blue -> Green -> Yellow -> Red
    n_bins = 256
    heatmap_cmap = LinearSegmentedColormap.from_list('blue_green_red', colors, N=n_bins)

    for r, pt in enumerate(ptypes):
        dfp = d[d['pitch_type']==pt]
        for c,(b,s) in enumerate(COUNTS):
            ax = axes[r][c]
            sub = dfp[(dfp['balls']==b) & (dfp['strikes']==s)]
            if not sub.empty:
                try:
                    # Enhanced KDE plot with professional styling
                    sns.kdeplot(
                        data=sub, 
                        x='plate_x', 
                        y='plate_z', 
                        fill=True, 
                        cmap=heatmap_cmap, 
                        bw_adjust=0.7, 
                        thresh=0.02, 
                        levels=120, 
                        ax=ax,
                        alpha=0.9,
                        antialiased=True
                    )
                except Exception:
                    pass
                # No scatter overlay - just smooth density heatmap
            _draw_zone(ax)
            ax.set_xlabel(''); ax.set_ylabel('')
            # Enhanced count labels
            if r==0: 
                ax.set_title(f"{b}-{s}", fontsize=11, weight='bold', color='#0a0a0a', pad=5)

        # Enhanced left-side row label
        left_ax = axes[r][0]
        label = PITCH_LABELS.get(pt, pt)
        left_ax.text(-0.12, 0.50, label, transform=left_ax.transAxes, ha='right', va='center', 
                     fontsize=12, fontweight='bold', color='#0a0a0a')

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    suffix = f"_vs_{'RHH' if stand=='R' else ('LHH' if stand=='L' else 'ALL')}"
    out = Path(out_dir) / f"{pitcher_name.replace(' ','_')}_heatmaps{suffix}.png"

    # Save with width ~1600px for crisp but light output
    target_px = 1600.0
    dpi_save = int(max(90, min(170, target_px / max(fig_w, 1.0))))

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.07, wspace=0.08, hspace=0.08)
    fig.savefig(out, dpi=dpi_save, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(out)
