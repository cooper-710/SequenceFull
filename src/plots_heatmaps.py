from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

PITCH_NAME_MAP = {
    "FF":"4-Seam Fastball","SI":"Sinker","SL":"Slider","CH":"Changeup","CU":"Curveball",
    "KC":"Knuckle Curve","FC":"Cutter","FS":"Splitter","FT":"2-Seam Fastball",
    "EP":"Eephus","ST":"Sweeper","SV":"Slurve","CS":"Slow Curve","KN":"Knuckleball","FO":"Forkball","SC":"Screwball"
}

COUNTS = [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1),(1,2),(2,2),(3,2),(3,1),(3,0),(0,2)]

def _draw_zone(ax):
    ax.set_xlim(-2,2); ax.set_ylim(0,5); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    w,h=0.57,0.67; xs,ys=-0.855,3.5
    for r in range(3):
        for c in range(3):
            ax.add_patch(patches.Rectangle((xs+c*w, ys-(r+1)*h), w, h, lw=1.4, ec='#1a1a1a', fc='none'))
    plate=[[-0.7085,0.7085],[0.7085,0.7085],[0.35425,0.2085],[0,0],[-0.35425,0.2085]]
    ax.add_patch(patches.Polygon(plate, closed=True, ec='black', fc='white', lw=2.0, zorder=2))

def render_heatmaps_for_pitcher(df_pitcher: pd.DataFrame, out_dir: str) -> str | None:
    d = df_pitcher.copy()
    d = d[d['plate_x'].notna() & d['plate_z'].notna() & d['balls'].notna() & d['strikes'].notna()]
    if d.empty: return None

    usage = d['pitch_type'].value_counts(normalize=True).sort_values(ascending=False)
    pitches = usage.index.tolist()
    if not pitches: return None

    nrows = len(pitches); ncols = len(COUNTS) + 1  # +1 for label gutter
    fig_w = 0.95 + len(COUNTS)*1.30
    fig_h = nrows * 1.55
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h),
                             gridspec_kw={'width_ratios':[1.15]+[1]*len(COUNTS)}, sharex=False, sharey=False)
    if nrows==1: axes = [axes]
    sns.set(style="white")
    
    # Create custom blue-green-red colormap
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']  # Blue -> Green -> Yellow -> Red
    n_bins = 256
    heatmap_cmap = LinearSegmentedColormap.from_list('blue_green_red', colors, N=n_bins)

    for r, p in enumerate(pitches):
        df_p = d[d['pitch_type']==p]
        lbl_ax = axes[r][0]
        lbl_ax.axis('off')
        full = PITCH_NAME_MAP.get(p, p)
        lbl_ax.text(0.98, 0.5, f"{full} — {usage[p]*100:.1f}%", ha='right', va='center',
                    fontsize=10.5, weight='bold')

        for c,(b,s) in enumerate(COUNTS, start=1):
            ax = axes[r][c]
            subset = df_p[(df_p['balls']==b) & (df_p['strikes']==s)]
            if not subset.empty:
                try:
                    sns.kdeplot(data=subset, x='plate_x', y='plate_z', fill=True,
                                cmap=heatmap_cmap, bw_adjust=0.5, thresh=0.05, levels=100, ax=ax)
                except Exception:
                    pass
                # No scatter overlay - just smooth density heatmap
            _draw_zone(ax)
            if r==0:
                ax.set_title(f"{b}-{s}", fontsize=10.5, weight='bold')

    pitcher = d['player_name'].iloc[0]
    fig.suptitle(f"{pitcher} — Pitch Type and Location by Count", fontsize=14, weight='bold', y=0.995)
    plt.tight_layout(rect=[0.02,0.01,0.995,0.985])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{pitcher.replace(' ','_')}_heatmaps.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.08)
    plt.close(fig)
    return str(path)
