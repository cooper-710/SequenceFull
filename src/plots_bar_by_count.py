from __future__ import annotations
from typing import Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

COUNT_ORDER = ['0-0','1-0','0-1','1-1','2-0','2-1','1-2','2-2','3-0','3-1','3-2','0-2']

PITCH_COLORS = {
    'FF':'#FF0000','FT':'#8B0000','SI':'#FFA500','FC':'#808080','SL':'#0000FF',
    'CU':'#800080','SV':'#4682B4','ST':'#008080','CH':'#008000','FS':'#00FFFF',
    'KC':'#4B0082','KN':'#D3D3D3','EP':'#FFC0CB','FO':'#00008B','SC':'#FF00FF',
    'PO':'#A52A2A','IN':'#F0E68C','UN':'#B0C4DE','CS':'#800000'
}
PITCH_ORDER = list(PITCH_COLORS.keys())

def _maybe_logo_artist(logo_path: Optional[str]):
    if not logo_path: return None
    try:
        img = mpimg.imread(logo_path)
        # small, crisp logo
        imagebox = OffsetImage(img, zoom=0.07, interpolation='bilinear')
        return AnnotationBbox(imagebox, (0.04, 1.18), xycoords='axes fraction',
                              frameon=False, box_alignment=(0, 1))
    except Exception:
        return None

def render_pitch_mix_by_count_png(
    df_pitcher: pd.DataFrame,
    pitcher_name: str,
    out_dir: str = "build/figures",
    logo_path: Optional[str] = None
) -> Optional[str]:
    """
    df_pitcher: statcast pitcher dataframe for ONE pitcher (any date window),
                needs columns: pitch_type, balls, strikes, player_name
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d = df_pitcher.copy()
    d = d[d['pitch_type'].notna() & d['balls'].notna() & d['strikes'].notna()]
    if d.empty:
        return None

    d['count'] = d['balls'].astype(int).astype(str) + '-' + d['strikes'].astype(int).astype(str)

    mix = (d.groupby(['count','pitch_type']).size()
             .reset_index(name='pitch_count'))
    mix['percentage'] = mix.groupby('count')['pitch_count'].transform(lambda x: x / x.sum() * 100.0)

    # enforce count order
    mix['count'] = pd.Categorical(mix['count'], categories=COUNT_ORDER, ordered=True)
    mix = mix.sort_values(['count','pitch_type'])

    unique_pitches = mix['pitch_type'].unique()
    palette = {p: PITCH_COLORS.get(p, '#bfbfbf') for p in unique_pitches}
    hue_order = [p for p in PITCH_ORDER if p in unique_pitches]

    # style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.edgecolor': 'black',
        'grid.color': '#dddddd',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.facecolor': 'white',
        'legend.edgecolor': 'black'
    })
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('white')

    sns.barplot(
        data=mix, x='count', y='percentage',
        hue='pitch_type', palette=palette, hue_order=hue_order, ax=ax
    )

    # subtle vertical separators between counts
    xticks = ax.get_xticks()
    for i in range(len(xticks)-1):
        ax.axvline((xticks[i] + xticks[i+1]) / 2, color='#e0e0e0', linewidth=0.8, linestyle='--', zorder=0)

    # optional logo
    logo_artist = _maybe_logo_artist(logo_path)
    if logo_artist:
        ax.add_artist(logo_artist)

    ax.set_ylabel("Pitch Usage (%)", fontsize=12, weight='bold', color='black')
    ax.set_xlabel("Count", fontsize=12, weight='bold', color='black')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.legend(
        title="Pitch Type", title_fontsize=12, fontsize=10,
        bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0
    )

    plt.tight_layout()
    out_path = Path(out_dir) / f"{pitcher_name.replace(' ','_')}_mix_by_count.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(out_path)
