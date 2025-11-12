# src/plots_pitch_table.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmarks import benchmarks, pitch_name_map

# ---------- helpers ----------
def add_pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    desc = d['description'].astype(str).str.lower()
    d['is_strike'] = desc.isin([
        'called_strike','foul','foul_tip','swinging_strike','swinging_strike_blocked','foul_bunt','hit_into_play'
    ])
    d['is_swing'] = desc.isin([
        'foul','foul_tip','swinging_strike','swinging_strike_blocked','missed_bunt','foul_bunt','hit_into_play'
    ])
    d['is_whiff'] = desc.isin(['swinging_strike','swinging_strike_blocked','missed_bunt'])
    # finish flags
    d['next_ab'] = d['at_bat_number'].shift(-1)
    d['next_pitcher'] = d['player_name'].shift(-1)
    d['is_final_pitch'] = (d['at_bat_number'] != d['next_ab']) | (d['player_name'] != d['next_pitcher'])
    d['is_k'] = d['is_final_pitch'] & desc.isin(['swinging_strike','called_strike','foul_tip','swinging_strike_blocked','missed_bunt'])
    d['is_bb'] = d['is_final_pitch'] & desc.isin(['ball','ball_in_dirt'])
    return d

def _wrap_label(s: str, limit: int = 16) -> str:
    s = (s or "").strip()
    if len(s) <= limit or " " not in s:
        return s
    cut = s.rfind(" ", 0, limit)
    if cut == -1:
        cut = s.find(" ", limit)
    return s if cut == -1 else s[:cut] + "\n" + s[cut+1:]

def _format_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ['Usage','MaxVelo','AvgVelo','IVB','HB','ReleaseH','ReleaseS',
            'Strike %','Swing %','Whiff %','% of Strikeouts','% of Walks']
    formatted = summary.copy()
    formatted['pitch_type'] = formatted['pitch_type'].map(pitch_name_map).fillna(formatted['pitch_type'])
    formatted['pitch_type'] = formatted['pitch_type'].map(lambda s: _wrap_label(str(s), limit=16))
    # Format velocities to 1 decimal place
    formatted['MaxVelo'] = formatted['MaxVelo'].round(1).astype(str)
    formatted['AvgVelo'] = formatted['AvgVelo'].round(1).astype(str)
    # Format movement and release to 2 decimal places
    formatted['IVB'] = (formatted['IVB']).round(2).astype(str) + ' in'
    formatted['HB'] = (formatted['HB']).round(2).astype(str) + ' in'
    formatted['ReleaseH'] = (formatted['ReleaseH']).round(2).astype(str) + ' ft'
    formatted['ReleaseS'] = (formatted['ReleaseS']).round(2).astype(str) + ' ft'
    # Format percentages to 1 decimal place
    for c in ['Usage','Strike %','Swing %','% of Strikeouts','% of Walks']:
        formatted[c] = formatted[c].round(1).astype(str) + '%'
    formatted['Whiff %'] = summary['Whiff %'].apply(lambda x: '--' if pd.isna(x) else f"{round(x,1)}%")
    return formatted[['pitch_type'] + cols]

def _short_headers() -> list[str]:
    return ['Pitch Type','Usage','Max V','Avg V','IVB','HB','Rel H','Rel S','Strike%','Swing%','Whiff%','% Ks','% BB']

def _benchmark_cell_color(pitch: str, metric: str, val: float, throws: str, hand_key: str) -> str:
    if pitch not in benchmarks.get(hand_key, {}): return '#ffffff'
    if metric not in benchmarks[hand_key][pitch]: return '#ffffff'
    ref = benchmarks[hand_key][pitch][metric]

    if metric == 'IVB':
        low_ivb = {'SI','CH','CU','KC','SV','FS'}
        high_ivb = {'FF','FT','FC'}
        if pitch in low_ivb:   return '#d4f7d4' if val <= ref else '#f7d4d4'
        if pitch in high_ivb:  return '#d4f7d4' if val >= ref else '#f7d4d4'
        return '#ffffff'

    if metric == 'HB':
        run_types   = {'FF','SI','FT','CH','FS'}
        sweep_types = {'SL','ST','SV','FC','CU'}
        if pitch in run_types:
            return '#d4f7d4' if (val >= ref if throws=='R' else val <= ref) else '#f7d4d4'
        if pitch in sweep_types:
            return '#d4f7d4' if (val <= ref if throws=='R' else val >= ref) else '#f7d4d4'
        return '#ffffff'

    if metric == '% of Walks':
        return '#d4f7d4' if val <= ref else '#f7d4d4'

    if metric in {'ReleaseH','ReleaseS'}:
        return '#d4f7d4' if abs(val - ref) <= 0.15 else '#f7d4d4'

    return '#d4f7d4' if val >= ref else '#f7d4d4'

def _summarize(df_p: pd.DataFrame) -> pd.DataFrame:
    grp = df_p.groupby('pitch_type', as_index=False).agg(
        PitchCount=('pitch_type','count'),
        MaxVelo=('release_speed','max'),
        AvgVelo=('release_speed','mean'),
        IVB=('pfx_z', lambda x: x.mean()*12),
        HB=('pfx_x', lambda x: x.mean()*-12),
        ReleaseH=('release_pos_z','mean'),
        ReleaseS=('release_pos_x','mean'),
        Strikes=('is_strike','sum'),
        Swings=('is_swing','sum'),
        Whiffs=('is_whiff','sum'),
        K_finishes=('is_k','sum'),
        BB_finishes=('is_bb','sum')
    )
    total_K = df_p['is_k'].sum()
    total_BB = df_p['is_bb'].sum()
    total_pitches = grp['PitchCount'].sum()

    grp['Usage'] = 100 * grp['PitchCount'] / max(total_pitches, 1)
    grp['Strike %'] = 100 * grp['Strikes'] / grp['PitchCount']
    grp['Swing %'] = 100 * grp['Swings'] / grp['PitchCount']
    grp['Whiff %'] = 100 * grp['Whiffs'] / grp['Swings'].replace(0, np.nan)
    grp['% of Strikeouts'] = 100 * grp['K_finishes'] / total_K if total_K > 0 else 0.0
    grp['% of Walks'] = 100 * grp['BB_finishes'] / total_BB if total_BB > 0 else 0.0
    grp['ReleaseS'] = -grp['ReleaseS']
    cols = ['Usage','MaxVelo','AvgVelo','IVB','HB','ReleaseH','ReleaseS','Strike %','Swing %','Whiff %','% of Strikeouts','% of Walks']
    grp[cols] = grp[cols].astype(float).round(3)
    return grp.sort_values('Usage', ascending=False).reset_index(drop=True)

# ---------- main render ----------
def render_pitch_table_png(df_pitcher_all: pd.DataFrame, out_dir: str) -> dict:
    """
    Creates two PNGs: vs RHH and vs LHH. Returns {"RHH": path_or_None, "LHH": path_or_None}
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if df_pitcher_all.empty:
        return {"RHH": None, "LHH": None}

    df = add_pitch_flags(df_pitcher_all)
    pitcher = str(df['player_name'].iloc[0])
    throws = str(df['p_throws'].iloc[0])

    outputs = {}
    for stand, label in (('R','RHH'),('L','LHH')):
        d = df[df['stand']==stand].copy()
        if d.empty:
            outputs[label] = None
            continue

        summary = _summarize(d)
        hand_key = 'RHP' if throws=='R' else 'LHP'
        cols = ['Usage','MaxVelo','AvgVelo','IVB','HB','ReleaseH','ReleaseS',
                'Strike %','Swing %','Whiff %','% of Strikeouts','% of Walks']

        n_rows = len(summary)

        # ---- Larger, better sizing for improved readability ----
        # Increased base width and height for better visibility
        fig_w = 14.0
        if n_rows >= 8:
            fig_h = 1.8 + 0.50 * n_rows
            font_sz = 10.0
            y_scale = 1.25
        elif n_rows >= 7:
            fig_h = 1.9 + 0.52 * n_rows
            font_sz = 10.2
            y_scale = 1.28
        elif n_rows >= 6:
            fig_h = 2.0 + 0.55 * n_rows
            font_sz = 10.5
            y_scale = 1.32
        else:
            fig_h = max(2.5, 2.1 + 0.60 * max(1, n_rows))
            font_sz = 11.0
            y_scale = 1.38

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')
        
        # Set background color to white to match Sequence.png aesthetic
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Optimized column widths - balanced distribution for better space usage
        ncols = 1 + len(cols)
        # Pitch type column: 18%, remaining columns share 82% evenly
        # This gives more space to data columns while keeping pitch names readable
        colWidths = [0.18] + [0.82 / (ncols - 1)] * (ncols - 1)

        cell_df_fmt = _format_table(summary)
        tbl = ax.table(
            cellText=cell_df_fmt.values,
            colLabels=_short_headers(),
            cellLoc='center',
            loc='upper left',
            colWidths=colWidths,
        )

        # Styling
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(font_sz)
        tbl.scale(1.0, y_scale)

        # Modern header styling with black to match logo
        header_color = '#0a0a0a'  # Black to match logo
        for j in range(ncols):
            cell = tbl[0, j]
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', weight='bold', size=font_sz * 1.05)
            cell.set_height(0.10)  # Taller header for better presence
            cell.set_text_props(ha='center', va='center')

        # Alternating row colors - subtle for clean Sequence.png aesthetic
        row_colors = ['#ffffff', '#f9fafb']  # White and very subtle grey
        for i in range(1, n_rows + 1):
            row_color = row_colors[i % 2]
            # Left column with subtle styling
            c0 = tbl[i, 0]
            c0.set_facecolor('#f3f4f6')  # Slightly darker grey for pitch type column for better contrast
            c0.get_text().set_ha('left')
            c0.get_text().set_x(0.04)  # More padding from left edge
            c0.get_text().set_weight('700')  # Bolder pitch names
            c0.get_text().set_size(font_sz * 1.08)  # Slightly larger
            c0.set_height(0.12)  # Taller rows for better readability
            
            # Data cells with alternating colors
            for j in range(1, ncols):
                cell = tbl[i, j]
                cell.set_facecolor(row_color)
                cell.get_text().set_weight('600')  # Slightly bolder for better readability
                cell.get_text().set_size(font_sz * 1.02)  # Slightly larger data text
                cell.set_height(0.12)  # Match row height

        # Conditional cell colors with more subtle, modern palette
        for r in range(1, n_rows + 1):
            pitch = summary.iloc[r-1]['pitch_type']
            row_color = row_colors[r % 2]
            for ci, metric in enumerate(cols, start=1):
                try:
                    val = float(summary.iloc[r-1][metric])
                except Exception:
                    continue
                benchmark_color = _benchmark_cell_color(pitch, metric, val, throws, hand_key)
                # Very subtle benchmark indicators - even more minimal
                cell = tbl[(r, ci)]
                if benchmark_color == '#d4f7d4':  # Good (green)
                    # Extremely subtle green tint - barely visible
                    cell.set_facecolor('#f7fdf7')  # Almost white with very slight hint of green
                elif benchmark_color == '#f7d4d4':  # Poor (red)
                    # Extremely subtle red tint - barely visible
                    cell.set_facecolor('#fffafa')  # Almost white with very slight hint of red
                else:
                    # Keep row color
                    cell.set_facecolor(row_color)

        # Enhanced borders with better visual hierarchy - cleaner, more modern look
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                # Header cells - no internal borders, only outer border
                cell.set_linewidth(0)
                # Outer edges of header - thicker for emphasis
                if c == 0:
                    cell.set_linewidth(3.0)  # Left edge - thicker
                    cell.set_edgecolor(header_color)
                elif c == ncols - 1:
                    cell.set_linewidth(3.0)  # Right edge - thicker
                    cell.set_edgecolor(header_color)
                # Bottom border of header - thicker separator
                cell.set_linewidth(3.0)
                cell.set_edgecolor(header_color)
            elif c == 0:
                # Left column (pitch type) - clean borders
                cell.set_linewidth(2.0)  # Thicker left border for definition
                cell.set_edgecolor('#d1d5db')
                if r == 1:
                    # Top border (below header)
                    cell.set_linewidth(2.0)
                    cell.set_edgecolor('#d1d5db')
                elif r == n_rows:
                    # Bottom border - same as other rows
                    cell.set_linewidth(2.0)
                    cell.set_edgecolor('#d1d5db')
            else:
                # Data cells - minimal borders for clean look
                if r == 1:
                    # Top border (below header) - subtle
                    cell.set_linewidth(1.5)
                    cell.set_edgecolor('#e5e7eb')
                elif r == n_rows:
                    # Bottom border - same as other rows
                    cell.set_linewidth(0.3)
                    cell.set_edgecolor('#f3f4f6')
                elif c == ncols - 1:
                    # Right edge - subtle
                    cell.set_linewidth(1.5)
                    cell.set_edgecolor('#e5e7eb')
                else:
                    # Internal borders - very minimal, almost invisible
                    cell.set_linewidth(0.3)
                    cell.set_edgecolor('#f3f4f6')

        # NOTE: we do NOT add a Matplotlib title line — the HTML heading already covers it.
        # That saves ~0.25–0.35 in vertical space.

        out_path = Path(out_dir) / f"{pitcher.replace(' ','_')}_vs_{label}.png"
        # Higher DPI and optimized padding for quality and space efficiency
        # White background to match Sequence.png aesthetic
        # Increased padding for better visual breathing room
        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.12)
        plt.close(fig)
        outputs[label] = str(out_path)

    return outputs

def render_pitch_table_combined_png(df_pitcher_all: pd.DataFrame, out_dir: str) -> Optional[str]:
    """
    Creates a single PNG combining all pitches (not split by batter handedness).
    Returns path or None.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if df_pitcher_all.empty:
        return None

    df = add_pitch_flags(df_pitcher_all)
    pitcher = str(df['player_name'].iloc[0])
    throws = str(df['p_throws'].iloc[0])

    # Use all data without filtering by batter handedness
    summary = _summarize(df)
    hand_key = 'RHP' if throws=='R' else 'LHP'
    cols = ['Usage','MaxVelo','AvgVelo','IVB','HB','ReleaseH','ReleaseS',
            'Strike %','Swing %','Whiff %','% of Strikeouts','% of Walks']

    n_rows = len(summary)

    # Slightly more compact sizing for combined table
    fig_w = 14.0
    if n_rows >= 8:
        fig_h = 1.7 + 0.48 * n_rows
        font_sz = 9.8
        y_scale = 1.22
    elif n_rows >= 7:
        fig_h = 1.8 + 0.50 * n_rows
        font_sz = 10.0
        y_scale = 1.25
    elif n_rows >= 6:
        fig_h = 1.9 + 0.52 * n_rows
        font_sz = 10.2
        y_scale = 1.28
    else:
        fig_h = max(2.4, 2.0 + 0.58 * max(1, n_rows))
        font_sz = 10.8
        y_scale = 1.35

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    
    # Set background color to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Column widths
    ncols = 1 + len(cols)
    colWidths = [0.18] + [0.82 / (ncols - 1)] * (ncols - 1)

    cell_df_fmt = _format_table(summary)
    tbl = ax.table(
        cellText=cell_df_fmt.values,
        colLabels=_short_headers(),
        cellLoc='center',
        loc='upper left',
        colWidths=colWidths,
    )

    # Styling
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_sz)
    tbl.scale(1.0, y_scale)

    # Header styling
    header_color = '#0a0a0a'
    for j in range(ncols):
        cell = tbl[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', weight='bold', size=font_sz * 1.05)
        cell.set_height(0.10)
        cell.set_text_props(ha='center', va='center')

    # Alternating row colors
    row_colors = ['#ffffff', '#f9fafb']
    for i in range(1, n_rows + 1):
        row_color = row_colors[i % 2]
        # Left column
        c0 = tbl[i, 0]
        c0.set_facecolor('#f3f4f6')
        c0.get_text().set_ha('left')
        c0.get_text().set_x(0.04)
        c0.get_text().set_weight('700')
        c0.get_text().set_size(font_sz * 1.08)
        c0.set_height(0.12)
        
        # Data cells
        for j in range(1, ncols):
            cell = tbl[i, j]
            cell.set_facecolor(row_color)
            cell.get_text().set_weight('600')
            cell.get_text().set_size(font_sz * 1.02)
            cell.set_height(0.12)

    # Conditional cell colors
    for r in range(1, n_rows + 1):
        pitch = summary.iloc[r-1]['pitch_type']
        row_color = row_colors[r % 2]
        for ci, metric in enumerate(cols, start=1):
            try:
                val = float(summary.iloc[r-1][metric])
            except Exception:
                continue
            benchmark_color = _benchmark_cell_color(pitch, metric, val, throws, hand_key)
            cell = tbl[(r, ci)]
            if benchmark_color == '#d4f7d4':
                cell.set_facecolor('#f7fdf7')
            elif benchmark_color == '#f7d4d4':
                cell.set_facecolor('#fffafa')
            else:
                cell.set_facecolor(row_color)

    # Borders
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_linewidth(0)
            if c == 0:
                cell.set_linewidth(3.0)
                cell.set_edgecolor(header_color)
            elif c == ncols - 1:
                cell.set_linewidth(3.0)
                cell.set_edgecolor(header_color)
            cell.set_linewidth(3.0)
            cell.set_edgecolor(header_color)
        elif c == 0:
            cell.set_linewidth(2.0)
            cell.set_edgecolor('#d1d5db')
            if r == 1:
                cell.set_linewidth(2.0)
                cell.set_edgecolor('#d1d5db')
            elif r == n_rows:
                cell.set_linewidth(2.0)
                cell.set_edgecolor('#d1d5db')
        else:
            if r == 1:
                cell.set_linewidth(1.5)
                cell.set_edgecolor('#e5e7eb')
            elif r == n_rows:
                cell.set_linewidth(0.3)
                cell.set_edgecolor('#f3f4f6')
            elif c == ncols - 1:
                cell.set_linewidth(1.5)
                cell.set_edgecolor('#e5e7eb')
            else:
                cell.set_linewidth(0.3)
                cell.set_edgecolor('#f3f4f6')

    out_path = Path(out_dir) / f"{pitcher.replace(' ','_')}_combined.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    plt.close(fig)
    return str(out_path)
