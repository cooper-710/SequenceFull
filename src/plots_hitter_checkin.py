# src/plots_hitter_checkin.py
from __future__ import annotations
from typing import Optional
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- small color helper (your logic) ----
def _hex_colormap(series: pd.Series, green_high: bool) -> List[str]:
    s = series.copy()
    mask = s.notna()
    if mask.sum() == 0:
        return ['#ffffff'] * len(s)
    v = s[mask].to_numpy(dtype=float)
    v = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-5)
    if not green_high:
        v = 1.0 - v
    reds = np.array([247, 212, 212]); greens = np.array([212, 247, 212])
    blended = (greens * v[:, None] + reds * (1 - v[:, None])).astype(int)
    hexs = ['#{:02x}{:02x}{:02x}'.format(*rgb) for rgb in blended]
    result = ['#ffffff'] * len(s)
    j = 0
    for i, ok in enumerate(mask.to_list()):
        if ok:
            result[i] = hexs[j]; j += 1
    return result

def _summarize(df_hand: pd.DataFrame) -> pd.DataFrame:
    # Seen / swings / whiffs
    desc = df_hand['description'].astype(str).str.lower()
    is_whiff = desc.isin(['swinging_strike','swinging_strike_blocked'])
    is_swing = desc.isin(['swinging_strike','swinging_strike_blocked','foul','foul_tip','hit_into_play'])

    pitch_summary = df_hand.groupby('pitch_type', as_index=False).agg(
        Seen=('pitch_type','count'),
        Swings=('pitch_type', lambda x: int(is_swing.loc[x.index].sum())),
        Whiffs=('pitch_type', lambda x: int(is_whiff.loc[x.index].sum())),
    )
    pitch_summary['Whiff %'] = 100.0 * pitch_summary['Whiffs'] / pitch_summary['Swings'].replace(0, np.nan)

    # EV/LA/HardHit
    if {'launch_speed','launch_angle'}.issubset(df_hand.columns):
        batted = df_hand.loc[
            df_hand['launch_speed'].notna()
            & desc.str.contains('hit_into_play')
            & ~desc.str.contains('bunt')
        ]
        ev = batted.groupby('pitch_type', as_index=False).agg(
            Avg_EV=('launch_speed','mean'),
            Avg_LA=('launch_angle','mean'),
            Hard_Hit_pct=('launch_speed', lambda x: float((x > 95).mean()*100.0))
        )
    else:
        ev = pd.DataFrame(columns=['pitch_type','Avg_EV','Avg_LA','Hard_Hit_pct'])

    # Join & clean
    df = pitch_summary.merge(ev, on='pitch_type', how='left')
    df = df[['pitch_type','Seen','Avg_EV','Avg_LA','Hard_Hit_pct','Whiff %']].copy()
    df['Avg_EV'] = df['Avg_EV'].round(1)
    df['Avg_LA'] = df['Avg_LA'].round(1)
    df['Hard_Hit_pct'] = df['Hard_Hit_pct'].round(1)
    df['Whiff %'] = df['Whiff %'].round(1)
    df = df.sort_values('Seen', ascending=False).reset_index(drop=True)
    df = df[df['Seen'] >= 5]  # your threshold
    df.rename(columns={'pitch_type':'Pitch Type','# Seen':'Seen'}, inplace=True)
    return df

def _format_for_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    out = df.copy()
    out.columns = ['Pitch Type','# Seen','Avg EV','Avg LA','Hard Hit %','Whiff %']
    # build color arrays
    ev_colors    = _hex_colormap(out['Avg EV'], green_high=True)
    hh_colors    = _hex_colormap(out['Hard Hit %'], green_high=True)
    whiff_colors = _hex_colormap(out['Whiff %'], green_high=False)
    colors = {'Avg EV': ev_colors, 'Hard Hit %': hh_colors, 'Whiff %': whiff_colors}
    # add units for display
    out['Avg EV'] = out['Avg EV'].apply(lambda x: f"{x} MPH" if pd.notna(x) else '--')
    out['Avg LA'] = out['Avg LA'].apply(lambda x: f"{x}°" if pd.notna(x) else '--')
    out['Hard Hit %'] = out['Hard Hit %'].apply(lambda x: f"{x}%" if pd.notna(x) else '--')
    out['Whiff %'] = out['Whiff %'].apply(lambda x: f"{x}%" if pd.notna(x) else '--')
    return out, colors

def _draw_table(table_df: pd.DataFrame, colors: dict, title: str, out_path: Path) -> Optional[str]:
    if table_df.empty:
        return None
    fig_h = max(2.0, 0.6 * len(table_df) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis('off')
    tbl = ax.table(cellText=table_df.values,
                   colLabels=table_df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)

    for (r,c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#0b2545')
            cell.set_text_props(weight='bold', color='white')
        elif c == 0:
            cell.set_facecolor('#eaecef')
        elif r > 0:
            colname = table_df.columns[c]
            if colname in colors:
                cell.set_facecolor(colors[colname][r-1])

    ax.set_title(title, fontsize=14, weight='bold', pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(out_path)

def calculate_overall_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate overall performance metrics from statcast data."""
    metrics = {}
    
    if df.empty:
        return metrics
    
    desc = df['description'].astype(str).str.lower()
    
    # Plate appearances and outcomes - need to get unique at-bats
    if 'events' in df.columns and 'at_bat_number' in df.columns:
        events = df['events'].astype(str).str.lower()
        
        # Get unique at-bats by grouping by at_bat_number and game_pk
        # Only count the final pitch of each at-bat (where event occurred)
        if 'game_pk' in df.columns:
            # Group by game and at-bat, take the last row (final pitch with event)
            df_with_ab = df[events.notna() & (events != 'nan') & (events != '') & (events != 'none')].copy()
            if not df_with_ab.empty:
                # Get the last pitch of each at-bat (where event occurred)
                pa_ending = df_with_ab.groupby(['game_pk', 'at_bat_number']).last().reset_index()
        else:
            # Fallback: just get rows with events
            pa_ending = df[events.notna() & (events != 'nan') & (events != '') & (events != 'none')].copy()
        
        if not pa_ending.empty:
            events_series = pa_ending['events'].astype(str).str.lower()
            
            # Exclude non-PA events
            excluded_events = [
                'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'sac_bunt_double_play',
                'catcher_interf', 'caught_stealing_2b', 'caught_stealing_3b',
                'caught_stealing_home', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b',
                'other_out'
            ]
            pa_ending = pa_ending[~events_series.isin(excluded_events)]
            events_series = pa_ending['events'].astype(str).str.lower()
            
            # Hits - all types
            hits = pa_ending[events_series.isin(['single', 'double', 'triple', 'home_run'])]
            
            # At-bats exclude walks, HBP, sacrifices, errors, and catcher interference
            outs = pa_ending[events_series.isin([
                'strikeout', 'strikeout_double_play', 'field_out', 'force_out', 
                'grounded_into_double_play', 'fielders_choice', 'fielders_choice_out',
                'double_play', 'triple_play'
            ])]
            
            # Walks and HBP (these count in PA but not in AB)
            walks = pa_ending[events_series.isin(['walk', 'intent_walk', 'hit_by_pitch'])]
            
            # Strikeouts
            strikeouts = pa_ending[events_series.isin(['strikeout', 'strikeout_double_play'])]
            
            # Plate appearances = all PA-ending events
            total_pa = len(pa_ending)
            # At-bats = hits + outs (excludes walks, HBP, sacrifices, errors)
            at_bats = len(hits) + len(outs)
            
            # Calculate batting average (hits / at-bats)
            if at_bats > 0:
                metrics['avg'] = round(len(hits) / at_bats, 3)
            
            # Calculate on-base percentage ((hits + walks) / plate appearances)
            if total_pa > 0:
                metrics['obp'] = round((len(hits) + len(walks)) / total_pa, 3)
                metrics['k_rate'] = round(len(strikeouts) / total_pa, 3)
                metrics['bb_rate'] = round(len(walks) / total_pa, 3)
            
            # Calculate SLG and ISO
            if 'avg' in metrics and at_bats > 0:
                total_bases = 0
                for _, row in hits.iterrows():
                    event = str(row['events']).lower()
                    if event == 'single':
                        total_bases += 1
                    elif event == 'double':
                        total_bases += 2
                    elif event == 'triple':
                        total_bases += 3
                    elif event == 'home_run':
                        total_bases += 4
                
                metrics['slg'] = round(total_bases / at_bats, 3)
                metrics['iso'] = round(metrics['slg'] - metrics['avg'], 3)
                
                if 'obp' in metrics:
                    metrics['ops'] = round(metrics['obp'] + metrics['slg'], 3)
    
    # Batted ball metrics
    if {'launch_speed', 'launch_angle'}.issubset(df.columns):
        batted = df[df['launch_speed'].notna() & desc.str.contains('hit_into_play') & ~desc.str.contains('bunt')]
        
        if not batted.empty:
            metrics['avg_ev'] = round(batted['launch_speed'].mean(), 1)
            metrics['avg_la'] = round(batted['launch_angle'].mean(), 1)
            metrics['hard_hit_pct'] = round((batted['launch_speed'] > 95).mean() * 100, 1)
            
            # Use launch_speed_angle column if available (contains barrel classification)
            if 'launch_speed_angle' in batted.columns:
                # launch_speed_angle = 6 means barrel
                barrels = batted[batted['launch_speed_angle'] == 6]
                metrics['barrel_pct'] = round((len(barrels) / len(batted)) * 100, 1) if len(batted) > 0 else 0.0
            else:
                # Fallback: calculate barrels manually (simplified formula)
                barrels = batted[(batted['launch_speed'] >= 98) & 
                                (batted['launch_angle'] >= 8) & 
                                (batted['launch_angle'] <= 50)]
                metrics['barrel_pct'] = round((len(barrels) / len(batted)) * 100, 1) if len(batted) > 0 else 0.0
    
    # Plate discipline
    is_swing = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'])
    is_whiff = desc.isin(['swinging_strike', 'swinging_strike_blocked'])
    
    total_pitches = len(df)
    if total_pitches > 0:
        total_swings = is_swing.sum()
        if total_swings > 0:
            metrics['whiff_pct'] = round(is_whiff.sum() / total_swings * 100, 1)
    
    return metrics

def identify_strengths_weaknesses(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify hitter strengths and weaknesses by pitch type."""
    strengths = []
    weaknesses = []
    
    if df.empty or 'pitch_type' not in df.columns:
        return {"strengths": strengths, "weaknesses": weaknesses}
    
    desc = df['description'].astype(str).str.lower()
    is_swing = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'])
    is_whiff = desc.isin(['swinging_strike', 'swinging_strike_blocked'])
    
    # Group by pitch type
    pitch_groups = df.groupby('pitch_type')
    
    pitch_performance = []
    for pitch_type, group in pitch_groups:
        if len(group) < 10:  # Need minimum sample size
            continue
        
        # Calculate metrics
        swings = is_swing.loc[group.index].sum()
        whiffs = is_whiff.loc[group.index].sum()
        whiff_rate = (whiffs / swings * 100) if swings > 0 else 0
        
        # Batted ball quality
        batted = group[group['launch_speed'].notna() & desc.loc[group.index].str.contains('hit_into_play')]
        if not batted.empty:
            avg_ev = batted['launch_speed'].mean()
            hard_hit = (batted['launch_speed'] > 95).mean() * 100
        else:
            avg_ev = 0
            hard_hit = 0
        
        pitch_performance.append({
            'pitch_type': pitch_type,
            'whiff_rate': whiff_rate,
            'avg_ev': avg_ev,
            'hard_hit_pct': hard_hit,
            'sample_size': len(group)
        })
    
    if not pitch_performance:
        return {"strengths": strengths, "weaknesses": weaknesses}
    
    # Identify strengths (low whiff, high EV/hard hit)
    low_whiff = [p for p in pitch_performance if p['whiff_rate'] < 20 and p['sample_size'] >= 20]
    high_ev = [p for p in pitch_performance if p['avg_ev'] > 90 and p['sample_size'] >= 10]
    high_hard_hit = [p for p in pitch_performance if p['hard_hit_pct'] > 40 and p['sample_size'] >= 10]
    
    if low_whiff:
        best = min(low_whiff, key=lambda x: x['whiff_rate'])
        strengths.append(f"Excellent contact vs {best['pitch_type']} ({best['whiff_rate']:.1f}% whiff)")
    
    if high_ev:
        best = max(high_ev, key=lambda x: x['avg_ev'])
        strengths.append(f"Strong exit velocity vs {best['pitch_type']} ({best['avg_ev']:.1f} MPH avg)")
    
    if high_hard_hit:
        best = max(high_hard_hit, key=lambda x: x['hard_hit_pct'])
        strengths.append(f"High hard hit rate vs {best['pitch_type']} ({best['hard_hit_pct']:.1f}%)")
    
    # Identify weaknesses (high whiff, low EV)
    high_whiff = [p for p in pitch_performance if p['whiff_rate'] > 30 and p['sample_size'] >= 20]
    low_ev = [p for p in pitch_performance if p['avg_ev'] < 85 and p['sample_size'] >= 10 and p['avg_ev'] > 0]
    
    if high_whiff:
        worst = max(high_whiff, key=lambda x: x['whiff_rate'])
        weaknesses.append(f"Struggles vs {worst['pitch_type']} ({worst['whiff_rate']:.1f}% whiff)")
    
    if low_ev:
        worst = min(low_ev, key=lambda x: x['avg_ev'])
        weaknesses.append(f"Low exit velocity vs {worst['pitch_type']} ({worst['avg_ev']:.1f} MPH avg)")
    
    return {"strengths": strengths[:3], "weaknesses": weaknesses[:3]}

def create_strengths_weaknesses_visual(
    df: pd.DataFrame,
    hitter_name: str,
    out_path: Path
) -> Optional[str]:
    """Create a clean, simple, and creative visual for performance by pitch type, split by pitcher handedness."""
    if df.empty or 'pitch_type' not in df.columns:
        return None
    
    # Check if we have pitcher handedness data
    has_handedness = 'p_throws' in df.columns
    
    def calculate_pitch_metrics(data_subset):
        """Helper function to calculate metrics for a subset of data."""
        desc = data_subset['description'].astype(str).str.lower()
        is_swing = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'])
        is_whiff = desc.isin(['swinging_strike', 'swinging_strike_blocked'])
        
        pitch_data = []
        pitch_groups = data_subset.groupby('pitch_type')
        
        for pitch_type, group in pitch_groups:
            if len(group) < 10:
                continue
            
            swings = is_swing.loc[group.index].sum()
            whiffs = is_whiff.loc[group.index].sum()
            whiff_rate = (whiffs / swings * 100) if swings > 0 else 0
            
            batted = group[group['launch_speed'].notna() & desc.loc[group.index].str.contains('hit_into_play')]
            if not batted.empty:
                avg_ev = batted['launch_speed'].mean()
            else:
                avg_ev = 0
            
            pitch_data.append({
                'pitch_type': pitch_type,
                'whiff_rate': whiff_rate,
                'avg_ev': avg_ev,
                'sample_size': len(group)
            })
        
        # Sort by sample size (most seen first)
        pitch_data.sort(key=lambda x: x['sample_size'], reverse=True)
        return pitch_data
    
    def draw_bar_chart(ax, pitch_data, title):
        """Helper function to draw a horizontal bar chart on the given axes."""
        if not pitch_data:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                   fontsize=14, color='#6b7280', transform=ax.transAxes)
            ax.axis('off')
            return
        
        pitch_types = [p['pitch_type'] for p in pitch_data]
        whiff_rates = [p['whiff_rate'] for p in pitch_data]
        avg_evs = [p['avg_ev'] for p in pitch_data]
        
        y_pos = np.arange(len(pitch_types))
        bar_height = 0.6
        
        # Create two sub-axes for side-by-side bars
        # Left side: Whiff Rate
        ax_left = ax.inset_axes([0.0, 0.0, 0.48, 1.0])
        # Right side: Exit Velocity
        ax_right = ax.inset_axes([0.52, 0.0, 0.48, 1.0])
        
        # Left chart: Whiff Rate (horizontal bars)
        # Lower whiff rate is better, so green for low, red for high
        max_whiff = max(whiff_rates) if whiff_rates else 50
        colors_whiff = ['#dc2626' if w > 25 else '#16a34a' for w in whiff_rates]  # Red for high, green for low
        bars_left = ax_left.barh(y_pos, whiff_rates, height=bar_height, 
                                 color=colors_whiff, edgecolor='white', linewidth=2, alpha=0.9)
        
        # Add value labels on bars - always white text for visibility
        for i, (bar, rate) in enumerate(zip(bars_left, whiff_rates)):
            width = bar.get_width()
            # Position label inside bar if there's room, otherwise outside
            if width > max_whiff * 0.15:  # If bar is wide enough, put label inside
                ax_left.text(width - max_whiff * 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{rate:.1f}%', ha='right', va='center', 
                            fontsize=10, weight='700', color='#ffffff')
            else:  # If bar is too narrow, put label outside
                ax_left.text(width + max_whiff * 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{rate:.1f}%', ha='left', va='center', 
                            fontsize=10, weight='700', color='#0a0a0a')
        
        ax_left.set_xlim(0, max_whiff * 1.15 if max_whiff > 0 else 50)
        ax_left.set_ylim(-0.5, len(pitch_types) - 0.5)
        ax_left.invert_yaxis()
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(pitch_types, fontsize=10, weight='700', color='#0a0a0a')
        ax_left.set_xlabel('Whiff Rate (%)', fontsize=10, weight='700', color='#0a0a0a', labelpad=8)
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['left'].set_visible(False)
        ax_left.spines['bottom'].set_color('#0a0a0a')
        ax_left.spines['bottom'].set_linewidth(2)
        ax_left.tick_params(axis='x', which='major', labelsize=9, colors='#4b5563', width=0, length=0)
        ax_left.tick_params(axis='y', which='major', labelsize=10, colors='#0a0a0a', width=0, length=0)
        ax_left.set_facecolor('white')
        ax_left.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.8, color='#d1d5db', zorder=0)
        ax_left.set_axisbelow(True)
        
        # Right chart: Exit Velocity (horizontal bars)
        # Higher exit velocity is better, so green for high, red for low
        ev_min = min(avg_evs) if avg_evs else 80
        ev_max = max(avg_evs) if avg_evs else 100
        ev_range = ev_max - ev_min if (ev_max - ev_min) > 0 else 20
        # Use 90 MPH as threshold - green above, red below
        colors_ev = ['#16a34a' if e > 90 else '#dc2626' for e in avg_evs]  # Green for high, red for low
        bars_right = ax_right.barh(y_pos, avg_evs, height=bar_height,
                                   color=colors_ev, edgecolor='white', linewidth=2, alpha=0.9)
        
        # Add value labels on bars - always white text for visibility
        for i, (bar, ev) in enumerate(zip(bars_right, avg_evs)):
            width = bar.get_width()
            # Position label inside bar if there's room, otherwise outside
            bar_width_pct = (width - ev_min) / ev_range if ev_range > 0 else 0.5
            if bar_width_pct > 0.15:  # If bar is wide enough, put label inside
                ax_right.text(width - ev_range * 0.02 if ev_range > 0 else width - 1, 
                             bar.get_y() + bar.get_height()/2, 
                             f'{ev:.1f}', ha='right', va='center', 
                             fontsize=10, weight='700', color='#ffffff')
            else:  # If bar is too narrow, put label outside
                ax_right.text(width + ev_range * 0.02 if ev_range > 0 else width + 1, 
                             bar.get_y() + bar.get_height()/2, 
                             f'{ev:.1f}', ha='left', va='center', 
                             fontsize=10, weight='700', color='#0a0a0a')
        
        ax_right.set_xlim(ev_min - (ev_max - ev_min) * 0.1 if (ev_max - ev_min) > 0 else 75, 
                         ev_max + (ev_max - ev_min) * 0.1 if (ev_max - ev_min) > 0 else 105)
        ax_right.set_ylim(-0.5, len(pitch_types) - 0.5)
        ax_right.invert_yaxis()
        ax_right.set_yticks(y_pos)
        ax_right.set_yticklabels([])  # No labels on right side
        ax_right.set_xlabel('Exit Velocity (MPH)', fontsize=10, weight='700', color='#0a0a0a', labelpad=8)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['bottom'].set_color('#0a0a0a')
        ax_right.spines['bottom'].set_linewidth(2)
        ax_right.tick_params(axis='x', which='major', labelsize=9, colors='#4b5563', width=0, length=0)
        ax_right.tick_params(axis='y', which='major', labelsize=0, width=0, length=0)
        ax_right.set_facecolor('white')
        ax_right.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.8, color='#d1d5db', zorder=0)
        ax_right.set_axisbelow(True)
        
        # Main axes styling - add title if provided
        ax.axis('off')
        if title:
            ax.text(0.5, 1.02, title, ha='center', va='bottom', 
                   transform=ax.transAxes, fontsize=14, weight='700', color='#0a0a0a')
    
    # Split by handedness if available
    if has_handedness:
        rhp_df = df[df['p_throws'] == 'R'].copy()
        lhp_df = df[df['p_throws'] == 'L'].copy()
        
        rhp_data = calculate_pitch_metrics(rhp_df)
        lhp_data = calculate_pitch_metrics(lhp_df)
        
        # Only create visual if we have data for at least one side
        if not rhp_data and not lhp_data:
            return None
        
        # Create figure with two subplots side-by-side
        fig = plt.figure(figsize=(16, 7), facecolor='white')
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25, left=0.06, right=0.97, top=0.85, bottom=0.12)
        
        # RHP subplot
        ax1 = fig.add_subplot(gs[0, 0])
        draw_bar_chart(ax1, rhp_data, 'vs RHP')
        
        # LHP subplot
        ax2 = fig.add_subplot(gs[0, 1])
        draw_bar_chart(ax2, lhp_data, 'vs LHP')
    else:
        # No handedness data, create single chart as before
        pitch_data = calculate_pitch_metrics(df)
        if not pitch_data:
            return None
        
        fig = plt.figure(figsize=(14, 7), facecolor='white')
        ax = fig.add_subplot(111)
        draw_bar_chart(ax, pitch_data, '')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close(fig)
    
    return str(out_path)

def render_hitter_checkin_pngs(
    df_hitter: pd.DataFrame,
    hitter_name: str,
    out_dir: str = "build/figures"
) -> Dict[str, Optional[str]]:
    """
    Returns {"RHP": path_or_None, "LHP": path_or_None}
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    required = {'pitch_type','p_throws','description'}
    if not required.issubset(df_hitter.columns):
        return {"RHP": None, "LHP": None}

    outputs: Dict[str, Optional[str]] = {}
    for hand, label in (('R','RHP'),('L','LHP')):
        side = df_hitter[df_hitter['p_throws']==hand].copy()
        if side.empty:
            outputs[label] = None
            continue
        summary = _summarize(side)
        table_df, colors = _format_for_table(summary)
        out_path = Path(out_dir) / f"{hitter_name.replace(' ','_')}_{label}_checkin.png"
        outputs[label] = _draw_table(table_df, colors, f"{hitter_name} vs. {label}", out_path)
    return outputs
