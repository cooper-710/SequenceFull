# src/plots_hitter_advanced.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PITCH_NAME_MAP = {
    "FF":"Four-Seam Fastball","FT":"Two-Seam Fastball","SI":"Sinker","FC":"Cutter",
    "SL":"Slider","ST":"Sweeper","CU":"Curveball","KC":"Knuckle Curve",
    "CS":"Slow Curve","CH":"Changeup","FS":"Splitter","KN":"Knuckleball",
    "SV":"Slurve","FO":"Forkball","EP":"Eephus","SC":"Screwball","PO":"Pitch Out",
    "IN":"Intentional Ball","UN":"Unknown"
}

def _draw_strike_zone(ax, include_home_plate=True):
    """Draw a standard strike zone with grid.
    
    Args:
        ax: Matplotlib axis
        include_home_plate: If False, don't draw home plate (for spray charts)
    """
    # Fixed axis limits to ensure consistent sizing across all charts
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    
    # Strike zone grid (3x3) - fixed dimensions
    w, h = 0.57, 0.67
    xs, ys = -0.855, 3.5
    for r in range(3):
        for c in range(3):
            ax.add_patch(patches.Rectangle(
                (xs + c*w, ys - (r+1)*h), w, h,
                linewidth=1.5,
                edgecolor='#4b5563',
                facecolor='none',
                alpha=0.6,
                linestyle='-'
            ))
    
    # Home plate (only if requested)
    if include_home_plate:
        plate = [[-0.7085, 0.7085], [0.7085, 0.7085], [0.35425, 0.2085], [0, 0], [-0.35425, 0.2085]]
        ax.add_patch(patches.Polygon(
            plate, closed=True,
            edgecolor='#1a1a1a',
            facecolor='white',
            linewidth=2.0,
            zorder=2,
            alpha=0.9
        ))

def render_xslg_by_pitch_type(
    df: pd.DataFrame,
    hitter_name: str,
    out_dir: str = "build/figures"
) -> Optional[str]:
    """
    Create a horizontal bar chart showing xSLG by pitch type.
    Returns path to saved image or None.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter for batted balls with xSLG data
    # Batted balls have type=='X' OR have launch_speed/launch_angle data
    is_batted = None
    if 'type' in df.columns:
        is_batted = (df['type'] == 'X')
    elif 'launch_speed' in df.columns:
        is_batted = df['launch_speed'].notna()
    elif 'launch_angle' in df.columns:
        is_batted = df['launch_angle'].notna()
    
    if is_batted is None:
        return None
    
    batted = df[
        is_batted &
        (df['estimated_slg_using_speedangle'].notna()) &
        (df['pitch_type'].notna())
    ].copy()
    
    if batted.empty:
        return None
    
    # Calculate xSLG by pitch type (mean)
    xslg_by_pitch = batted.groupby('pitch_type')['estimated_slg_using_speedangle'].agg(['mean', 'count']).reset_index()
    xslg_by_pitch = xslg_by_pitch[xslg_by_pitch['count'] >= 5]  # Minimum sample size
    xslg_by_pitch = xslg_by_pitch.sort_values('mean', ascending=True)
    
    if xslg_by_pitch.empty:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(xslg_by_pitch) * 0.5)))
    fig.patch.set_facecolor('white')
    
    # Get pitch types and xSLG values
    pitch_types = xslg_by_pitch['pitch_type'].values
    xslg_values = xslg_by_pitch['mean'].values
    
    # Color bars based on xSLG value (green = high, red = low)
    colors = []
    xslg_min = xslg_values.min()
    xslg_max = xslg_values.max()
    xslg_range = xslg_max - xslg_min if (xslg_max - xslg_min) > 0 else 1
    
    for val in xslg_values:
        normalized = (val - xslg_min) / xslg_range
        if normalized > 0.6:
            colors.append('#16a34a')  # Green for high
        elif normalized > 0.4:
            colors.append('#eab308')  # Yellow for medium
        else:
            colors.append('#dc2626')  # Red for low
    
    # Create horizontal bars
    y_pos = np.arange(len(pitch_types))
    bars = ax.barh(y_pos, xslg_values, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, xslg_values)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                ha='left', va='center',
                fontsize=11, weight='700', color='#0a0a0a')
    
    # Set y-axis labels
    pitch_labels = [PITCH_NAME_MAP.get(pt, pt) for pt in pitch_types]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pitch_labels, fontsize=11, weight='600', color='#0a0a0a')
    
    # Set x-axis
    ax.set_xlim(0, max(xslg_values) * 1.15)
    ax.set_xlabel('Expected Slugging (xSLG)', fontsize=12, weight='700', color='#0a0a0a', labelpad=10)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#0a0a0a')
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='x', which='major', labelsize=10, colors='#4b5563')
    ax.tick_params(axis='y', which='major', labelsize=10, colors='#0a0a0a', left=False)
    ax.set_facecolor('white')
    ax.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.8, color='#d1d5db', zorder=0)
    ax.set_axisbelow(True)
    
    # Title
    ax.set_title(f"{hitter_name} | xSLG vs Pitch Type", fontsize=14, weight='700', color='#0a0a0a', pad=15)
    
    plt.tight_layout()
    out_path = Path(out_dir) / f"{hitter_name.replace(' ', '_')}_xslg_by_pitch.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig)
    
    return str(out_path)

def render_xwoba_location_heatmaps(
    df: pd.DataFrame,
    hitter_name: str,
    out_dir: str = "build/figures"
) -> Optional[str]:
    """
    Create a grid of strike zone heatmaps showing xwOBA by pitch type.
    Returns path to saved image or None.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # FIRST: Determine top 9 pitch types from FULL dataframe (same as xSLG + Whiff%)
    # This ensures both charts show the same pitch types
    pitch_counts = df.groupby('pitch_type').size()
    valid_pitches = pitch_counts[pitch_counts >= 10].index.tolist()
    
    if not valid_pitches:
        return None
    
    # Sort by usage and limit to top 9 (match xSLG + Whiff% exactly)
    usage = df['pitch_type'].value_counts(normalize=True)
    valid_pitches = sorted(valid_pitches, key=lambda x: usage.get(x, 0), reverse=True)[:9]
    
    if not valid_pitches:
        return None
    
    # THEN: Filter for pitches with location and xwOBA data (for visualization only)
    # Only include the selected pitch types
    data = df[
        (df['plate_x'].notna()) &
        (df['plate_z'].notna()) &
        (df['estimated_woba_using_speedangle'].notna()) &
        (df['pitch_type'].notna()) &
        (df['pitch_type'].isin(valid_pitches))  # Only include our selected pitch types
    ].copy()
    
    if data.empty:
        return None
    
    # Calculate grid: 3 columns, adjust rows as needed (max 3 rows for 9 pitches)
    ncols = 3
    nrows = (len(valid_pitches) + ncols - 1) // ncols  # Ceiling division
    
    # Use gridspec for consistent subplot sizing (matching xSLG + Whiff% chart)
    fig = plt.figure(figsize=(12, max(10, nrows * 2.5)))
    fig.patch.set_facecolor('white')
    
    # Match the gridspec settings from xSLG + Whiff% chart for consistent sizing
    gs = fig.add_gridspec(nrows, ncols, hspace=0.3, wspace=0.2, left=0.06, right=0.97, top=0.92, bottom=0.08)
    
    # Create axes array as 2D list
    axes = []
    for row in range(nrows):
        axes_row = []
        for col in range(ncols):
            axes_row.append(fig.add_subplot(gs[row, col]))
        axes.append(axes_row)
    # axes is now always 2D: axes[row][col]
    
    # Create custom colormap for xwOBA (blue = low, red = high)
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    heatmap_cmap = LinearSegmentedColormap.from_list('blue_green_red', colors, N=256)
    
    # Plot each pitch type
    for idx, pitch_type in enumerate(valid_pitches):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]  # Always access as 2D array
        
        pitch_data = data[data['pitch_type'] == pitch_type].copy()
        
        if not pitch_data.empty:
            # Use KDE heatmap showing xwOBA values
            try:
                # Filter for pitches with valid xwOBA values
                valid_data = pitch_data[
                    (pitch_data['plate_x'].notna()) &
                    (pitch_data['plate_z'].notna()) &
                    (pitch_data['estimated_woba_using_speedangle'].notna())
                ]
                
                if not valid_data.empty and len(valid_data) >= 5:
                    # Create smooth KDE heatmap showing xwOBA values
                    # Use seaborn KDE for smooth heatmap visualization
                    try:
                        # Use KDE plot with xwOBA values for coloring
                        # Create a weighted visualization by using multiple points weighted by xwOBA
                        expanded_points = []
                        x_vals = valid_data['plate_x'].values
                        z_vals = valid_data['plate_z'].values
                        xwoba_vals = valid_data['estimated_woba_using_speedangle'].values
                        
                        # Normalize xwOBA to create appropriate weights (scale to 1-10 range)
                        min_xwoba = xwoba_vals.min()
                        max_xwoba = xwoba_vals.max()
                        xwoba_range = max_xwoba - min_xwoba if (max_xwoba - min_xwoba) > 0 else 1
                        normalized_xwoba = ((xwoba_vals - min_xwoba) / xwoba_range) * 9 + 1
                        
                        # Create expanded dataset weighted by xwOBA
                        for x, z, weight in zip(x_vals, z_vals, normalized_xwoba):
                            num_points = int(weight)
                            for _ in range(max(1, num_points)):
                                expanded_points.append({'plate_x': x, 'plate_z': z})
                        
                        if expanded_points:
                            expanded_df = pd.DataFrame(expanded_points)
                            
                            # Create smooth KDE heatmap
                            sns.kdeplot(
                                data=expanded_df,
                                x='plate_x',
                                y='plate_z',
                                fill=True,
                                cmap=heatmap_cmap,
                                bw_adjust=0.7,
                                thresh=0.02,
                                levels=25,
                                ax=ax,
                                alpha=0.85,
                                antialiased=True
                            )
                            # Ensure axis limits remain fixed after KDE plot
                            ax.set_xlim(-2, 2)
                            ax.set_ylim(0, 5)
                    except Exception as e:
                        # Fallback: Simple scatter with smoothing
                        try:
                            scatter = ax.scatter(
                                valid_data['plate_x'],
                                valid_data['plate_z'],
                                c=valid_data['estimated_woba_using_speedangle'],
                                cmap=heatmap_cmap,
                                s=40,
                                alpha=0.7,
                                edgecolors='white',
                                linewidth=0.3
                            )
                        except Exception:
                            pass
            except Exception:
                # Fallback: use simple KDE with coloring by xwOBA values
                try:
                    # Group nearby points and show mean xwOBA
                    valid_data = pitch_data[
                        (pitch_data['plate_x'].notna()) &
                        (pitch_data['plate_z'].notna()) &
                        (pitch_data['estimated_woba_using_speedangle'].notna())
                    ]
                    
                    if not valid_data.empty:
                        # Use scatter with KDE-like density coloring
                        scatter = ax.scatter(
                            valid_data['plate_x'],
                            valid_data['plate_z'],
                            c=valid_data['estimated_woba_using_speedangle'],
                            cmap=heatmap_cmap,
                            s=30,
                            alpha=0.7,
                            edgecolors='white',
                            linewidth=0.3
                        )
                        # Ensure axis limits remain fixed after scatter plot
                        ax.set_xlim(-2, 2)
                        ax.set_ylim(0, 5)
                except Exception:
                    pass
        
        # Draw strike zone (without home plate for cleaner look, matching xSLG chart)
        _draw_strike_zone(ax, include_home_plate=False)
        
        # Ensure axis labels are removed
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
        
        # Final check: ensure axis limits are exactly the same as xSLG chart
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 5)
        
        # Add title with pitch type and usage %
        usage_pct = usage.get(pitch_type, 0) * 100
        pitch_label = PITCH_NAME_MAP.get(pitch_type, pitch_type)
        ax.set_title(f"{pitch_label} - {usage_pct:.1f}%", fontsize=9, weight='700', color='#0a0a0a', pad=5)
    
    # Hide unused subplots
    for idx in range(len(valid_pitches), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis('off')  # Always access as 2D array
    
    # Main title
    fig.suptitle(f"{hitter_name} | xwOBA Location by Pitch Type", fontsize=14, weight='700', color='#0a0a0a', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = Path(out_dir) / f"{hitter_name.replace(' ', '_')}_xwoba_heatmaps.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig)
    
    return str(out_path)

def render_xslg_whiff_spray(
    df: pd.DataFrame,
    hitter_name: str,
    out_dir: str = "build/figures"
) -> Optional[str]:
    """
    Create visualization with strike zone charts per pitch type showing xSLG heatmap, whiff zones, and spray direction.
    Note: Spray chart is now separate (see render_spray_chart).
    Returns path to saved image or None.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Get pitch types with sufficient sample size
    pitch_counts = df.groupby('pitch_type').size()
    valid_pitches = pitch_counts[pitch_counts >= 10].index.tolist()
    
    if not valid_pitches:
        return None
    
    # Sort by usage and limit to top 9
    usage = df['pitch_type'].value_counts(normalize=True)
    valid_pitches = sorted(valid_pitches, key=lambda x: usage.get(x, 0), reverse=True)[:9]
    
    # Create figure with 3x3 grid for strike zones only
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Increase top margin and hspace to prevent title overlap
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.2, left=0.06, right=0.97, top=0.92, bottom=0.08)
    
    # Create custom colormap for xSLG
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    xslg_cmap = LinearSegmentedColormap.from_list('blue_green_red', colors, N=256)
    
    # Plot strike zone charts (3x3 grid)
    for idx, pitch_type in enumerate(valid_pitches):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        pitch_data = df[df['pitch_type'] == pitch_type].copy()
        
        # Filter for batted balls for xSLG
        # Batted balls have type=='X' OR have launch_speed/launch_angle data
        is_batted_pitch = None
        if 'type' in pitch_data.columns:
            is_batted_pitch = (pitch_data['type'] == 'X')
        elif 'launch_speed' in pitch_data.columns:
            is_batted_pitch = pitch_data['launch_speed'].notna()
        elif 'launch_angle' in pitch_data.columns:
            is_batted_pitch = pitch_data['launch_angle'].notna()
        
        if is_batted_pitch is None:
            batted = pd.DataFrame()
        else:
            # Check if estimated_slg column exists
            if 'estimated_slg_using_speedangle' in pitch_data.columns:
                batted = pitch_data[
                    is_batted_pitch &
                    (pitch_data['plate_x'].notna()) &
                    (pitch_data['plate_z'].notna()) &
                    (pitch_data['estimated_slg_using_speedangle'].notna())
                ]
            else:
                batted = pd.DataFrame()
        
        # Draw xSLG heatmap if we have batted balls
        if not batted.empty and len(batted) >= 3:
            try:
                # Create smooth KDE heatmap for xSLG values (similar to xwOBA)
                valid_batted = batted[
                    (batted['plate_x'].notna()) &
                    (batted['plate_z'].notna()) &
                    (batted['estimated_slg_using_speedangle'].notna())
                ]
                
                if not valid_batted.empty:
                    # Use weighted KDE approach for xSLG heatmap
                    expanded_points = []
                    x_vals = valid_batted['plate_x'].values
                    z_vals = valid_batted['plate_z'].values
                    xslg_vals = valid_batted['estimated_slg_using_speedangle'].values
                    
                    # Normalize xSLG to create appropriate weights
                    min_xslg = xslg_vals.min()
                    max_xslg = xslg_vals.max()
                    xslg_range = max_xslg - min_xslg if (max_xslg - min_xslg) > 0 else 1
                    normalized_xslg = ((xslg_vals - min_xslg) / xslg_range) * 9 + 1
                    
                    # Create expanded dataset weighted by xSLG
                    for x, z, weight in zip(x_vals, z_vals, normalized_xslg):
                        num_points = int(weight)
                        for _ in range(max(1, num_points)):
                            expanded_points.append({'plate_x': x, 'plate_z': z})
                    
                    if expanded_points:
                        expanded_df = pd.DataFrame(expanded_points)
                        
                        # Create smooth KDE heatmap
                        sns.kdeplot(
                            data=expanded_df,
                            x='plate_x',
                            y='plate_z',
                            fill=True,
                            cmap=xslg_cmap,
                            bw_adjust=0.7,
                            thresh=0.02,
                            levels=25,
                            ax=ax,
                            alpha=0.85,
                            antialiased=True
                        )
                        # Ensure axis limits remain fixed after KDE plot
                        ax.set_xlim(-2, 2)
                        ax.set_ylim(0, 5)
            except Exception:
                pass
        
        # Identify whiff zones (areas with high whiff rate)
        desc = pitch_data['description'].astype(str).str.lower()
        is_whiff = desc.isin(['swinging_strike', 'swinging_strike_blocked'])
        whiff_data = pitch_data[is_whiff & pitch_data['plate_x'].notna() & pitch_data['plate_z'].notna()]
        
        if not whiff_data.empty and len(whiff_data) >= 3:
            # Draw whiff zones as green contours
            try:
                sns.kdeplot(
                    data=whiff_data,
                    x='plate_x',
                    y='plate_z',
                    levels=[0.3, 0.5, 0.7],
                    fill=False,
                    colors=['#16a34a'],
                    linewidths=2,
                    alpha=0.8,
                    ax=ax
                )
                # Ensure axis limits remain fixed after whiff contour plot
                ax.set_xlim(-2, 2)
                ax.set_ylim(0, 5)
            except Exception:
                pass
        
        # Draw strike zone (without home plate for cleaner look)
        _draw_strike_zone(ax, include_home_plate=False)
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
        
        # Title
        pitch_label = PITCH_NAME_MAP.get(pitch_type, pitch_type)
        ax.set_title(pitch_label, fontsize=10, weight='700', color='#0a0a0a', pad=5)
    
    # Hide unused strike zone subplots
    for idx in range(len(valid_pitches), 9):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Main title (removed "Spray" as requested)
    # Position title higher to avoid overlap with subplot titles
    fig.suptitle(f"{hitter_name} | xSLG + Whiff%", fontsize=14, weight='700', color='#0a0a0a', y=0.98)
    
    # Use tight_layout with more padding at top to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(out_dir) / f"{hitter_name.replace(' ', '_')}_xslg_whiff_spray.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig)
    
    return str(out_path)

def render_spray_chart(
    df: pd.DataFrame,
    hitter_name: str,
    out_dir: str = "build/figures"
) -> Optional[str]:
    """
    Create a large baseball field spray chart showing batted ball locations.
    Returns path to saved image or None.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter for batted balls with hit coordinates
    # Batted balls have type=='X' OR have launch_speed/launch_angle data
    is_batted_all = None
    if 'type' in df.columns:
        is_batted_all = (df['type'] == 'X')
    elif 'launch_speed' in df.columns:
        is_batted_all = df['launch_speed'].notna()
    elif 'launch_angle' in df.columns:
        is_batted_all = df['launch_angle'].notna()
    
    # Check if hc_x and hc_y columns exist
    has_hc_x = 'hc_x' in df.columns
    has_hc_y = 'hc_y' in df.columns
    
    if is_batted_all is not None and has_hc_x and has_hc_y:
        batted_all = df[
            is_batted_all &
            (df['hc_x'].notna()) &
            (df['hc_y'].notna())
        ].copy()
    else:
        batted_all = pd.DataFrame()
    
    if batted_all.empty:
        return None
    
    # Create compact figure for spray chart (to fit on same page)
    # Smaller figure size to fit on page with other visuals
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    
    # Statcast coordinate system reference (fixed, from Statcast documentation)
    statcast_home_x = 125.42
    statcast_home_y = 198.27
    
    # Fixed field dimensions in Statcast coordinate units
    # Based on spraychart.html: fence at 290ft = 145 Statcast units, stadium at 400ft = 200 Statcast units
    # Use FIXED dimensions, not calculated from data
    FIXED_FENCE_DISTANCE = 145  # 290 feet in Statcast units (outfield fence)
    FIXED_STADIUM_DISTANCE = 200  # 400 feet in Statcast units (deepest point)
    FOUL_LINE_DISTANCE = 145  # ~290ft at foul poles in Statcast units
    BASE_DISTANCE = 45  # ~90ft in Statcast units
    
    # Calculate scale factor to fit the fixed fence within our figure
    # We want the fence to take up most of the vertical space (with some margin)
    # Figure height in inches * DPI = pixels, but we'll use a simpler approach
    # Target: fence should be ~75% of plot height
    plot_height = 6.5  # Approximate plot height in display units
    target_fence_height = plot_height * 0.75
    
    # Scale factor: how many Statcast units per display unit
    # If fence is 145 Statcast units and we want it to be target_fence_height display units
    scale_factor = FIXED_FENCE_DISTANCE / target_fence_height
    
    # Scale all dimensions
    fence_scaled = FIXED_FENCE_DISTANCE / scale_factor
    stadium_scaled = FIXED_STADIUM_DISTANCE / scale_factor
    foul_line_scaled = FOUL_LINE_DISTANCE / scale_factor
    base_distance_scaled = BASE_DISTANCE / scale_factor
    
    # Set axis limits - use fixed fence distance, not data-dependent
    margin = 2
    ax.set_xlim(-foul_line_scaled - margin, foul_line_scaled + margin)
    ax.set_ylim(-margin, fence_scaled + margin)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor('#f0f0f0')
    
    # Draw field outline (NO infield diamond, bases, or home plate - just foul lines and fence)
    # Home plate position (for reference, but not drawn)
    plate_center_x = 0
    plate_center_y = 0
    
    # Foul lines (approximately 45 degrees from center)
    foul_line_angle = np.pi / 4  # 45 degrees
    left_foul_x = -foul_line_scaled * np.cos(foul_line_angle)
    left_foul_y = foul_line_scaled * np.sin(foul_line_angle)
    ax.plot([plate_center_x, left_foul_x], [plate_center_y, left_foul_y], 'k-', linewidth=2, alpha=0.6)
    
    right_foul_x = foul_line_scaled * np.cos(foul_line_angle)
    right_foul_y = foul_line_scaled * np.sin(foul_line_angle)
    ax.plot([plate_center_x, right_foul_x], [plate_center_y, right_foul_y], 'k-', linewidth=2, alpha=0.6)
    
    # Outfield fence (arc from left to right foul pole, deeper in center)
    # Draw a circular arc with radius = fence_scaled, but only between foul line angles
    # The fence is deeper in center field
    foul_line_angle_rad = foul_line_angle  # ~45 degrees in radians
    
    # Draw arc from left foul line angle to right foul line angle
    # In our coordinate system: center field is up (theta=0), right is theta=pi/2, left is theta=-pi/2
    # Foul lines are at approximately -foul_line_angle_rad and +foul_line_angle_rad from vertical
    theta_start = -foul_line_angle_rad
    theta_end = foul_line_angle_rad
    theta_arc = np.linspace(theta_start, theta_end, 200)
    
    # Circular arc with radius = fence_scaled
    arc_x = fence_scaled * np.sin(theta_arc)
    arc_y = fence_scaled * np.cos(theta_arc)
    
    ax.plot(arc_x, arc_y, 'k-', linewidth=2.5, alpha=0.7, zorder=2)
    
    # Center field marker (small vertical line at deepest point)
    cf_marker_height = fence_scaled * 0.02
    ax.plot([0, 0], [fence_scaled - cf_marker_height, fence_scaled + cf_marker_height], 'k-', linewidth=2, alpha=0.6, zorder=2)
    
    # Plot batted balls
    event_colors = {
        'single': '#4CAF50',
        'double': '#2196F3',
        'triple': '#9C27B0',
        'home_run': '#F44336',
        'field_out': '#000000'
    }
    
    # Convert Statcast coordinates to field coordinates
    # hc_x: increases toward right field
    # hc_y: decreases as distance increases
    # Home plate is approximately at (125.42, 198.27)
    statcast_home_x = 125.42
    statcast_home_y = 198.27
    
    singles = []
    doubles = []
    triples = []
    home_runs = []
    outs = []
    
    for _, row in batted_all.iterrows():
        hc_x = row['hc_x']
        hc_y = row['hc_y']
        event = str(row.get('events', 'field_out')).lower()
        
        # Convert Statcast coordinates to display coordinates
        # Matching spraychart.html JavaScript approach exactly
        delta_x = hc_x - statcast_home_x  # Horizontal offset (Statcast units)
        distance_statcast = statcast_home_y - hc_y  # Distance (Statcast units, inverted because y decreases)
        
        # Calculate angle from home plate (for filtering)
        angle = np.arctan2(delta_x, distance_statcast)
        
        # Convert to display coordinates using our scale factor
        # This matches: canvasX = centerX + (deltaX * scale), canvasY = centerY - (distance * scale)
        display_x = delta_x / scale_factor
        display_y = distance_statcast / scale_factor
        
        # Filter: only plot if within reasonable field bounds
        # Allow some margin beyond the fence for home runs
        max_distance_scaled = stadium_scaled + 3  # Allow up to stadium distance
        distance_scaled = np.sqrt(display_x**2 + display_y**2)
        max_angle = np.pi / 2.2  # Slightly less than 90 degrees to exclude extreme foul territory
        
        if (0 <= distance_scaled <= max_distance_scaled and 
            abs(angle) <= max_angle and
            abs(display_x) <= (foul_line_scaled + 2)):
            # Store by event type for layered plotting
            if event == 'single':
                singles.append((display_x, display_y))
            elif event == 'double':
                doubles.append((display_x, display_y))
            elif event == 'triple':
                triples.append((display_x, display_y))
            elif event == 'home_run':
                home_runs.append((display_x, display_y))
            else:
                outs.append((display_x, display_y))
    
    # Plot with layering (outs first, then hits)
    if outs:
        x_outs, y_outs = zip(*outs)
        ax.scatter(x_outs, y_outs, c='#000000', marker='X', s=50, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=5, label='Out')
    
    if singles:
        x_singles, y_singles = zip(*singles)
        ax.scatter(x_singles, y_singles, c='#4CAF50', marker='o', s=60, alpha=0.7, edgecolors='white', linewidth=1, zorder=6, label='Single')
    
    if doubles:
        x_doubles, y_doubles = zip(*doubles)
        ax.scatter(x_doubles, y_doubles, c='#2196F3', marker='o', s=70, alpha=0.7, edgecolors='white', linewidth=1, zorder=6, label='Double')
    
    if triples:
        x_triples, y_triples = zip(*triples)
        ax.scatter(x_triples, y_triples, c='#9C27B0', marker='o', s=80, alpha=0.7, edgecolors='white', linewidth=1, zorder=6, label='Triple')
    
    if home_runs:
        x_hrs, y_hrs = zip(*home_runs)
        ax.scatter(x_hrs, y_hrs, c='#F44336', marker='o', s=90, alpha=0.8, edgecolors='white', linewidth=1.5, zorder=6, label='Home Run')
    
    # Add legend
    legend_elements = []
    if singles:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=10, label='Single'))
    if doubles:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=10, label='Double'))
    if triples:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9C27B0', markersize=10, label='Triple'))
    if home_runs:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=10, label='Home Run'))
    if outs:
        legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#000000', markersize=10, label='Out'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95, fancybox=True, shadow=True)
    
    # Title
    ax.set_title(f"{hitter_name} | Spray Chart", fontsize=16, weight='700', color='#0a0a0a', pad=20)
    
    plt.tight_layout()
    out_path = Path(out_dir) / f"{hitter_name.replace(' ', '_')}_spray_chart.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close(fig)
    
    return str(out_path)

