# src/build_pitch_tables.py
from datetime import date
from typing import Optional, Dict
from scrape_savant import fetch_pitcher_statcast
from plots_pitch_table import render_pitch_table_png, render_pitch_table_combined_png

def build_pitch_tables_for_pitcher(pitcher_id: int, pitcher_name: str, start_date: str, end_date: str, out_dir="build/figures"):
    """
    Returns dict {"RHH": path_or_None, "LHH": path_or_None}
    """
    df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
    # Check if we have sufficient data before rendering
    if df is None or df.empty:
        return {"RHH": None, "LHH": None}
    
    # Some pybaseball pulls may miss player_name; inject provided name for display
    if 'player_name' not in df.columns:
        df['player_name'] = pitcher_name
    
    # Check if we have required columns and minimum data
    required_cols = {'pitch_type', 'description', 'p_throws', 'stand'}
    if not required_cols.issubset(df.columns):
        return {"RHH": None, "LHH": None}
    
    # Need at least some pitches to create a meaningful table
    if len(df) < 5:
        return {"RHH": None, "LHH": None}
    
    return render_pitch_table_png(df, out_dir)

def build_pitch_table_combined_for_pitcher(pitcher_id: int, pitcher_name: str, start_date: str, end_date: str, out_dir="build/figures") -> Optional[str]:
    """
    Returns a single combined pitch table path (not split by batter handedness).
    """
    df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
    # Check if we have sufficient data before rendering
    if df is None or df.empty:
        return None
    
    # Some pybaseball pulls may miss player_name; inject provided name for display
    if 'player_name' not in df.columns:
        df['player_name'] = pitcher_name
    
    # Check if we have required columns and minimum data
    required_cols = {'pitch_type', 'description', 'p_throws'}
    if not required_cols.issubset(df.columns):
        return None
    
    # Need at least some pitches to create a meaningful table
    if len(df) < 5:
        return None
    
    return render_pitch_table_combined_png(df, out_dir)
