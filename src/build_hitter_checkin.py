from typing import Optional
# src/build_hitter_checkin.py
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys
import os
# Add parent directory to path to import generate_report
sys.path.insert(0, str(Path(__file__).parent.parent))
from pybaseball import batting_stats_range, batting_stats
from scrape_savant import lookup_batter_id, fetch_batter_statcast
from plots_hitter_checkin import render_hitter_checkin_pngs, calculate_overall_metrics, identify_strengths_weaknesses, create_strengths_weaknesses_visual
from plots_hitter_advanced import render_xslg_by_pitch_type, render_xwoba_location_heatmaps, render_xslg_whiff_spray, render_spray_chart
import statsapi

def _fetch_statsapi_stats(hitter_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch stats from MLB StatsAPI as fallback/primary source."""
    metrics = {}
    try:
        # Try to find player ID
        player_id = lookup_batter_id(hitter_name)
        if not player_id:
            return metrics
        
        # Parse dates to get season
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        season = start_dt.year
        
        # NOTE: Position should be fetched separately from person endpoint, not here
        # This function only returns batting stats (avg, ops, hr, rbi)
        
        # Get stats from statsapi using player_stat_data function
        try:
            # Use statsapi's player_stat_data function which returns structured data
            player_data = statsapi.player_stat_data(player_id, group='[hitting]', type='season', season=season)
            
            if player_data and 'stats' in player_data:
                # Find hitting stats for the season - try exact match first
                stats_found = False
                for stat_group in player_data['stats']:
                    if (stat_group.get('group', '').lower() == 'hitting' and 
                        stat_group.get('type', '').lower() == 'season' and
                        stat_group.get('season') == season):
                        
                        stats = stat_group.get('stats', {})
                        stats_found = True
                        
                        # Extract stats - check for various possible key names
                        # Batting Average
                        for key in ['avg', 'average', 'battingAverage']:
                            if key in stats and stats[key] is not None:
                                try:
                                    val = float(stats[key])
                                    if val >= 0:  # Sanity check
                                        metrics['avg'] = round(val, 3)
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        # OPS
                        for key in ['ops']:
                            if key in stats and stats[key] is not None:
                                try:
                                    val = float(stats[key])
                                    if val >= 0:  # Sanity check
                                        metrics['ops'] = round(val, 3)
                                        break
                                except (ValueError, TypeError):
                                    pass
                        
                        # Home Runs
                        for key in ['homeRuns', 'hr', 'homeRunsAllowed']:
                            if key in stats and stats[key] is not None:
                                try:
                                    val = int(stats[key])
                                    if val >= 0:  # Sanity check
                                        metrics['hr'] = val
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        # RBIs
                        for key in ['rbi', 'rbis']:
                            if key in stats and stats[key] is not None:
                                try:
                                    val = int(stats[key])
                                    if val >= 0:  # Sanity check
                                        metrics['rbi'] = val
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        break
                
                # If no season-specific stats found, try to get latest/current stats
                if not stats_found:
                    # Try current year or latest available season
                    latest_season = None
                    latest_stats = None
                    for stat_group in player_data['stats']:
                        if (stat_group.get('group', '').lower() == 'hitting' and 
                            stat_group.get('type', '').lower() == 'season'):
                            ssn = stat_group.get('season')
                            if ssn and (latest_season is None or ssn > latest_season):
                                latest_season = ssn
                                latest_stats = stat_group.get('stats', {})
                    
                    if latest_stats:
                        # Extract from latest stats using same logic as above
                        for key in ['avg', 'average', 'battingAverage']:
                            if key in latest_stats and latest_stats[key] is not None:
                                try:
                                    val = float(latest_stats[key])
                                    if val >= 0:
                                        metrics['avg'] = round(val, 3)
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        if 'ops' in latest_stats and latest_stats['ops'] is not None:
                            try:
                                val = float(latest_stats['ops'])
                                if val >= 0:
                                    metrics['ops'] = round(val, 3)
                            except (ValueError, TypeError):
                                pass
                        
                        for key in ['homeRuns', 'hr']:
                            if key in latest_stats and latest_stats[key] is not None:
                                try:
                                    val = int(latest_stats[key])
                                    if val >= 0:
                                        metrics['hr'] = val
                                        break
                                except (ValueError, TypeError):
                                    continue
                        
                        for key in ['rbi', 'rbis']:
                            if key in latest_stats and latest_stats[key] is not None:
                                try:
                                    val = int(latest_stats[key])
                                    if val >= 0:
                                        metrics['rbi'] = val
                                        break
                                except (ValueError, TypeError):
                                    continue
        except Exception:
            pass
    except Exception:
        pass
    
    return metrics

def _fetch_pybaseball_stats(hitter_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch official batting stats from pybaseball (Baseball Reference) for the date range."""
    metrics = {}
    
    try:
        # Get season from start_date
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        season = start_dt.year
        
        # Try season-level stats first (more reliable than date range)
        stats_df = None
        try:
            stats_df = batting_stats(season, season)
        except Exception:
            # If season stats fail, try date range
            try:
                stats_df = batting_stats_range(start_date, end_date)
            except Exception:
                pass
        
        if stats_df is None or stats_df.empty:
            # Try using statsapi as fallback
            return _fetch_statsapi_stats(hitter_name, start_date, end_date)
        
        # Debug: Print available columns
        # print(f"Available columns: {stats_df.columns.tolist()}")
        
        # Find the player by name (try exact match first, then partial)
        hitter_name_lower = hitter_name.lower().strip()
        
        # Try to find player - check Name column (case-insensitive)
        name_col = None
        for col in stats_df.columns:
            if col.lower() == 'name':
                name_col = col
                break
        
        if name_col and name_col in stats_df.columns:
            # Try exact match
            player_row = stats_df[stats_df[name_col].astype(str).str.lower().str.strip() == hitter_name_lower]
            
            # If no exact match, try partial match (last name)
            if player_row.empty:
                name_parts = hitter_name_lower.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    player_row = stats_df[stats_df[name_col].astype(str).str.lower().str.contains(last_name, na=False)]
            
            if not player_row.empty:
                row = player_row.iloc[0]
                
                # Extract stats - check all possible column name variations
                # Batting Average
                for col in ['BA', 'Avg', 'AVG', 'Batting Average']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            metrics['avg'] = round(float(row[col]), 3)
                            break
                        except (ValueError, TypeError):
                            continue
                
                # OPS
                for col in ['OPS']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            metrics['ops'] = round(float(row[col]), 3)
                        except (ValueError, TypeError):
                            pass
                
                # OBP
                for col in ['OBP']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            metrics['obp'] = round(float(row[col]), 3)
                        except (ValueError, TypeError):
                            pass
                
                # SLG
                for col in ['SLG']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            metrics['slg'] = round(float(row[col]), 3)
                        except (ValueError, TypeError):
                            pass
                
                # Position - Baseball Reference might not have this in batting_stats_range
                # Try multiple column names
                for col in ['Pos', 'Position', 'POS', 'Pos Summary']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            pos_val = str(row[col]).strip()
                            if pos_val and pos_val.lower() != 'nan':
                                metrics['pos'] = pos_val
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Home runs
                for col in ['HR', 'Homeruns', 'Home Runs']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            hr_val = row[col]
                            if pd.notna(hr_val):
                                metrics['hr'] = int(float(hr_val))
                                break
                        except (ValueError, TypeError):
                            continue
                
                # RBIs
                for col in ['RBI', 'RBIs', 'Runs Batted In']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            rbi_val = row[col]
                            if pd.notna(rbi_val):
                                metrics['rbi'] = int(float(rbi_val))
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Calculate strikeout and walk rates (as decimals, not percentages)
                if 'PA' in row.index and 'SO' in row.index:
                    try:
                        pa = float(row['PA']) if pd.notna(row['PA']) else 0
                        so = float(row['SO']) if pd.notna(row['SO']) else 0
                        if pa > 0:
                            metrics['k_rate'] = round(so / pa, 3)
                    except (ValueError, TypeError):
                        pass
                
                if 'PA' in row.index and 'BB' in row.index:
                    try:
                        pa = float(row['PA']) if pd.notna(row['PA']) else 0
                        bb = float(row['BB']) if pd.notna(row['BB']) else 0
                        if pa > 0:
                            metrics['bb_rate'] = round(bb / pa, 3)
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        # If pybaseball fails, return empty dict
        # Don't print error to avoid cluttering output
        pass
    
    return metrics

def build_hitter_checkin(
    hitter_name: str,
    start_date: str,
    end_date: str,
    out_dir: str = "build/figures"
) -> Dict[str, Any]:
    """
    Resolve hitter ID, fetch Savant data for window, and generate mini-scouting report.
    Returns {
        "RHP": path_or_None,
        "LHP": path_or_None,
        "overall_metrics": {...},
        "rhp_metrics": {...},
        "lhp_metrics": {...},
        "overall_sw": {"strengths": [...], "weaknesses": [...]},
        "rhp_sw": {"strengths": [...], "weaknesses": [...]},
        "lhp_sw": {"strengths": [...], "weaknesses": [...]}
    }
    """
    batter_id = lookup_batter_id(hitter_name)
    df = fetch_batter_statcast(batter_id, start_date, end_date)
    
    result: Dict[str, Any] = {
        "RHP": None,
        "LHP": None,
        "overall_metrics": {},
        "rhp_metrics": {},
        "lhp_metrics": {},
        "overall_sw": {"strengths": [], "weaknesses": []},
        "rhp_sw": {"strengths": [], "weaknesses": []},
        "lhp_sw": {"strengths": [], "weaknesses": []},
        "xslg_bar": None,
        "xwoba_heatmaps": None,
        "xslg_whiff_spray": None,
        "spray_chart": None
    }
    
    if df.empty:
        return result
    
    # Check for minimum data threshold - need at least 15 batted balls for meaningful analysis
    if 'type' in df.columns:
        batted_balls = df[df['type'] == 'X']
    elif 'launch_speed' in df.columns:
        batted_balls = df[df['launch_speed'].notna()]
    else:
        batted_balls = pd.DataFrame()
    
    if len(batted_balls) < 15:
        # Too little data - return empty result (will be filtered out)
        return result
    
    # Fetch official stats from pybaseball (Baseball Reference) for traditional stats
    pybaseball_stats = _fetch_pybaseball_stats(hitter_name, start_date, end_date)
    
    # IMPORTANT: Position should ONLY come from statsapi person endpoint, never from pybaseball/statcast
    # Remove position from pybaseball_stats if it exists (it might be wrong data)
    if 'pos' in pybaseball_stats:
        del pybaseball_stats['pos']
    
    # Get position from statsapi (primary source - always correct)
    position_from_api = None
    try:
        player_id = lookup_batter_id(hitter_name)
        if player_id:
            try:
                person_data = statsapi.get('person', {'personId': player_id})
                if person_data and 'people' in person_data and len(person_data['people']) > 0:
                    person = person_data['people'][0]
                    if 'primaryPosition' in person and 'abbreviation' in person['primaryPosition']:
                        position_from_api = person['primaryPosition']['abbreviation']
            except Exception:
                pass
    except Exception:
        pass
    
    # ALWAYS try statsapi as fallback/supplement to fill any missing stats
    # This ensures we get as many stats as possible
    statsapi_stats = _fetch_statsapi_stats(hitter_name, start_date, end_date)
    
    # Merge statsapi stats into pybaseball stats (fill any gaps)
    # Priority: pybaseball > statsapi, but use statsapi if pybaseball is missing something
    for key, value in statsapi_stats.items():
        if key != 'pos':  # Position handled separately
            if key not in pybaseball_stats or pybaseball_stats[key] is None:
                if value is not None:
                    pybaseball_stats[key] = value
    
    # Set position from API (override any other source)
    if position_from_api:
        pybaseball_stats['pos'] = position_from_api
    
    # Calculate Statcast-specific metrics (EV, Hard Hit %, Barrel %, Whiff Rate, xSLG, xwOBA)
    statcast_metrics = calculate_overall_metrics(df)
    
    # Calculate xSLG and xwOBA from Statcast data
    if not batted_balls.empty:
        # xSLG: average of estimated_slg_using_speedangle for batted balls
        if 'estimated_slg_using_speedangle' in batted_balls.columns:
            xslg_values = batted_balls['estimated_slg_using_speedangle'].dropna()
            if len(xslg_values) > 0:
                statcast_metrics['xslg'] = round(float(xslg_values.mean()), 3)
        
        # xwOBA: average of estimated_woba_using_speedangle for all plate appearances
        if 'estimated_woba_using_speedangle' in df.columns:
            xwoba_values = df['estimated_woba_using_speedangle'].dropna()
            if len(xwoba_values) > 0:
                statcast_metrics['xwoba'] = round(float(xwoba_values.mean()), 3)
    
    # Use pybaseball for ALL traditional stats - these are the source of truth
    # Only use statcast for advanced metrics that pybaseball doesn't provide
    result["overall_metrics"] = {}
    
    # Traditional stats keys that should ONLY come from pybaseball (never statcast)
    traditional_stats = ["avg", "ops", "obp", "slg", "k_rate", "bb_rate", "pos", "hr", "rbi"]
    
    # Always add ALL pybaseball stats first (including pos, hr, rbi, avg, ops)
    for key, value in pybaseball_stats.items():
        if value is not None:
            result["overall_metrics"][key] = value
    
    # statsapi_stats was already fetched above, so just use it here
    statsapi_fallback = statsapi_stats
    # Fill in any missing stats from statsapi (already merged above, but double-check)
    for key in ['avg', 'ops', 'hr', 'rbi']:
        if key not in result["overall_metrics"] or result["overall_metrics"][key] is None:
            if key in statsapi_fallback and statsapi_fallback[key] is not None:
                result["overall_metrics"][key] = statsapi_fallback[key]
    
    # Only use statcast as fallback for traditional stats if both pybaseball and statsapi failed
    # But ONLY for stats that pybaseball should have (not pos, hr, rbi)
    if not pybaseball_stats and not statsapi_fallback:
        for key in ["avg", "ops", "obp", "slg", "k_rate", "bb_rate"]:
            if key in statcast_metrics and statcast_metrics[key] is not None:
                result["overall_metrics"][key] = statcast_metrics[key]
    
    # Always add statcast advanced metrics (EV, hard hit, barrel, whiff, xSLG, xwOBA) - these are statcast-only
    for key in ["avg_ev", "avg_la", "hard_hit_pct", "barrel_pct", "whiff_pct", "xslg", "xwoba"]:
        if key in statcast_metrics and statcast_metrics[key] is not None:
            result["overall_metrics"][key] = statcast_metrics[key]
    
    result["overall_sw"] = identify_strengths_weaknesses(df)
    
    # Generate split metrics (for splits, we'll need to calculate from Statcast data)
    if 'p_throws' in df.columns:
        rhp_df = df[df['p_throws'] == 'R'].copy()
        lhp_df = df[df['p_throws'] == 'L'].copy()
        
        if not rhp_df.empty:
            result["rhp_metrics"] = calculate_overall_metrics(rhp_df)
            result["rhp_sw"] = identify_strengths_weaknesses(rhp_df)
        
        if not lhp_df.empty:
            result["lhp_metrics"] = calculate_overall_metrics(lhp_df)
            result["lhp_sw"] = identify_strengths_weaknesses(lhp_df)
    
    # Generate tables (keep existing functionality)
    tables = render_hitter_checkin_pngs(df, hitter_name, out_dir=out_dir)
    result["RHP"] = tables.get("RHP")
    result["LHP"] = tables.get("LHP")
    
    # Helper function to convert path to URI
    def _to_uri(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return Path(p).resolve().as_uri()
    
    # Generate strengths/weaknesses visual
    try:
        sw_visual_path = Path(out_dir) / f"{hitter_name.replace(' ','_')}_sw_visual.png"
        sw_visual = create_strengths_weaknesses_visual(df, hitter_name, sw_visual_path)
        if sw_visual and Path(sw_visual).exists():
            result["sw_visual_url"] = _to_uri(sw_visual)
    except Exception as e:
        pass
    
    # Generate advanced visualizations
    # Ensure out_dir exists
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        xslg_bar_path = render_xslg_by_pitch_type(df, hitter_name, out_dir=out_dir)
        if xslg_bar_path:
            xslg_path = Path(xslg_bar_path)
            if xslg_path.exists():
                result["xslg_bar"] = _to_uri(str(xslg_path))
    except Exception as e:
        # Visualization generation failed - continue without it
        pass
    
    try:
        xwoba_heatmaps_path = render_xwoba_location_heatmaps(df, hitter_name, out_dir=out_dir)
        if xwoba_heatmaps_path:
            xwoba_path = Path(xwoba_heatmaps_path)
            if xwoba_path.exists():
                result["xwoba_heatmaps"] = _to_uri(str(xwoba_path))
    except Exception as e:
        pass
    
    try:
        xslg_whiff_spray_path = render_xslg_whiff_spray(df, hitter_name, out_dir=out_dir)
        if xslg_whiff_spray_path:
            xslg_whiff_path = Path(xslg_whiff_spray_path)
            if xslg_whiff_path.exists():
                result["xslg_whiff_spray"] = _to_uri(str(xslg_whiff_path))
    except Exception as e:
        pass
    
    try:
        spray_chart_path = render_spray_chart(df, hitter_name, out_dir=out_dir)
        if spray_chart_path:
            spray_path = Path(spray_chart_path)
            if spray_path.exists():
                result["spray_chart"] = _to_uri(str(spray_path))
    except Exception as e:
        pass
    
    return result
