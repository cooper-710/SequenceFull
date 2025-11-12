
from __future__ import annotations

import argparse, io, json, re, urllib.request, os, uuid, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Suppress urllib3 warnings about OpenSSL - must be done before importing httpx/urllib3
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*OpenSSL.*')

import statsapi
import requests
from render import render_html_to_pdf
from next_opponent import next_game_info, next_series_game_info
from optimize_assets import optimize_dir
from scrape_savant import lookup_batter_id, fetch_pitcher_statcast
from build_movement import build_movement_for_pitcher
from build_bar_by_count import build_pitch_mix_by_count_for_pitcher
from build_pitch_tables import build_pitch_tables_for_pitcher, build_pitch_table_combined_for_pitcher
from build_hitter_checkin import build_hitter_checkin

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow should be available but fail gracefully if not
    Image = None

try:
    from build_heatmaps import build_heatmaps_for_pitcher
except Exception:
    def build_heatmaps_for_pitcher(*args, **kwargs):
        return None

REDEEM_URL_DEFAULT = "https://cooper-710.github.io/Redeem-Token/"
TOKEN_ENDPOINTS = [
    "https://token-generator.mv66946f7j.workers.dev/",
    "https://token-generator.mv66946f7j.workers.dev",
]

REQUEST_HEADERS = {
    "User-Agent": "SequenceBioLab/1.0 (+https://sequencebiolab.com)"
}
PITCH_NAME_MAP = {
    "FF":"Four-Seam","FA":"Four-Seam","FT":"Sinker","SI":"Sinker","ST":"Sweeper",
    "SL":"Slider","FC":"Cutter","CT":"Cutter","CH":"Changeup","CU":"Curveball",
    "KC":"Knuckle Curve","CS":"Slurve","FS":"Splitter","FO":"Forkball",
    "SF":"Split-Finger","KN":"Knuckleball","EP":"Eephus",
}

def _to_uri(p: Optional[str]) -> Optional[str]:
    return None if not p else Path(p).resolve().as_uri()


def _whiten_background(img: "Image.Image", threshold: int = 235) -> "Image.Image":
    """
    Lift near-white pixels (background greys) to pure white so the headshot
    blends cleanly with the report background.
    """
    if Image is None:
        return img

    try:
        data = img.load()
        width, height = img.size
        for y in range(height):
            for x in range(width):
                r, g, b = data[x, y]
                if r >= threshold and g >= threshold and b >= threshold:
                    data[x, y] = (255, 255, 255)
    except Exception:
        pass
    return img


def _write_headshot_image(headshot_file: Path, image_bytes: bytes) -> None:
    """
    Persist fetched headshots while stripping transparency so Chrome's PDF
    renderer doesn't introduce visible artifacts.
    """
    if not image_bytes:
        return

    if Image is None:
        headshot_file.write_bytes(image_bytes)
        return

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            has_alpha = (
                ("A" in img.getbands())
                or (img.mode == "P" and "transparency" in img.info)
            )
            if has_alpha:
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.alpha_composite(img)
                processed = background.convert("RGB")
            else:
                processed = img.convert("RGB")

            processed = _whiten_background(processed)
            processed.save(headshot_file, format="PNG", optimize=True)
    except Exception:
        headshot_file.write_bytes(image_bytes)


def _sanitize_cached_headshot(headshot_file: Path) -> None:
    """
    Re-process previously cached headshots so they benefit from the transparency
    fix without requiring a manual directory clear.
    """
    if not headshot_file.exists() or Image is None:
        return

    try:
        with Image.open(headshot_file) as img:
            has_alpha = (
                ("A" in img.getbands())
                or (img.mode == "P" and "transparency" in img.info)
            )
    except Exception:
        return

    if not has_alpha:
        return

    try:
        original_bytes = headshot_file.read_bytes()
    except Exception:
        return

    _write_headshot_image(headshot_file, original_bytes)


def _headshot_needs_refetch(headshot_file: Path) -> bool:
    """
    Detect legacy cached headshots that were stored as JPEGs with a PNG
    extension, which introduced visible compression artifacts.
    """
    try:
        with headshot_file.open("rb") as fh:
            signature = fh.read(4)
    except Exception:
        return True

    # PNG files start with 0x89 50 4E 47. Anything else should be refreshed.
    return signature != b"\x89PNG"

# Team abbreviation to MLB team ID mapping (for logos)
TEAM_ABBR_TO_ID = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145, "CIN": 113,
    "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119,
    "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133, "PHI": 143,
    "PIT": 134, "SD": 135, "SF": 137, "SEA": 136, "STL": 138, "TB": 139, "TEX": 140,
    "TOR": 141, "WSH": 120,
    # Alternative abbreviations
    "LAA": 108, "ANA": 108, "CHW": 145, "KCR": 118, "SDP": 135, "SFG": 137, "TBR": 139,
    "TEX": 140, "WSN": 120, "WAS": 120,
}

def _team_id_to_abbr(team_id: int) -> Optional[str]:
    """Get team abbreviation from team ID."""
    try:
        teams = statsapi.get('teams', {'sportId': 1})['teams']
        for team in teams:
            if team['id'] == team_id:
                return team.get('fileCode') or team.get('abbreviation')
    except Exception:
        pass
    return None

def _get_team_name(team_abbr: Optional[str] = None, team_id: Optional[int] = None) -> Optional[str]:
    """Get full team name from abbreviation or ID."""
    try:
        teams = statsapi.get('teams', {'sportId': 1})['teams']
        for team in teams:
            if team_id and team['id'] == team_id:
                return team.get('name') or team.get('teamName')
            if team_abbr:
                team_abbr_upper = team_abbr.upper()
                if (team.get('fileCode', '').upper() == team_abbr_upper or 
                    team.get('abbreviation', '').upper() == team_abbr_upper):
                    return team.get('name') or team.get('teamName')
    except Exception:
        pass
    return None

def _fetch_team_logo(team_abbr: Optional[str] = None, team_id: Optional[int] = None, out_dir: str = "build/logos") -> Optional[str]:
    """
    Fetch and cache team logo from MLB static assets.
    
    Args:
        team_abbr: Team abbreviation (e.g., "NYY", "LAD")
        team_id: Team ID (alternative to team_abbr)
        out_dir: Directory to cache logos
        
    Returns:
        Local file path as URI, or None if fetch fails
    """
    # Get team ID and abbreviation
    if team_id and not team_abbr:
        team_abbr = _team_id_to_abbr(team_id)
    
    if not team_abbr and not team_id:
        return None
    
    if not team_id:
        team_abbr_upper = team_abbr.upper()
        team_id = TEAM_ABBR_TO_ID.get(team_abbr_upper)
        
        if not team_id:
            # Try to get team ID from statsapi
            try:
                teams = statsapi.get('teams', {'sportId': 1})['teams']
                for team in teams:
                    if (team.get('fileCode', '').upper() == team_abbr_upper or 
                        team.get('abbreviation', '').upper() == team_abbr_upper):
                        team_id = team['id']
                        break
            except Exception:
                pass
    
    if not team_id:
        return None
    
    # Create output directory
    logo_dir = Path(out_dir)
    logo_dir.mkdir(parents=True, exist_ok=True)
    
    # Logo file path
    logo_file = logo_dir / f"{team_id}.svg"
    
    # If already cached, return it
    if logo_file.exists():
        return _to_uri(str(logo_file))
    
    # Try to fetch logo from MLB static assets
    logo_urls = [
        f"https://www.mlbstatic.com/team-logos/{team_id}.svg",
        f"https://a.espncdn.com/i/teamlogos/mlb/500/{team_id}.png",
        f"https://content.mlb.com/images/headshots/current/60x60/{team_id}.png",
    ]
    
    for url in logo_urls:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
            if resp.status_code == 200 and 'image' in (resp.headers.get('Content-Type', '') or '').lower():
                logo_file.write_bytes(resp.content)
                return _to_uri(str(logo_file))
        except requests.RequestException:
            continue
        except Exception:
            continue
    
    return None

def _fetch_player_headshot(player_name: str, out_dir: str = "build/headshots") -> Optional[str]:
    """
    Fetch and cache player headshot from MLB static assets.
    
    Args:
        player_name: Player name (e.g., "Gerrit Cole")
        out_dir: Directory to cache headshots
        
    Returns:
        Local file path as URI, or None if fetch fails
    """
    if not player_name:
        return None
    
    # Get player ID - for pitchers, lookup_batter_id works because it searches all players
    try:
        player_id = lookup_batter_id(player_name)
        if not player_id:
            return None
    except Exception:
        return None
    
    # Create output directory
    headshot_dir = Path(out_dir)
    headshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Headshot file path (use player ID as filename)
    headshot_file = headshot_dir / f"{player_id}.png"
    
    # If already cached, return it
    if headshot_file.exists():
        if _headshot_needs_refetch(headshot_file):
            try:
                headshot_file.unlink()
            except Exception:
                pass
        else:
            _sanitize_cached_headshot(headshot_file)
            return _to_uri(str(headshot_file))
    
    # Try to fetch headshot from MLB static assets
    # Multiple URL patterns to try
    headshot_urls = [
        f"https://img.mlbstatic.com/mlb-photos/image/upload/f_png,w_213,q_100/v1/people/{player_id}/headshot/67/current",
        f"https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/f_png,w_213,q_100/v1/people/{player_id}/headshot/67/current",
        f"https://content.mlb.com/images/headshots/current/60x60/{player_id}.png",
        f"https://content.mlb.com/images/headshots/current/213x213/{player_id}.png",
        f"https://securea.mlb.com/mlb/images/players/head_shot/{player_id}.jpg",
    ]
    
    for url in headshot_urls:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
            if resp.status_code == 200:
                content_type = (resp.headers.get('Content-Type', '') or '').lower()
                if 'image' in content_type:
                    _write_headshot_image(headshot_file, resp.content)
                    return _to_uri(str(headshot_file))
        except requests.RequestException:
            continue
        except Exception:
            continue
    
    return None

def _fetch_token() -> Optional[str]:
    """Fetch token from endpoints or generate new one."""
    import urllib.parse
    for endpoint in TOKEN_ENDPOINTS:
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                raw = response.read().decode('utf-8')
                import re
                m = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', raw)
                if m: return m.group(0)
        except Exception:
            pass
    tdir = Path("build"); tdir.mkdir(parents=True, exist_ok=True)
    tf = tdir/"token.txt"
    try:
        v = tf.read_text().strip()
        if v: return v
    except Exception:
        pass
    v = str(uuid.uuid4()); tf.write_text(v); return v

def _normalize_redeem_url(url: Optional[str]) -> str:
    """Normalize redeem URL."""
    if not url: return REDEEM_URL_DEFAULT
    url = url.strip()
    if not url.endswith('/'): url += '/'
    return url

def _pitch_labels_for_pitcher(pid: int, start: str, end: str, stand: Optional[str]=None) -> List[str]:
    try:
        df = fetch_pitcher_statcast(pid, start_date=start, end_date=end)
        if df is None or df.empty or "pitch_type" not in df.columns: return []
        if stand in ('R','L'): df = df[df['stand']==stand]
        order = df["pitch_type"].dropna().astype(str).value_counts().index.tolist()
        return [PITCH_NAME_MAP.get(k, k) for k in order]
    except Exception:
        return []

def _opponent_active_hitters(opponent_id: int) -> List[Dict]:
    """Get active position players (non-pitchers) from opponent team."""
    try:
        ro = statsapi.get("team_roster", {"teamId": opponent_id, "rosterType": "active"})
        out = []
        for r in ro.get("roster", []):
            # Filter for position players (not pitchers)
            position_type = str((r.get("position") or {}).get("type","")).lower()
            if position_type != "pitcher":
                out.append({"id": int(r["person"]["id"]), "name": r["person"]["fullName"]})
        return out
    except Exception:
        return []

def _fetch_pitcher_stats(pitcher_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch pitcher stats: age, W-L, IP, ERA, WHIP, Ks using statsapi (primary) and pybaseball (fallback)."""
    stats = {
        "age": None,
        "wins": None,
        "losses": None,
        "ip": None,
        "era": None,
        "whip": None,
        "strikeouts": None
    }
    
    try:
        from scrape_savant import lookup_batter_id
        from datetime import datetime
        
        # Get season from start_date
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        season = start_dt.year
        
        # PRIMARY: Use statsapi (most reliable)
        try:
            player_id = lookup_batter_id(pitcher_name)
            if player_id:
                # Get pitcher stats from statsapi
                player_data = statsapi.player_stat_data(player_id, group='[pitching]', type='season', season=season)
                
                if player_data and 'stats' in player_data:
                    # Find pitching stats for the season
                    for stat_group in player_data['stats']:
                        if (stat_group.get('group', '').lower() == 'pitching' and 
                            stat_group.get('type', '').lower() == 'season' and
                            stat_group.get('season') == season):
                            
                            pitching_stats_data = stat_group.get('stats', {})
                            
                            # Wins
                            for key in ['wins', 'w']:
                                if key in pitching_stats_data and pitching_stats_data[key] is not None:
                                    try:
                                        stats['wins'] = int(pitching_stats_data[key])
                                        break
                                    except (ValueError, TypeError):
                                        continue
                            
                            # Losses
                            for key in ['losses', 'l']:
                                if key in pitching_stats_data and pitching_stats_data[key] is not None:
                                    try:
                                        stats['losses'] = int(pitching_stats_data[key])
                                        break
                                    except (ValueError, TypeError):
                                        continue
                            
                            # IP (innings pitched) - can be string like "187.2"
                            for key in ['inningsPitched', 'ip', 'innings', 'inningsPitched']:
                                if key in pitching_stats_data and pitching_stats_data[key] is not None:
                                    try:
                                        ip_val = pitching_stats_data[key]
                                        if isinstance(ip_val, str):
                                            # Handle fractional innings like "187.2"
                                            ip_val = ip_val.replace(' ', '')
                                        stats['ip'] = float(ip_val)
                                        break
                                    except (ValueError, TypeError):
                                        continue
                            
                            # ERA
                            if 'era' in pitching_stats_data and pitching_stats_data['era'] is not None:
                                try:
                                    stats['era'] = float(pitching_stats_data['era'])
                                except (ValueError, TypeError):
                                    pass
                            
                            # WHIP - calculate if not present
                            if 'whip' in pitching_stats_data and pitching_stats_data['whip'] is not None:
                                try:
                                    stats['whip'] = float(pitching_stats_data['whip'])
                                except (ValueError, TypeError):
                                    pass
                            else:
                                # Calculate WHIP: (BB + H) / IP
                                try:
                                    bb = pitching_stats_data.get('baseOnBalls') or pitching_stats_data.get('walks') or pitching_stats_data.get('bb', 0)
                                    h = pitching_stats_data.get('hits') or pitching_stats_data.get('h', 0)
                                    ip = stats.get('ip') or pitching_stats_data.get('inningsPitched') or pitching_stats_data.get('ip', 0)
                                    if ip and ip > 0:
                                        stats['whip'] = round((float(bb) + float(h)) / float(ip), 2)
                                except (ValueError, TypeError, ZeroDivisionError):
                                    pass
                            
                            # Strikeouts
                            for key in ['strikeOuts', 'strikeouts', 'so', 'k', 'strikeOuts', 'strikeouts']:
                                if key in pitching_stats_data and pitching_stats_data[key] is not None:
                                    try:
                                        stats['strikeouts'] = int(pitching_stats_data[key])
                                        break
                                    except (ValueError, TypeError):
                                        continue
                            
                            break
                
                # Get age from person data (separate call)
                try:
                    person_data = statsapi.get('person', {'personId': player_id})
                    if person_data and 'people' in person_data and len(person_data['people']) > 0:
                        # Calculate age from birth date
                        birth_date = person_data['people'][0].get('birthDate')
                        if birth_date:
                            birth = datetime.strptime(birth_date, '%Y-%m-%d')
                            today = datetime.now()
                            age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
                            stats['age'] = age
                except Exception:
                    pass
        except Exception:
            pass
        
        # FALLBACK: Try pybaseball if statsapi didn't work
        if not any(v is not None for v in [stats['wins'], stats['ip'], stats['era']]):
            try:
                from pybaseball import pitching_stats
                import pandas as pd
                
                stats_df = pitching_stats(season, season)
                
                if stats_df is not None and not stats_df.empty:
                    pitcher_name_lower = pitcher_name.lower().strip()
                    
                    if 'Name' in stats_df.columns:
                        player_row = stats_df[stats_df['Name'].astype(str).str.lower().str.strip() == pitcher_name_lower]
                        
                        if player_row.empty:
                            name_parts = pitcher_name_lower.split()
                            if len(name_parts) >= 2:
                                last_name = name_parts[-1]
                                player_row = stats_df[stats_df['Name'].astype(str).str.lower().str.contains(last_name, na=False)]
                        
                        if not player_row.empty:
                            row = player_row.iloc[0]
                            
                            if stats['wins'] is None and 'W' in row.index and pd.notna(row['W']):
                                try:
                                    stats['wins'] = int(float(row['W']))
                                except (ValueError, TypeError):
                                    pass
                            
                            if stats['losses'] is None and 'L' in row.index and pd.notna(row['L']):
                                try:
                                    stats['losses'] = int(float(row['L']))
                                except (ValueError, TypeError):
                                    pass
                            
                            if stats['ip'] is None and 'IP' in row.index and pd.notna(row['IP']):
                                try:
                                    stats['ip'] = float(row['IP'])
                                except (ValueError, TypeError):
                                    pass
                            
                            if stats['era'] is None and 'ERA' in row.index and pd.notna(row['ERA']):
                                try:
                                    stats['era'] = float(row['ERA'])
                                except (ValueError, TypeError):
                                    pass
                            
                            if stats['whip'] is None and 'WHIP' in row.index and pd.notna(row['WHIP']):
                                try:
                                    stats['whip'] = float(row['WHIP'])
                                except (ValueError, TypeError):
                                    pass
                            
                            if stats['strikeouts'] is None and 'SO' in row.index and pd.notna(row['SO']):
                                try:
                                    stats['strikeouts'] = int(float(row['SO']))
                                except (ValueError, TypeError):
                                    pass
                            
                            if stats['age'] is None and 'Age' in row.index and pd.notna(row['Age']):
                                try:
                                    stats['age'] = int(float(row['Age']))
                                except (ValueError, TypeError):
                                    pass
            except Exception:
                pass
    except Exception:
        pass
    
    return stats

def pitcher_handedness_from_savant(name: str, start: str, end: str) -> str:
    """Get pitcher's throwing hand from statcast data."""
    try:
        # Use lookup_batter_id which works for all players
        pid = lookup_batter_id(name)
        if not pid:
            return "R"
        df = fetch_pitcher_statcast(pid, start, end)
        if df is not None and not df.empty and "p_throws" in df.columns:
            throws_values = df["p_throws"].dropna()
            if not throws_values.empty:
                throws_mode = throws_values.mode()
                if not throws_mode.empty:
                    return str(throws_mode.iloc[0]).upper()
    except Exception:
        pass
    return "R"

def _build_all_for_hitter(hitter_dict: Dict, season_start: str, game_date: str) -> Dict:
    """Build hitter checkin data for an opponent hitter."""
    hid, hname = int(hitter_dict["id"]), hitter_dict["name"]
    try:
        hc = build_hitter_checkin(hname, season_start, game_date, out_dir="build/figures")
    except Exception:
        hc = {
            "RHP": None,
            "LHP": None,
            "overall_metrics": {},
            "rhp_metrics": {},
            "lhp_metrics": {},
            "overall_sw": {"strengths": [], "weaknesses": []},
            "rhp_sw": {"strengths": [], "weaknesses": []},
            "lhp_sw": {"strengths": [], "weaknesses": []},
        }
    
    # Fetch hitter headshot
    headshot_url = None
    try:
        headshot_url = _fetch_player_headshot(hname)
    except Exception:
        pass
    
    return {
        "name": hname,
        "checkin": hc,
        "headshot_url": headshot_url,
    }

def _collect_hitter_assets_for_lineup(opponent_id: int, season_start: str, game_date: str, workers: int) -> Dict:
    """Collect hitter checkin data for all active hitters on opponent team."""
    assets = {"oppo_hitter_checkins": []}
    hitters = _opponent_active_hitters(opponent_id)
    if not hitters:
        return assets
    
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_build_all_for_hitter, h, season_start, game_date) for h in hitters]
        for f in as_completed(futs):
            try:
                res = f.result()
                hname = res["name"]
                checkin = res.get("checkin", {})
                
                # Check if hitter has minimum data (at least 15 batted balls)
                # This is checked in build_hitter_checkin, but verify here too
                # Also check if hitter has at least one valid checkin visualization
                has_rhp = checkin.get("RHP") is not None and checkin.get("RHP") != ""
                has_lhp = checkin.get("LHP") is not None and checkin.get("LHP") != ""
                has_advanced_viz = (checkin.get("xslg_bar") or checkin.get("xwoba_heatmaps") or 
                                   checkin.get("xslg_whiff_spray") or checkin.get("spray_chart"))
                
                # Only include hitter if they have at least one valid visualization AND enough data
                # The build_hitter_checkin function already filters for minimum data, so if checkin is empty
                # or has no visualizations, it means insufficient data
                if (has_rhp or has_lhp or has_advanced_viz) and checkin.get("overall_metrics"):
                    # Convert paths to URIs - checkin already has URIs from build_hitter_checkin
                    # But we need to ensure they're properly formatted
                    checkin_copy = res["checkin"].copy()
                    # These should already be URIs from build_hitter_checkin, but verify
                    for key in ["RHP", "LHP", "sw_visual_url", "xslg_bar", "xwoba_heatmaps", "xslg_whiff_spray", "spray_chart"]:
                        if checkin_copy.get(key) and not checkin_copy[key].startswith("file://"):
                            # If it's a path, convert to URI
                            if isinstance(checkin_copy[key], str) and Path(checkin_copy[key]).exists():
                                checkin_copy[key] = _to_uri(checkin_copy[key])
                    
                    assets["oppo_hitter_checkins"].append({
                        "name": hname,
                        "checkin": checkin_copy,
                        "headshot_url": res.get("headshot_url"),
                    })
            except Exception:
                continue
    
    return assets

def _get_opponent_info(team_abbr: str, opponent_team: Optional[str] = None, use_next_series: bool = False) -> Dict:
    """Get opponent information - either from next game/series or manually specified opponent."""
    if opponent_team:
        # User specified an opponent - find the next game between these teams
        from next_opponent import next_games, _resolve_team_id
        team_id = _resolve_team_id(team_abbr)
        opponent_id = _resolve_team_id(opponent_team)
        
        # Find games between these teams
        games = next_games(team_abbr, days_ahead=30, include_started=False)
        matching_games = [g for g in games if g["opponent_id"] == opponent_id]
        
        if matching_games:
            return matching_games[0]
        else:
            # No upcoming game found, create a synthetic opponent info
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).date().isoformat()
            return {
                "game_date": today,
                "opponent_name": opponent_team,
                "opponent_id": opponent_id,
                "home_id": team_id,
                "away_id": opponent_id,
                "home_name": team_abbr,
                "away_name": opponent_team,
            }
    else:
        # Use automatic opponent detection
        opp = next_series_game_info(team_abbr) if use_next_series else next_game_info(team_abbr)
        return opp

def build_pitcher_context(team_abbr: str, pitcher_name: str, season_start: str, use_next_series: bool=False, opponent_team: Optional[str]=None, token: Optional[str]=None, redeem_url: Optional[str]=None, pdf_date: Optional[str]=None, workers: int=4) -> Dict:
    """Build context dictionary for pitcher report generation."""
    opp = _get_opponent_info(team_abbr, opponent_team, use_next_series)
    game_date, opponent_name, opponent_id = opp["game_date"], opp["opponent_name"], opp["opponent_id"]
    tok = token or _fetch_token() or ""
    red = _normalize_redeem_url(redeem_url or REDEEM_URL_DEFAULT)
    
    # Fetch team logos and pitcher headshot (non-blocking, graceful failure)
    team_logo_url = None
    opponent_logo_url = None
    player_headshot_url = None
    team_name_full = None
    opponent_name_full = None
    try:
        team_logo_url = _fetch_team_logo(team_abbr=team_abbr)
        team_name_full = _get_team_name(team_abbr=team_abbr)
    except Exception:
        pass
    try:
        opponent_logo_url = _fetch_team_logo(team_id=opponent_id)
        opponent_name_full = _get_team_name(team_id=opponent_id)
    except Exception:
        pass
    try:
        player_headshot_url = _fetch_player_headshot(pitcher_name)
    except Exception:
        pass
    
    # Get pitcher ID and build pitcher's own analytics
    pitcher_id = None
    try:
        pitcher_id = lookup_batter_id(pitcher_name)
    except Exception:
        pass
    
    # Get pitcher's throwing hand
    pitcher_handedness = pitcher_handedness_from_savant(pitcher_name, season_start, game_date)
    want_side = 'RHH' if pitcher_handedness == 'R' else 'LHH'
    
    # Build pitcher's own analytics
    pitcher_figures = {}
    if pitcher_id:
        try:
            # Movement plot
            pitcher_figures["movement"] = build_movement_for_pitcher(
                pitcher_id, pitcher_name, season_start, game_date, 
                out_dir="build/figures", include_density=True, normalize_by_throws=False
            )
            if pitcher_figures["movement"]:
                pitcher_figures["movement"] = _to_uri(pitcher_figures["movement"])
        except Exception:
            pitcher_figures["movement"] = None
        
        try:
            # Heatmaps
            pitcher_figures["heatmaps"] = build_heatmaps_for_pitcher(
                pitcher_id, pitcher_name, season_start, game_date, 
                out_dir="build/figures", stand=None
            )
            if pitcher_figures["heatmaps"]:
                pitcher_figures["heatmaps"] = _to_uri(pitcher_figures["heatmaps"])
                # Get pitch labels
                pitcher_figures["pitch_labels"] = _pitch_labels_for_pitcher(pitcher_id, season_start, game_date)
        except Exception:
            pitcher_figures["heatmaps"] = None
            pitcher_figures["pitch_labels"] = []
        
        try:
            # Combined pitch table (all pitches, not split by batter handedness)
            pitcher_figures["pitch_table_combined"] = build_pitch_table_combined_for_pitcher(
                pitcher_id, pitcher_name, season_start, game_date, out_dir="build/figures"
            )
            if pitcher_figures["pitch_table_combined"]:
                pitcher_figures["pitch_table_combined"] = _to_uri(pitcher_figures["pitch_table_combined"])
        except Exception:
            pitcher_figures["pitch_table_combined"] = None
        
        try:
            # Pitch mix by count
            pitcher_figures["pitch_mix"] = build_pitch_mix_by_count_for_pitcher(
                pitcher_id, pitcher_name, season_start, game_date, 
                out_dir="build/figures", logo_path=None, stand=None
            )
            if pitcher_figures["pitch_mix"]:
                pitcher_figures["pitch_mix"] = _to_uri(pitcher_figures["pitch_mix"])
        except Exception:
            pitcher_figures["pitch_mix"] = None
    
    # Fetch pitcher stats
    pitcher_stats = {}
    try:
        pitcher_stats = _fetch_pitcher_stats(pitcher_name, season_start, game_date)
    except Exception:
        pass
    
    # Get pitcher handedness from MLB API if not in stats
    if "handedness" not in pitcher_stats:
        try:
            if pitcher_id:
                person_data = statsapi.get("people", {"personIds": pitcher_id})
                if person_data and "people" in person_data and len(person_data["people"]) > 0:
                    pitch_hand = person_data["people"][0].get("pitchHand", {})
                    if pitch_hand:
                        hand_code = pitch_hand.get("code", "").upper()
                        if hand_code == "L":
                            pitcher_stats["handedness"] = "LHP"
                        elif hand_code == "R":
                            pitcher_stats["handedness"] = "RHP"
        except Exception:
            pass
    
    # Collect opponent hitter assets
    hitter_assets = _collect_hitter_assets_for_lineup(opponent_id, season_start, game_date, workers)
    
    ctx: Dict = {
        "player": pitcher_name,
        "opponent": opponent_name,
        "team_name": team_name_full or team_abbr,
        "opponent_name": opponent_name_full or opponent_name,
        "date": pdf_date or game_date,
        "token": tok,
        "redeem_url": red,
        "game_date": game_date,
        "season_start": season_start,
        "team_logo_url": team_logo_url,
        "opponent_logo_url": opponent_logo_url,
        "player_headshot_url": player_headshot_url,
        "pitcher_stats": pitcher_stats,
        "figures": {
            **pitcher_figures,
            **hitter_assets,
        },
        "opponent_info": opp,
    }
    
    return ctx

def _safe_filename(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "", s).strip()

def _resolve_team_for_pitcher(pitcher_name: str) -> Optional[str]:
    """Resolve team abbreviation for a pitcher by name."""
    try:
        candidates = statsapi.lookup_player(pitcher_name) or []
        pit = next((m for m in candidates if str(m.get("fullName","")).lower()==pitcher_name.lower()), candidates[0] if candidates else None)
        if not pit: return None
        pid = int(pit["id"])
        try:
            people = statsapi.get("people", {"personIds": pid}).get("people", [])
            ct = (people[0].get("currentTeam") or {}) if people else {}
            if ct.get("id"):
                info = statsapi.get("teams", {"teamId": int(ct["id"])})
                t = (info.get("teams") or [])[0]
                abbr = (t.get("fileCode") or "").upper()
                if abbr: return abbr
        except Exception:
            pass
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser(description="Generate automated pitcher scouting PDF.")
    ap.add_argument("--team", required=True)
    ap.add_argument("--pitcher", required=True)
    ap.add_argument("--season_start", default="2025-03-20")
    ap.add_argument("--opponent", default=None, help="Opponent team abbreviation (optional, auto-detects if not provided)")
    ap.add_argument("--token", default=None)
    ap.add_argument("--redeem_url", default=None)
    ap.add_argument("--out", default="build/pdf")
    ap.add_argument("--template", default="templates/pitcher_report.html")
    ap.add_argument("--pdf_name", default=None)
    ap.add_argument("--pdf_date", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-next-series", action="store_true")
    args = ap.parse_args()

    team_abbr = (args.team or "").upper()
    if team_abbr in ("", "AUTO"):
        auto_team = _resolve_team_for_pitcher(args.pitcher)
        team_abbr = auto_team or "NYM"

    opponent_team = args.opponent.upper() if args.opponent else None

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ctx = build_pitcher_context(
            team_abbr=team_abbr, 
            pitcher_name=args.pitcher, 
            season_start=args.season_start, 
            use_next_series=args.use_next_series, 
            opponent_team=opponent_team, 
            token=args.token, 
            redeem_url=args.redeem_url, 
            pdf_date=args.pdf_date, 
            workers=max(1, args.workers)
        )

        opp_for_name = ctx.get("opponent") or ctx.get("opponent_info", {}).get("opponent_name") or "Unknown"
        pdf_name = args.pdf_name or f"{args.pitcher} vs {opp_for_name}.pdf"
        out_pdf = out_dir / _safe_filename(pdf_name)

        optimize_dir("build/figures")
        render_html_to_pdf(args.template, ctx, out_pdf)
        print(f"Saved report: {out_pdf}")
    except Exception as e:
        import traceback
        import sys
        # Print full traceback to stderr
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise  # Re-raise to ensure non-zero exit code

if __name__ == "__main__":
    main()

