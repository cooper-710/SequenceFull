
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
from scrape_savant import lookup_batter_id, fetch_batter_statcast, fetch_pitcher_statcast
from build_movement import build_movement_for_pitcher
from build_bar_by_count import build_pitch_mix_by_count_for_pitcher
from build_pitch_tables import build_pitch_tables_for_pitcher

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
    Persist a fetched headshot to disk while stripping alpha channels that can
    introduce rendering artifacts in Chrome's PDF output.
    """
    if not image_bytes:
        return

    if Image is None:
        headshot_file.write_bytes(image_bytes)
        return

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Detect transparency and composite onto white to avoid speckling
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
    Re-process an existing cached headshot to ensure it does not contain an
    alpha channel. This lets previously downloaded images benefit from the
    transparency fix without requiring a manual cache clear.
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
                        team_abbr = team.get('fileCode') or team.get('abbreviation')
                        break
            except Exception:
                pass
    
    if not team_id:
        return None
    
    team_abbr_upper = (team_abbr or str(team_id)).upper()
    
    # Create output directory
    logo_dir = Path(out_dir)
    logo_dir.mkdir(parents=True, exist_ok=True)
    
    # Logo file path
    logo_file = logo_dir / f"{team_abbr_upper}.svg"
    
    # If already cached, return it
    if logo_file.exists():
        return _to_uri(str(logo_file))
    
    # Try to fetch logo from MLB static assets
    logo_urls = [
        f"https://www.mlbstatic.com/team-logos/{team_id}.svg",
        f"https://www.mlbstatic.com/team-logos/{team_id}_light.svg",
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
        player_name: Player name (e.g., "Pete Alonso")
        out_dir: Directory to cache headshots
        
    Returns:
        Local file path as URI, or None if fetch fails
    """
    if not player_name:
        return None
    
    # Get player ID
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

def _normalize_redeem_url(u: Optional[str]) -> str:
    if not u: return REDEEM_URL_DEFAULT
    s = str(u).strip().replace("\u2014","-").replace("\u2013","-").replace("\u2212","-").replace("—","-").replace("–","-")
    if not s.startswith(("http://","https://")): s = "https://" + s.lstrip("/")
    if not s.endswith("/"): s += "/"
    return s

def _fetch_token() -> Optional[str]:
    env = os.environ.get("SEQUENCE_TOKEN", "").strip()
    if env: return env
    for url in TOKEN_ENDPOINTS:
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                raw = r.read().decode("utf-8", "ignore").strip()
            if raw.startswith("{"):
                tok = (json.loads(raw).get("token") or "").strip()
                if tok: return tok
            m = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", raw)
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

def _pitch_labels_for_pitcher(pid: int, start: str, end: str, stand: Optional[str]=None) -> List[str]:
    try:
        df = fetch_pitcher_statcast(pid, start_date=start, end_date=end)
        if df is None or df.empty or "pitch_type" not in df.columns: return []
        if stand in ('R','L'): df = df[df['stand']==stand]
        order = df["pitch_type"].dropna().astype(str).value_counts().index.tolist()
        return [PITCH_NAME_MAP.get(k, k) for k in order]
    except Exception:
        return []

def _opponent_active_pitchers(opponent_id: int) -> List[Dict]:
    ro = statsapi.get("team_roster", {"teamId": opponent_id, "rosterType": "active"})
    out = []
    for r in ro.get("roster", []):
        if str((r.get("position") or {}).get("type","")).lower()=="pitcher":
            out.append({"id": int(r["person"]["id"]), "name": r["person"]["fullName"]})
    return out

def _fetch_pitcher_stats(pitcher_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch pitcher stats: age, W-L, IP, ERA, WHIP, Ks using pybaseball for the date range."""
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
        from pybaseball import pitching_stats_range
        import pandas as pd
        
        # Fetch pitching stats for the date range
        stats_df = pitching_stats_range(start_date, end_date)
        
        if stats_df.empty:
            return stats
        
        # Find the pitcher by name (case-insensitive, handle variations)
        pitcher_name_lower = pitcher_name.lower().strip()
        
        if 'Name' in stats_df.columns:
            # Try exact match first
            player_row = stats_df[stats_df['Name'].str.lower().str.strip() == pitcher_name_lower]
            
            # If no exact match, try last name match
            if player_row.empty:
                name_parts = pitcher_name_lower.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    player_row = stats_df[stats_df['Name'].str.lower().str.contains(last_name, na=False)]
            
            if not player_row.empty:
                row = player_row.iloc[0]
                
                # Get stats from the row
                if 'Age' in row and pd.notna(row['Age']):
                    stats['age'] = int(row['Age'])
                
                if 'W' in row and pd.notna(row['W']):
                    stats['wins'] = int(row['W'])
                
                if 'L' in row and pd.notna(row['L']):
                    stats['losses'] = int(row['L'])
                
                if 'IP' in row and pd.notna(row['IP']):
                    stats['ip'] = float(row['IP'])
                
                if 'ERA' in row and pd.notna(row['ERA']):
                    stats['era'] = float(row['ERA'])
                
                if 'WHIP' in row and pd.notna(row['WHIP']):
                    stats['whip'] = float(row['WHIP'])
                
                if 'SO' in row and pd.notna(row['SO']):
                    stats['strikeouts'] = int(row['SO'])
    except Exception:
        pass
    
    return stats

def _build_all_for_pitcher(p: Dict, season_start: str, game_date: str, want_stand: Optional[str], logo_path: Optional[str] = None) -> Dict:
    pid, pname = int(p["id"]), p["name"]
    try: hm = build_heatmaps_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures", stand=want_stand)
    except Exception: hm = None
    try: tables = build_pitch_tables_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures")
    except Exception: tables = None
    try: move = build_movement_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures", include_density=False, normalize_by_throws=False)
    except Exception: move = None
    try: mix = build_pitch_mix_by_count_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures", logo_path=logo_path, stand=want_stand)
    except Exception: mix = None
    labels = _pitch_labels_for_pitcher(pid, season_start, game_date, stand=want_stand)
    # Fetch pitcher headshot
    headshot_url = None
    try:
        headshot_url = _fetch_player_headshot(pname)
    except Exception:
        pass
    # Fetch pitcher stats using pybaseball
    pitcher_stats = {}
    try:
        pitcher_stats = _fetch_pitcher_stats(pname, season_start, game_date)
    except Exception:
        pass
    
    # Fetch pitcher handedness - try statcast first, then MLB API as fallback
    if "handedness" not in pitcher_stats:
        try:
            df = fetch_pitcher_statcast(pid, start_date=season_start, end_date=game_date)
            if df is not None and not df.empty and "p_throws" in df.columns:
                throws_values = df["p_throws"].dropna()
                if not throws_values.empty:
                    throws_mode = throws_values.mode()
                    if not throws_mode.empty:
                        throws = str(throws_mode.iloc[0]).upper()
                        pitcher_stats["handedness"] = "LHP" if throws == "L" else "RHP"
        except Exception:
            pass
    
    # Fallback: get handedness from MLB API if statcast didn't work
    if "handedness" not in pitcher_stats:
        try:
            person_data = statsapi.get("people", {"personIds": pid})
            if person_data and "people" in person_data and len(person_data["people"]) > 0:
                pitch_hand = person_data["people"][0].get("pitchHand", {})
                if pitch_hand:
                    hand_code = pitch_hand.get("code", "").upper()
                    if hand_code == "L":
                        pitcher_stats["handedness"] = "LHP"
                    elif hand_code == "R":
                        pitcher_stats["handedness"] = "RHP"
                    # Also try description field as fallback
                    elif not pitcher_stats.get("handedness"):
                        hand_desc = pitch_hand.get("description", "").upper()
                        if "LEFT" in hand_desc:
                            pitcher_stats["handedness"] = "LHP"
                        elif "RIGHT" in hand_desc:
                            pitcher_stats["handedness"] = "RHP"
        except Exception:
            pass
    
    return {
        "name": pname,
        "heat": {"name": pname, "path": hm, "labels": labels},
        "tables": {"name": pname, "RHH": (tables.get("RHH") if isinstance(tables, dict) else None), "LHH": (tables.get("LHH") if isinstance(tables, dict) else None)},
        "move": {"name": pname, "path": move},
        "mix":  {"name": pname, "path": mix},
        "headshot": headshot_url,
        "stats": pitcher_stats,
    }

def _collect_pitcher_assets_for_staff(opponent_id: int, season_start: str, game_date: str, want_stand: Optional[str], workers: int) -> Dict:
    assets = { "oppo_heatmaps": [], "oppo_pitch_tables": [], "oppo_pitch_movement": [], "oppo_pitch_mix_by_count": [], "oppo_pitcher_headshots": [] }
    staff = _opponent_active_pitchers(opponent_id)
    if not staff: return assets
    
    # Fetch opponent logo for charts (convert URI to file path if needed)
    opponent_logo_path = None
    try:
        logo_uri = _fetch_team_logo(team_id=opponent_id)
        if logo_uri:
            # Convert file:// URI to local path for matplotlib
            if logo_uri.startswith('file://'):
                from urllib.parse import urlparse
                opponent_logo_path = urlparse(logo_uri).path
            else:
                opponent_logo_path = logo_uri
    except Exception:
        pass
    
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_build_all_for_pitcher, p, season_start, game_date, want_stand, opponent_logo_path) for p in staff]
        for f in as_completed(futs):
            try:
                res = f.result(); pname = res["name"]
                
                # Require the full visualization set so we don't render blank panels
                has_heatmap = res["heat"].get("path") is not None and res["heat"].get("path") != ""
                has_table = ((res["tables"].get("RHH") is not None and res["tables"].get("RHH") != "") or 
                            (res["tables"].get("LHH") is not None and res["tables"].get("LHH") != ""))
                has_movement = res["move"].get("path") is not None and res["move"].get("path") != ""
                has_mix = res["mix"].get("path") is not None and res["mix"].get("path") != ""
                
                # Only include pitcher if every visualization is available
                if has_heatmap and has_table and has_movement and has_mix:
                    # Only add entries with valid paths (not None or empty)
                    assets["oppo_heatmaps"].append({ "name": pname, **res["heat"] })
                    assets["oppo_pitch_tables"].append({ "name": pname, **res["tables"] })
                    assets["oppo_pitch_movement"].append({ "name": pname, **res["move"] })
                    assets["oppo_pitch_mix_by_count"].append({ "name": pname, **res["mix"] })
                    # Always add headshot and stats when pitcher is included
                    assets["oppo_pitcher_headshots"].append({ 
                        "name": pname, 
                        "headshot_url": res.get("headshot"),
                        "stats": res.get("stats", {})
                    })
            except Exception:
                continue
    for key in assets:
        for it in assets[key]:
            for sub in ("path","RHH","LHH"):
                if it.get(sub): it[sub] = _to_uri(it[sub])
    return assets

def hitter_stand_from_savant(name: str, start: str, end: str) -> str:
    try:
        bid = lookup_batter_id(name)
        df = fetch_batter_statcast(bid, start, end)
        if df is not None and not df.empty and "stand" in df.columns:
            m = df["stand"].dropna().astype(str).mode()
            if not m.empty: return m.iloc[0]
    except Exception:
        pass
    return "R"

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
            if use_next_series:
                # Find the first game of the series with this opponent
                return matching_games[0]
            else:
                return matching_games[0]
        else:
            # No upcoming game found, create a synthetic opponent info
            # We'll use today's date and the opponent info
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

def build_context(team_abbr: str, hitter_name: str, season_start: str, use_next_series: bool=False, opponent_team: Optional[str]=None, token: Optional[str]=None, redeem_url: Optional[str]=None, pdf_date: Optional[str]=None, workers: int=4) -> Dict:
    opp = _get_opponent_info(team_abbr, opponent_team, use_next_series)
    game_date, opponent_name, opponent_id = opp["game_date"], opp["opponent_name"], opp["opponent_id"]
    tok = token or _fetch_token() or ""
    red = _normalize_redeem_url(redeem_url or REDEEM_URL_DEFAULT)
    
    # Fetch team logos and player headshot (non-blocking, graceful failure)
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
        player_headshot_url = _fetch_player_headshot(hitter_name)
    except Exception:
        pass
    
    ctx: Dict = {
        "player": hitter_name,
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
        "figures": {},
    }
    from build_hitter_checkin import build_hitter_checkin
    hitter_stand = hitter_stand_from_savant(hitter_name, season_start, game_date)
    want_side = 'RHH' if hitter_stand == 'R' else 'LHH'
    want_stand_char = 'R' if hitter_stand == 'R' else 'L'
    hc = build_hitter_checkin(hitter_name, season_start, game_date, want_side)
    ctx["figures"]["hitter_checkin"] = hc
    staff_assets = _collect_pitcher_assets_for_staff(opponent_id, season_start, game_date, want_stand=want_stand_char, workers=workers)
    ctx["figures"].update(staff_assets)
    ctx["opponent_info"] = opp
    return ctx

def _safe_filename(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "", s).strip()

def _resolve_team_for_hitter(hitter_name: str) -> Optional[str]:
    try:
        candidates = statsapi.lookup_player(hitter_name) or []
        hit = next((m for m in candidates if str(m.get("fullName","")).lower()==hitter_name.lower()), candidates[0] if candidates else None)
        if not hit: return None
        pid = int(hit["id"])
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
    ap = argparse.ArgumentParser(description="Generate automated hitter scouting PDF.")
    ap.add_argument("--team", required=True)
    ap.add_argument("--hitter", required=True)
    ap.add_argument("--season_start", default="2025-03-20")
    ap.add_argument("--opponent", default=None, help="Opponent team abbreviation (optional, auto-detects if not provided)")
    ap.add_argument("--token", default=None)
    ap.add_argument("--redeem_url", default=None)
    ap.add_argument("--out", default="build/pdf")
    ap.add_argument("--template", default="templates/hitter_report.html")
    ap.add_argument("--pdf_name", default=None)
    ap.add_argument("--pdf_date", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-next-series", action="store_true")
    args = ap.parse_args()

    team_abbr = (args.team or "").upper()
    if team_abbr in ("", "AUTO"):
        auto_team = _resolve_team_for_hitter(args.hitter)
        team_abbr = auto_team or "NYM"

    opponent_team = args.opponent.upper() if args.opponent else None

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ctx = build_context(team_abbr=team_abbr, hitter_name=args.hitter, season_start=args.season_start, use_next_series=args.use_next_series, opponent_team=opponent_team, token=args.token, redeem_url=args.redeem_url, pdf_date=args.pdf_date, workers=max(1, args.workers))

        opp_for_name = ctx.get("opponent") or ctx.get("opponent_info", {}).get("opponent_name") or "Unknown"
        pdf_name = args.pdf_name or f"{args.hitter} vs {opp_for_name}.pdf"
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
