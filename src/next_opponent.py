# src/next_opponent.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import statsapi  # pip install MLB-StatsAPI

# ---------- Team index helpers ----------

def _build_team_index() -> Dict[str, int]:
    """
    Build a case-insensitive index mapping common keys (fileCode, abbreviation, names) -> teamId.
    """
    idx: Dict[str, int] = {}
    teams = statsapi.get('teams', {'sportId': 1})['teams']
    for t in teams:
        tid = t['id']
        keys = set()
        for k in (
            t.get('fileCode'),
            t.get('abbreviation'),
            t.get('teamName'),
            t.get('name'),
            t.get('clubName'),
            t.get('shortName'),
        ):
            if k:
                keys.add(k.upper())
        # also support city + team (e.g., "NEW YORK METS")
        city = t.get('venue', {}).get('city')
        if city and t.get('teamName'):
            keys.add(f"{city} {t['teamName']}".upper())
        for k in keys:
            idx[k] = tid
    return idx

_TEAM_INDEX = _build_team_index()

def _resolve_team_id(team_key: str) -> int:
    k = team_key.strip().upper()
    if k not in _TEAM_INDEX:
        raise ValueError(f"Unrecognized team key: {team_key!r}")
    return _TEAM_INDEX[k]

# ---------- Core logic ----------

def _probables_from_game(game: Dict[str, Any], team_id: int) -> List[Dict[str, Any]]:
    """
    Return a list of probable pitchers for the OPPONENT of `team_id` in this game.
    List length is usually 0 or 1 (MLB sometimes lists both if TBA changes).
    """
    home_id, away_id = game['home_id'], game['away_id']
    prob: List[Dict[str, Any]] = []

    # schedule dict keys commonly present in statsapi.schedule()
    # 'home_probable_pitcher', 'home_probable_pitcher_id', 'away_probable_pitcher', 'away_probable_pitcher_id'
    if team_id == home_id:
        opp_name = game.get('away_probable_pitcher')
        opp_id   = game.get('away_probable_pitcher_id')
    else:
        opp_name = game.get('home_probable_pitcher')
        opp_id   = game.get('home_probable_pitcher_id')

    if opp_name and opp_id:
        prob.append({"id": int(opp_id), "name": str(opp_name)})

    return prob

def next_games(team_key: str, days_ahead: int = 7, include_started: bool = False) -> List[Dict[str, Any]]:
    """
    Find all upcoming games for a team in [today, today+days_ahead], earliest first.

    Returns a list of dicts:
      {
        "game_date": "YYYY-MM-DD",
        "game_datetime": "YYYY-MM-DDTHH:MM:SSZ",   # UTC ISO 8601 if provided
        "game_pk": <int>,
        "home_id": <int>, "home_name": <str>,
        "away_id": <int>, "away_name": <str>,
        "opponent_id": <int>, "opponent_name": <str>,
        "is_home": <bool>,
        "venue": <str>,
        "series_description": <str|None>,
        "status": <str>,                           # Scheduled, Pre-Game, In Progress, Final, etc.
        "probable_pitchers": [{"id": <int>, "name": <str>}, ...]  # opponent probables (0–1 typical)
      }
    """
    team_id = _resolve_team_id(team_key)
    tz = timezone.utc
    today = datetime.now(tz).date()

    end_date = today + timedelta(days=days_ahead)
    try:
        schedule = statsapi.schedule(
            start_date=today.isoformat(),
            end_date=end_date.isoformat(),
            team=team_id
        )
    except Exception:
        schedule = []

    results: List[Dict[str, Any]] = []

    for g in schedule:
        status = g.get('status', '')
        if not include_started and status in ('Final', 'Game Over'):
            continue  # skip completed

        game_date = g.get('game_date') or g.get('gameDate')
        if not game_date:
            continue

        home_id, away_id = g['home_id'], g['away_id']
        is_home = home_id == team_id
        opponent_id = away_id if is_home else home_id
        opponent_name = g['away_name'] if is_home else g['home_name']

        game_datetime = None
        if g.get('game_datetime'):
            try:
                dt = datetime.fromisoformat(g['game_datetime'].replace('Z', '+00:00')).astimezone(timezone.utc)
                game_datetime = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            except Exception:
                game_datetime = None

        results.append({
            "game_date": game_date,
            "game_datetime": game_datetime,
            "game_pk": g.get("game_id") or g.get("game_pk"),
            "home_id": home_id,
            "home_name": g.get('home_name'),
            "away_id": away_id,
            "away_name": g.get('away_name'),
            "opponent_id": opponent_id,
            "opponent_name": opponent_name,
            "is_home": is_home,
            "venue": g.get('venue_name'),
            "series_description": g.get('series_description'),
            "status": status,
            "probable_pitchers": _probables_from_game(g, team_id),
        })

    # Sort by datetime (or date as fallback), then by game number (if present)
    def _sort_key(item: Dict[str, Any]):
        dt = item.get("game_datetime")
        if dt:
            try:
                return (datetime.fromisoformat(dt.replace('Z', '+00:00')), 0)
            except Exception:
                pass
        # fallback to game_date only
        return (datetime.fromisoformat(item["game_date"]), 0)

    results.sort(key=_sort_key)
    return results

def next_game_info(team_key: str, days_ahead: int = 7) -> Dict[str, Any]:
    """
    Convenience wrapper: return the earliest upcoming game dict within the window.
    Raises if none found.
    """
    games = next_games(team_key, days_ahead=days_ahead, include_started=False)
    if not games:
        raise RuntimeError(f"No upcoming games found within {days_ahead} days for {team_key}.")
    return games[0]

def next_series_game_info(team_key: str, days_ahead: int = 14) -> dict:
    games = next_games(team_key, days_ahead=days_ahead, include_started=False)
    if not games:
        raise RuntimeError(f"No upcoming games found within {days_ahead} days for {team_key}.")
    def series_chunks(gs):
        chunk = []
        last_opp = None
        for g in gs:
            opp = g["opponent_id"]
            if last_opp is None or opp == last_opp:
                chunk.append(g)
            else:
                yield chunk
                chunk = [g]
            last_opp = opp
        if chunk:
            yield chunk
    chunks = list(series_chunks(games))
    if not chunks:
        raise RuntimeError("No series chunks computed.")
    if len(chunks) == 1:
        raise RuntimeError("Only one opponent in window. Increase days_ahead.")
    return chunks[1][0]

def next_series_game_info(team_key: str, days_ahead: int = 14) -> dict:
    games = next_games(team_key, days_ahead=days_ahead, include_started=False)
    if not games:
        raise RuntimeError(f"No upcoming games found within {days_ahead} days for {team_key}.")
    def series_chunks(gs):
        chunk = []
        last_opp = None
        for g in gs:
            opp = g["opponent_id"]
            if last_opp is None or opp == last_opp:
                chunk.append(g)
            else:
                yield chunk
                chunk = [g]
            last_opp = opp
        if chunk:
            yield chunk
    chunks = list(series_chunks(games))
    if not chunks:
        raise RuntimeError("No series chunks computed.")
    if len(chunks) == 1:
        raise RuntimeError("Only one opponent in window. Increase days_ahead.")
    return chunks[1][0]
