from __future__ import annotations

import argparse, json, re, urllib.request
import uuid, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import statsapi
from render import render_html_to_pdf
from next_opponent import next_game_info, next_series_game_info
from optimize_assets import optimize_dir
from scrape_savant import (
    lookup_batter_id, fetch_batter_statcast, fetch_pitcher_statcast
)
from build_movement import build_movement_for_pitcher
from build_bar_by_count import build_pitch_mix_by_count_for_pitcher
from build_pitch_tables import build_pitch_tables_for_pitcher

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
PITCH_NAME_MAP = {
    "FF":"Four-Seam","FA":"Four-Seam","FT":"Sinker","SI":"Sinker","ST":"Sweeper",
    "SL":"Slider","FC":"Cutter","CT":"Cutter","CH":"Changeup","CU":"Curveball",
    "KC":"Knuckle Curve","CS":"Slurve","FS":"Splitter","FO":"Forkball",
    "SF":"Split-Finger","KN":"Knuckleball","EP":"Eephus",
}

def _to_uri(p: Optional[str]) -> Optional[str]:
    return None if not p else Path(p).resolve().as_uri()

def _normalize_redeem_url(u: Optional[str]) -> str:
    if not u: return REDEEM_URL_DEFAULT
    s = str(u).strip().replace("\u2014","-").replace("\u2013","-").replace("\u2212","-").replace("—","-").replace("–","-")
    if not s.startswith(("http://","https://")): s = "https://" + s.lstrip("/")
    if not s.endswith("/"): s += "/"
    return s

def _fetch_token() -> Optional[str]:
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
    return None

def _team_abbr_from_id(team_id: int) -> Optional[str]:
    try:
        info = statsapi.get("teams", {"teamId": team_id})
        t = (info.get("teams") or [])[0]
        return (t.get("fileCode") or "").upper() or None
    except Exception:
        return None

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
                abbr = _team_abbr_from_id(int(ct["id"]))
                if abbr: return abbr
        except Exception:
            pass
        all_teams = statsapi.get("teams", {"sportId": 1}).get("teams", [])
        for rosterType in ("active", "40Man"):
            for t in all_teams:
                tid = int(t["id"])
                roster = statsapi.get("team_roster", {"teamId": tid, "rosterType": rosterType}).get("roster", [])
                if any(int(r["person"]["id"])==pid for r in roster):
                    return (t.get("fileCode") or "").upper() or None
    except Exception:
        pass
    return None

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

# -------- Parallel worker (runs in a separate process) --------
def _build_all_for_pitcher(p: Dict, season_start: str, game_date: str, want_stand: Optional[str]) -> Dict:
    pid, pname = int(p["id"]), p["name"]
    hm = None
    try:
        hm = build_heatmaps_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures", stand=want_stand)
    except Exception:
        hm = None
    try:
        tables = build_pitch_tables_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures")
    except Exception:
        tables = None
    try:
        move = build_movement_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures",
                                          include_density=True, normalize_by_throws=False)
    except Exception:
        move = None
    try:
        mix = build_pitch_mix_by_count_for_pitcher(pid, pname, season_start, game_date, out_dir="build/figures",
                                                   logo_path=None, stand=want_stand)
    except Exception:
        mix = None
    labels = _pitch_labels_for_pitcher(pid, season_start, game_date, stand=want_stand)
    return {
        "name": pname,
        "heat": {"name": pname, "path": hm, "labels": labels},
        "tables": {"name": pname,
                   "RHH": (tables.get("RHH") if isinstance(tables, dict) else None),
                   "LHH": (tables.get("LHH") if isinstance(tables, dict) else None)},
        "move": {"name": pname, "path": move},
        "mix":  {"name": pname, "path": mix},
    }

def _collect_pitcher_assets_for_staff(opponent_id: int, season_start: str, game_date: str, want_stand: Optional[str], workers: int) -> Dict:
    assets = { "oppo_heatmaps": [], "oppo_pitch_tables": [], "oppo_pitch_movement": [], "oppo_pitch_mix_by_count": [] }
    staff = _opponent_active_pitchers(opponent_id)
    if not staff:
        return assets

    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_build_all_for_pitcher, p, season_start, game_date, want_stand) for p in staff]
        for f in as_completed(futs):
            try:
                res = f.result()
                pname = res["name"]
                assets["oppo_heatmaps"].append({ "name": pname, **res["heat"] })
                assets["oppo_pitch_tables"].append({ "name": pname, **res["tables"] })
                assets["oppo_pitch_movement"].append({ "name": pname, **res["move"] })
                assets["oppo_pitch_mix_by_count"].append({ "name": pname, **res["mix"] })
            except Exception:
                continue

    # convert paths to file:// URIs
    for key in assets:
        for it in assets[key]:
            for sub in ("path","RHH","LHH"):
                if it.get(sub):
                    it[sub] = _to_uri(it[sub])
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
def build_context(team_abbr, hitter_name, season_start, use_next_series=False, token=None, redeem_url=None, pdf_date=None, workers=4) -> Dict:
    opp = next_series_game_info(team_abbr) if use_next_series else next_game_info(team_abbr)
    game_date, opponent_name, opponent_id = opp["game_date"], opp["opponent_name"], opp["opponent_id"]
    tok = token or _fetch_token() or ""
    red = _normalize_redeem_url(redeem_url or REDEEM_URL_DEFAULT)
    ctx: Dict = {
        "player": hitter_name,
        "opponent": opponent_name,
        "date": pdf_date or game_date,
        "token": tok,
        "redeem_url": red,
        "game_date": game_date,
        "season_start": season_start,
        "figures": {},
    }
    from build_hitter_checkin import build_hitter_checkin

def _fetch_token():
    try:
        v=os.environ.get("SEQUENCE_TOKEN")
        if v: return v.strip()
    except Exception:
        pass
    token_dir=Path("build")
    token_dir.mkdir(parents=True, exist_ok=True)
    token_file=token_dir/"token.txt"
    try:
        v=token_file.read_text().strip()
        if v: return v
    except Exception:
        pass
    v=str(uuid.uuid4())
    token_file.write_text(v)
    return v

    hitter_stand = hitter_stand_from_savant(hitter_name, season_start, game_date)
    want_side = 'RHH' if hitter_stand == 'R' else 'LHH'
    want_stand_char = 'R' if hitter_stand == 'R' else 'L'
    hc = build_hitter_checkin(hitter_name, season_start, game_date, want_side)
    ctx["figures"]["hitter_checkin"] = hc
    staff_assets = _collect_pitcher_assets_for_staff(opponent_id, season_start, game_date,
                                                     want_stand=want_stand_char, workers=workers)
    ctx["figures"].update(staff_assets)
    ctx["opponent_info"] = opp
    return ctx

def _safe_filename(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "", s).strip()

def main():
    ap = argparse.ArgumentParser(description="Generate automated hitter scouting PDF.")
    ap.add_argument("--team", required=True, help="Home team abbr (e.g., NYM) or 'AUTO'")
    ap.add_argument("--hitter", required=True)
    ap.add_argument("--season_start", default="2025-03-20")
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

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_context(team_abbr=team_abbr, hitter_name=args.hitter, season_start=args.season_start, use_next_series=args.use_next_series,
                        token=args.token, redeem_url=args.redeem_url, pdf_date=args.pdf_date,
                        workers=max(1, args.workers))

    opp_for_name = (ctx or {}).get('opponent') or (ctx or {}).get('opponent_name') or ((ctx or {}).get('opponent_info') or {}).get('opponent_name') or 'Unknown'
    opp_info = next_series_game_info(team_abbr) if args.use_next_series else next_game_info(team_abbr)
opp_for_name = opp_info.get("opponent_name") or opp_info.get("opponent")
    opp_info = next_series_game_info(team_abbr) if args.use_next_series else next_game_info(team_abbr)
    opp_for_name = opp_info.get("opponent_name") or opp_info.get("opponent")
    pdf_name = args.pdf_name or f"{args.hitter} vs {opp_for_name}.pdf"
    html_name = args.pdf_name or f"{args.hitter} vs {opp_for_name}.html"
    out_pdf = out_dir / _safe_filename(pdf_name)
    optimize_dir("build/figures")
    render_html_to_pdf(args.template, ctx, out_pdf)
    print(f"Saved report: {out_pdf}")

if __name__ == "__main__":
    main()