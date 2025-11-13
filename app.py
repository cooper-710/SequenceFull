#!/usr/bin/env python3
"""
Web UI for Scouting Report Generator
"""

import os
import sys
import subprocess
import json
import secrets
import re
import requests
import mimetypes
import textwrap
from functools import lru_cache, wraps
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session, g, redirect, url_for, flash, abort
from datetime import datetime, timedelta, timezone, time, date
import threading
import uuid
from collections import defaultdict
from urllib.parse import quote_plus
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import statsapi
from typing import Optional, List, Dict, Any, Set, Tuple

import settings_manager

# --- Render Free Tier fallback (safe to keep forever) ---
# If these env vars aren't set by the host, default to /tmp so free-tier works.
if not os.environ.get("APP_DATA_DIR"):
    os.environ["APP_DATA_DIR"] = "/tmp"

if not os.environ.get("APP_SQLITE_PATH"):
    from pathlib import Path as _P
    _base = _P(os.environ["APP_DATA_DIR"])
    _base.mkdir(parents=True, exist_ok=True)
    os.environ["APP_SQLITE_PATH"] = str(_base / "database" / "players.db")
# --- end fallback ---


# Import database and client modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from database import PlayerDB
except ImportError as e:
    print(f"Warning: Could not import PlayerDB: {e}")
    PlayerDB = None

try:
    from next_opponent import next_games
except ImportError as e:
    print(f"Warning: Could not import next_games: {e}")
    next_games = None

app = Flask(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).parent.resolve()

# Import CSV data loader
try:
    from csv_data_loader import CSVDataLoader
    csv_loader = CSVDataLoader(str(ROOT_DIR))
    print(f"CSV data loader initialized successfully")
except ImportError as e:
    print(f"Warning: Could not import CSVDataLoader: {e}")
    csv_loader = None
except Exception as e:
    print(f"Warning: Error initializing CSVDataLoader: {e}")
    import traceback
    traceback.print_exc()
    csv_loader = None
OUT_DIR = ROOT_DIR / "build" / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_UPLOAD_DIR = ROOT_DIR / "static" / "uploads" / "profile_photos"
PROFILE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_PROFILE_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_PROFILE_IMAGE_TYPES = {"png", "jpeg", "gif", "webp"}
MAX_PROFILE_IMAGE_BYTES = 5 * 1024 * 1024
def detect_image_type(data: bytes) -> Optional[str]:
    """Detect image type for a small subset of formats using magic headers."""
    if not data or len(data) < 4:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None

# Store job statuses (in production, use Redis or a database)
job_status = {}

AUTH_EXEMPT_ENDPOINTS = {
    "login",
    "register",
    "static"
}

TEAM_ABBR_TO_ID = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145, "CIN": 113,
    "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119,
    "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133, "PHI": 143,
    "PIT": 134, "SD": 135, "SF": 137, "SEA": 136, "STL": 138, "TB": 139, "TEX": 140,
    "TOR": 141, "WSH": 120,
    "ANA": 108, "CHW": 145, "KCR": 118, "SDP": 135, "SFG": 137, "TBR": 139,
    "WSN": 120, "WAS": 120
}

DIVISION_OPTIONS = [
    {"id": 201, "name": "American League East", "league_id": 103},
    {"id": 202, "name": "American League Central", "league_id": 103},
    {"id": 200, "name": "American League West", "league_id": 103},
    {"id": 204, "name": "National League East", "league_id": 104},
    {"id": 205, "name": "National League Central", "league_id": 104},
    {"id": 203, "name": "National League West", "league_id": 104},
]

LEAGUE_OPTIONS = [
    {"id": 103, "name": "American League"},
    {"id": 104, "name": "National League"},
]

LEADER_CATEGORY_ABBR = {
    "homeRuns": "HR",
    "runsBattedIn": "RBI",
    "battingAverage": "AVG",
    "era": "ERA",
    "strikeouts": "K",
    "whip": "WHIP",
}


_REPORT_OPP_PATTERN = re.compile(r"_vs_([A-Za-z0-9]+)", re.IGNORECASE)

REPORT_LEAD_DAYS = 3
_PENDING_REPORT_KEYS: Set[str] = set()
_PENDING_REPORT_LOCK = threading.Lock()


def _purge_concluded_series_documents(reference_ts: Optional[float] = None) -> None:
    """Remove player documents tied to series that have already finished."""
    if not PlayerDB:
        return
    try:
        db = PlayerDB()
        if reference_ts is None:
            reference_ts = datetime.now().timestamp() - SERIES_AUTO_DELETE_GRACE_SECONDS
        expired_docs = db.list_expired_player_documents(reference_ts)
        for doc in expired_docs:
            deleted = db.delete_player_document(doc["id"])
            if not deleted:
                continue
            db.record_player_document_event(
                player_id=deleted["player_id"],
                filename=deleted["filename"],
                action="auto_delete_series",
                performed_by=None,
            )
            file_path = Path(deleted.get("path") or "")
            if file_path.exists() and file_path.is_file():
                try:
                    file_path.unlink()
                except OSError as exc:
                    print(f"Warning removing expired document file {file_path}: {exc}")
        db.close()
    except Exception as exc:
        print(f"Warning purging expired player documents: {exc}")


def _cache_get(cache: Dict[Any, Any], key: Any):
    entry = cache.get(key)
    if not entry:
        return None
    value, expires_at = entry
    if expires_at and expires_at > datetime.utcnow():
        return value
    cache.pop(key, None)
    return None


def _cache_set(cache: Dict[Any, Any], key: Any, value: Any, ttl_seconds: int):
    cache[key] = (value, datetime.utcnow() + timedelta(seconds=ttl_seconds))


_UPCOMING_GAMES_CACHE: Dict[Any, Any] = {}
_LEAGUE_LEADERS_CACHE: Dict[Any, Any] = {}
_STANDINGS_CACHE: Dict[Any, Any] = {}
_TEAM_METADATA_CACHE: Dict[Any, Any] = {}


JOURNAL_VISIBILITY_OPTIONS = ("private", "public")
MAX_JOURNAL_TIMELINE_ENTRIES = 365


def _normalize_journal_visibility(value: Optional[str], default: str = "private") -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in JOURNAL_VISIBILITY_OPTIONS:
        return default
    return normalized


def _prepare_journal_timeline(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group journal entries by date and prepare display metadata."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        entry_date = (entry.get("entry_date") or "").strip()
        if not entry_date:
            continue
        normalized_visibility = _normalize_journal_visibility(entry.get("visibility"), "private")
        body_text = entry.get("body") or ""
        preview = body_text.strip()
        max_preview = 160
        if len(preview) > max_preview:
            preview = preview[:max_preview].rstrip() + "…"
        try:
            display_date = datetime.strptime(entry_date, "%Y-%m-%d").strftime("%b %d, %Y")
        except ValueError:
            display_date = entry_date
        updated_at_ts = entry.get("updated_at")
        updated_at_human = None
        if updated_at_ts:
            try:
                updated_at_human = datetime.fromtimestamp(updated_at_ts).strftime("%b %d, %Y %I:%M %p")
            except (ValueError, OSError):
                updated_at_human = None

        grouped[entry_date].append({
            **entry,
            "visibility": normalized_visibility,
            "display_date": display_date,
            "preview": preview,
            "updated_at_human": updated_at_human,
        })

    timeline: List[Dict[str, Any]] = []
    for date_key in sorted(grouped.keys(), reverse=True):
        timeline.append({
            "date": date_key,
            "display_date": grouped[date_key][0].get("display_date"),
            "entries": sorted(grouped[date_key], key=lambda item: item["visibility"]),
        })
    return timeline


def _augment_journal_entry(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Add display metadata to a single journal entry."""
    if not entry:
        return None
    enriched = dict(entry)
    entry_date = (enriched.get("entry_date") or "").strip()
    try:
        enriched["display_date"] = datetime.strptime(entry_date, "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        enriched["display_date"] = entry_date
    updated_at_ts = enriched.get("updated_at")
    if updated_at_ts:
        try:
            enriched["updated_at_human"] = datetime.fromtimestamp(updated_at_ts).strftime("%b %d, %Y %I:%M %p")
        except (ValueError, OSError):
            enriched["updated_at_human"] = None
    else:
        enriched["updated_at_human"] = None
    enriched["visibility"] = _normalize_journal_visibility(enriched.get("visibility"), default="private")
    return enriched


def _format_journal_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        return date_str


def _mock_schedule_enabled_default() -> bool:
    env_value = os.environ.get("USE_MOCK_SCHEDULE")
    if env_value is None:
        return True
    return env_value.strip().lower() in {"1", "true", "yes", "on"}


app.config["USE_MOCK_SCHEDULE"] = _mock_schedule_enabled_default()

if app.config["USE_MOCK_SCHEDULE"]:
    print("Mock schedule enabled; set USE_MOCK_SCHEDULE=0 to restore live data.")


@lru_cache(maxsize=1)
def _team_directory() -> Dict[int, Dict[str, str]]:
    """Cache team metadata keyed by team id for quick lookups."""
    try:
        teams = statsapi.get("teams", {"sportId": 1}).get("teams", [])
    except Exception:
        return {}
    directory: Dict[int, Dict[str, str]] = {}
    for entry in teams:
        team_id = entry.get("id")
        if not team_id:
            continue
        abbr = entry.get("abbreviation") or entry.get("fileCode") or entry.get("teamCode")
        directory[int(team_id)] = {
            "abbr": (abbr or "").upper(),
            "name": entry.get("teamName"),
        }
    return directory


def _build_mock_upcoming_games(team_abbr: Optional[str], limit: int = 5) -> List[Dict[str, Any]]:
    """Generate a deterministic mock schedule for local testing."""
    now = datetime.now().astimezone()
    base_first_pitch = now.replace(hour=19, minute=10, second=0, microsecond=0)

    blueprint = [
        {
            "days_offset": -6,
            "status": "Final",
            "opponent": "Miami Marlins",
            "opponent_abbr": "MIA",
            "opponent_id": 146,
            "home": False,
            "venue": "loanDepot park",
            "series": "3-game series",
            "game_pk": 499900,
            "probable_pitchers": ["Jesús Luzardo"],
        },
        {
            "days_offset": -5,
            "status": "Final",
            "opponent": "Miami Marlins",
            "opponent_abbr": "MIA",
            "opponent_id": 146,
            "home": False,
            "venue": "loanDepot park",
            "series": "3-game series",
            "game_pk": 499901,
            "probable_pitchers": ["Sandy Alcantara"],
        },
        {
            "days_offset": 0,
            "status": "In Progress",
            "opponent": "Washington Nationals",
            "opponent_abbr": "WSH",
            "opponent_id": 120,
            "home": True,
            "venue": "Citi Field",
            "series": "Division matchup",
            "game_pk": 500000,
            "probable_pitchers": ["Josiah Gray"],
        },
        {
            "days_offset": 1,
            "status": "Pre-Game",
            "opponent": "Washington Nationals",
            "opponent_abbr": "WSH",
            "opponent_id": 120,
            "home": True,
            "venue": "Citi Field",
            "series": "Division matchup",
            "game_pk": 500001,
            "probable_pitchers": ["MacKenzie Gore"],
        },
        {
            "days_offset": 2,
            "status": "Scheduled",
            "opponent": "Philadelphia Phillies",
            "opponent_abbr": "PHI",
            "opponent_id": 143,
            "home": True,
            "venue": "Citi Field",
            "series": "3-game series",
            "game_pk": 500100,
            "probable_pitchers": ["Zack Wheeler"],
        },
        {
            "days_offset": 3,
            "status": "Scheduled",
            "opponent": "Philadelphia Phillies",
            "opponent_abbr": "PHI",
            "opponent_id": 143,
            "home": True,
            "venue": "Citi Field",
            "series": "3-game series",
            "game_pk": 500101,
            "probable_pitchers": ["Aaron Nola"],
        },
        {
            "days_offset": 5,
            "status": "Scheduled",
            "opponent": "Atlanta Braves",
            "opponent_abbr": "ATL",
            "opponent_id": 144,
            "home": False,
            "venue": "Truist Park",
            "series": "Division matchup",
            "game_pk": 500200,
            "probable_pitchers": ["Max Fried"],
        },
        {
            "days_offset": 6,
            "status": "Scheduled",
            "opponent": "Atlanta Braves",
            "opponent_abbr": "ATL",
            "opponent_id": 144,
            "home": False,
            "venue": "Truist Park",
            "series": "Division matchup",
            "status": "Scheduled",
            "game_pk": 500201,
            "probable_pitchers": ["Chris Sale"],
        },
    ]

    formatted: List[Dict[str, Any]] = []
    for entry in blueprint[:max(limit, len(blueprint))]:
        game_dt = base_first_pitch + timedelta(days=entry["days_offset"])
        formatted_time = game_dt.strftime("%I:%M %p %Z") if game_dt.tzinfo else game_dt.strftime("%I:%M %p")
        formatted.append({
            "date": game_dt.strftime("%a, %b %d"),
            "time": formatted_time,
            "opponent": entry["opponent"],
            "opponent_abbr": entry["opponent_abbr"],
            "opponent_id": entry["opponent_id"],
            "home": entry["home"],
            "venue": entry["venue"],
            "series": entry["series"],
            "status": entry["status"],
            "game_pk": entry["game_pk"],
            "probable_pitchers": entry["probable_pitchers"],
            "reports": [],
            "game_date_iso": game_dt.date().isoformat(),
            "game_datetime_iso": game_dt.astimezone(timezone.utc).isoformat(),
        })
    return formatted


def _team_abbr_from_id(team_id: Optional[int]) -> Optional[str]:
    if not team_id:
        return None
    return (_team_directory().get(int(team_id)) or {}).get("abbr")


def ensure_secret_key() -> str:
    """Load or generate a persistent secret key for session signing."""
    env_secret = os.environ.get("APP_SECRET_KEY")
    if env_secret:
        return env_secret

    settings = settings_manager.load_settings()
    general = settings.get("general", {}) if isinstance(settings, dict) else {}
    secret_key = general.get("secret_key")

    if secret_key:
        return secret_key

    secret_key = secrets.token_hex(32)
    try:
        settings_manager.update_settings({"general": {"secret_key": secret_key}})
        refresh_settings_cache()
    except Exception as exc:  # pragma: no cover - best effort persistence
        print(f"Warning: Unable to persist generated secret key: {exc}")
    return secret_key


def generate_csrf_token() -> str:
    """Ensure a CSRF token exists in the session and return it."""
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_hex(32)
        session["csrf_token"] = token
    return token


def validate_csrf(token: str) -> bool:
    """Validate an incoming CSRF token."""
    if not token:
        return False
    return token == session.get("csrf_token")


def get_safe_redirect(default_endpoint: str = "home") -> str:
    """Return a safe redirect target within this application."""
    target = request.args.get("next") or request.form.get("next")
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for(default_endpoint)


PLAYER_DOCS_DIR = ROOT_DIR / "build" / "player_documents"
PLAYER_DOCS_DIR.mkdir(parents=True, exist_ok=True)

WORKOUT_DOCS_DIR = ROOT_DIR / "build" / "workouts"
WORKOUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)

WORKOUT_CATEGORY = "workout"
WORKOUT_ALLOWED_EXTENSIONS = {".pdf"}

SERIES_AUTO_DELETE_GRACE_SECONDS = 0

app.secret_key = ensure_secret_key()


@lru_cache(maxsize=1)
def get_cached_settings():
    """Return cached application settings."""
    return settings_manager.load_settings()


def refresh_settings_cache():
    """Clear cached settings so the next access reloads from disk."""
    get_cached_settings.cache_clear()


def ensure_default_admin():
    """Guarantee a default admin user exists for initial access."""
    if not PlayerDB:
        return

    default_email = os.environ.get("DEFAULT_ADMIN_EMAIL", "admin@sequencebiolab.com").strip().lower()
    default_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "1234")

    try:
        db = PlayerDB()
        existing = db.get_user_by_email(default_email)
        password_hash = generate_password_hash(default_password)

        if not existing:
            db.create_user(
                email=default_email,
                password_hash=password_hash,
                first_name="Sequence",
                last_name="Admin",
                is_admin=True
            )
            print(f"Default admin user created: {default_email}")
        else:
            if not existing.get("is_admin"):
                db.set_user_admin(existing["id"], True)
            if not check_password_hash(existing.get("password_hash", ""), default_password):
                db.update_user_password(existing["id"], password_hash)
        db.close()
    except Exception as exc:
        print(f"Warning: Unable to ensure default admin user: {exc}")


ensure_default_admin()

@app.before_request
def attach_settings_to_request():
    """Load settings for the current request context."""
    g.app_settings = get_cached_settings()
    # Ensure the CSRF token is primed for subsequent form usage
    generate_csrf_token()


@app.before_request
def load_authenticated_user():
    """Attach the currently authenticated user (if any) to the request context."""
    g.user = None
    session.setdefault("is_admin", False)
    user_id = session.get("user_id")
    if not user_id or not PlayerDB:
        session["is_admin"] = False
        return

    try:
        db = PlayerDB()
        g.user = db.get_user_by_id(user_id)
        db.close()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Failed to load user {user_id}: {exc}")
        g.user = None

    session["is_admin"] = bool(g.user.get("is_admin")) if g.user else False
    if g.user:
        session["first_name"] = g.user.get("first_name", "")
        session["last_name"] = g.user.get("last_name", "")
        if g.user.get("theme_preference"):
            session["theme_preference"] = g.user["theme_preference"]


@app.before_request
def enforce_global_authentication():
    """Redirect unauthenticated visitors to the login page for protected routes."""
    if session.get("user_id"):
        return

    endpoint = request.endpoint or ""

    if endpoint in AUTH_EXEMPT_ENDPOINTS:
        return

    if endpoint.startswith("static"):
        return

    # Allow access to favicon or other public assets served via send_file endpoints if any
    if endpoint in {"favicon"}:
        return

    # Avoid redirect loops when login/register POST fails
    if endpoint in {"login", "register"}:
        return

    next_path = request.path if request.path not in {url_for("login"), url_for("register")} else None
    flash("Please log in to continue.", "warning")
    return redirect(url_for("login", next=next_path))


@app.context_processor
def inject_app_settings():
    """Expose app settings and theme to templates."""
    settings = getattr(g, "app_settings", None) or get_cached_settings()
    general = settings.get("general", {}) if isinstance(settings, dict) else {}
    theme = general.get("theme", "dark")
    user = getattr(g, "user", None)
    user_theme = (user or {}).get("theme_preference")
    if user_theme:
        theme = user_theme
    return {
        "app_settings": settings,
        "app_theme": theme,
        "csrf_token": generate_csrf_token(),
        "current_user": user
    }


def login_required(fn):
    """Decorator to enforce authentication before accessing a view."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    """Decorator ensuring the current user has admin privileges."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        if not session.get("is_admin"):
            flash("Admin privileges are required to access that page.", "error")
            return redirect(url_for("home"))
        return fn(*args, **kwargs)

    return wrapper


def parse_bool(value, default=False):
    """Coerce a value into a boolean with a default fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def clean_str(value):
    """Return a trimmed string representation or empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _sanitize_filename_component(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "", (value or "")).strip()


def _current_player_full_name() -> Optional[str]:
    user = getattr(g, "user", None)
    if not user:
        return None
    first = (user.get("first_name") or "").strip()
    last = (user.get("last_name") or "").strip()
    parts = [part for part in (first, last) if part]
    if not parts:
        return None
    return " ".join(parts)


def _lookup_team_for_name(first_name: Optional[str], last_name: Optional[str]) -> Optional[str]:
    """Attempt to determine a team abbreviation for the given player name."""
    if not PlayerDB:
        return None
    first = (first_name or "").strip()
    last = (last_name or "").strip()
    if not last:
        return None
    try:
        db = PlayerDB()
        candidates = db.search_players(search=last, limit=50)
        team_abbr = None
        for candidate in candidates:
            cand_first = (candidate.get("first_name") or "").split()
            cand_last = (candidate.get("last_name") or "").split()
            # Basic matching on first + last
            cand_first_name = cand_first[0] if cand_first else ""
            cand_last_name = cand_last[-1] if cand_last else ""
            if cand_first_name and first and cand_first_name.lower() != first.lower():
                continue
            if cand_last_name and last and cand_last_name.lower() != last.lower():
                continue
            team_abbr = (candidate.get("team_abbr") or candidate.get("team") or "").strip().upper()
            if team_abbr:
                break
        db.close()
        return team_abbr or None
    except Exception as exc:
        print(f"Warning resolving team for {first_name} {last_name}: {exc}")
        return None


def _resolve_default_season_start() -> str:
    settings = getattr(g, "app_settings", {}) or {}
    report_defaults = settings.get("reports", {}) if isinstance(settings, dict) else {}
    return clean_str(report_defaults.get("default_season_start")) or "2025-03-20"


def _coerce_utc_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_game_datetime(game: Dict[str, Any]) -> Optional[datetime]:
    for key in ("game_datetime_iso", "game_datetime", "game_date_iso"):
        dt = _coerce_utc_datetime(game.get(key))
        if dt:
            return dt
    return None


def _report_exists_for_player(player_name: str, opponent_abbr: Optional[str], opponent_label: Optional[str]) -> bool:
    if not player_name:
        return True
    player_slug = _sanitize_filename_component(player_name).lower()
    player_slug_alt = player_slug.replace(" ", "_")
    opponent_code = (opponent_abbr or "").upper()
    stubs = []
    if opponent_label:
        stubs.append(_sanitize_filename_component(f"{player_name} vs {opponent_label}").lower())
    if opponent_code:
        stubs.append(_sanitize_filename_component(f"{player_name} vs {opponent_code}").lower())
    stubs_alt = [s.replace(" ", "_") for s in stubs]

    for pdf in OUT_DIR.glob("*.pdf"):
        stem_lower = pdf.stem.lower()
        if opponent_code and f"_vs_{opponent_code.lower()}" in stem_lower and (player_slug in stem_lower or player_slug_alt in stem_lower):
            return True
        for stub in stubs:
            if stub and stub in stem_lower:
                return True
        for stub in stubs_alt:
            if stub and stub in stem_lower:
                return True
    return False


def _schedule_auto_reports(games: List[Dict[str, Any]], team_abbr: Optional[str]) -> None:
    if not games:
        return
    player_name = _current_player_full_name()
    if not player_name:
        return

    season_start = _resolve_default_season_start()
    for game in games:
        _maybe_trigger_report(game, team_abbr, player_name, season_start)


def _maybe_trigger_report(game: Dict[str, Any], team_abbr: Optional[str], player_name: str, season_start: str) -> None:
    opponent_abbr = (game.get("opponent_abbr") or "").upper()
    opponent_label = game.get("opponent")
    if not opponent_abbr:
        return

    game_dt = _extract_game_datetime(game)
    if not game_dt:
        return

    now = datetime.now(timezone.utc)
    days_out = (game_dt.date() - now.date()).days
    if days_out < 0 or days_out > REPORT_LEAD_DAYS:
        return

    if _report_exists_for_player(player_name, opponent_abbr, opponent_label):
        return

    key = "|".join([
        player_name.lower(),
        opponent_abbr,
        str(game.get("game_pk") or game.get("game_date_iso") or game_dt.date().isoformat())
    ])

    with _PENDING_REPORT_LOCK:
        if key in _PENDING_REPORT_KEYS:
            return
        _PENDING_REPORT_KEYS.add(key)

    def _worker():
        try:
            filename = f"{_sanitize_filename_component(player_name).replace(' ', '_')}_vs_{opponent_abbr}.pdf"
            generate_single_report(
                hitter_name=player_name,
                team=team_abbr or "AUTO",
                season_start=season_start,
                use_next_series=False,
                opponent_team=opponent_abbr,
                pdf_name=filename
            )
        except Exception as exc:
            print(f"Warning: auto-generation failed for {player_name} vs {opponent_abbr}: {exc}")
        finally:
            with _PENDING_REPORT_LOCK:
                _PENDING_REPORT_KEYS.discard(key)

    thread = threading.Thread(target=_worker, name=f"report-auto-{opponent_abbr}-{game.get('game_pk')}", daemon=True)
    thread.start()


def _validate_auth_form_fields(email: str, password: str, first_name: str = "", last_name: str = "", confirm: str = ""):
    """Perform basic validation for authentication forms."""
    errors = []
    if not email:
        errors.append("Email is required.")
    elif "@" not in email or "." not in email.split("@")[-1]:
        errors.append("Please enter a valid email address.")

    if first_name is not None and not first_name:
        errors.append("First name is required.")
    if last_name is not None and not last_name:
        errors.append("Last name is required.")

    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long.")

    if confirm and password != confirm:
        errors.append("Password confirmation does not match.")

    return errors


def _determine_user_team(user):
    """Best-effort resolution of the user's team abbreviation."""
    team_abbr = None

    if user:
        team_abbr = _lookup_team_for_name(user.get("first_name"), user.get("last_name"))

    if not team_abbr:
        try:
            settings = getattr(g, "app_settings", {}) or get_cached_settings()
            report_defaults = settings.get("reports", {}) if isinstance(settings, dict) else {}
            default_team = (report_defaults.get("default_team") or "").strip().upper()
            if default_team and default_team != "AUTO":
                team_abbr = default_team
        except Exception:
            team_abbr = None

    if not team_abbr or team_abbr == "AUTO":
        team_abbr = "NYM"  # sensible default

    return team_abbr


def _collect_upcoming_games(team_abbr, limit=5):
    """Return a formatted list of upcoming games for the given team."""
    use_mock = bool(app.config.get("USE_MOCK_SCHEDULE"))
    cache_key = ("MOCK" if use_mock else "LIVE", (team_abbr or "").upper(), limit)
    cached = _cache_get(_UPCOMING_GAMES_CACHE, cache_key)
    if cached is not None:
        return cached

    if use_mock:
        formatted_mock = _build_mock_upcoming_games(team_abbr, limit)
        _cache_set(_UPCOMING_GAMES_CACHE, cache_key, formatted_mock, ttl_seconds=60)
        return formatted_mock

    if not next_games:
        return []

    try:
        games = next_games(team_abbr, days_ahead=14)
    except Exception as exc:
        print(f"Warning fetching upcoming games: {exc}")
        _cache_set(_UPCOMING_GAMES_CACHE, cache_key, [], ttl_seconds=120)
        return []

    formatted = []
    for game in games[:limit]:
        date_str = game.get("game_date")
        display_date = date_str
        display_time = "TBD"
        if date_str:
            try:
                display_date = datetime.fromisoformat(date_str).strftime("%a, %b %d")
            except Exception:
                pass

        game_time = game.get("game_datetime")
        if game_time:
            try:
                display_time = datetime.fromisoformat(game_time.replace("Z", "+00:00")).astimezone().strftime("%I:%M %p %Z")
            except Exception:
                display_time = "TBD"

        team_abbr_code = _team_abbr_from_id(game.get("opponent_id"))
        probables = [
            p.get("name")
            for p in (game.get("probable_pitchers") or [])
            if p.get("name")
        ]
        formatted.append({
            "date": display_date,
            "time": display_time,
            "opponent": game.get("opponent_name"),
            "opponent_abbr": team_abbr_code,
            "opponent_id": game.get("opponent_id"),
            "home": game.get("is_home"),
            "venue": game.get("venue"),
            "series": game.get("series_description"),
            "status": game.get("status"),
            "game_pk": game.get("game_pk"),
            "probable_pitchers": probables,
            "reports": [],
            "game_date_iso": date_str,
            "game_datetime_iso": game_time,
        })
    _cache_set(_UPCOMING_GAMES_CACHE, cache_key, formatted, ttl_seconds=300)
    return formatted


def _collect_series_for_team(team_abbr: Optional[str], days_ahead: int = 14) -> List[Dict[str, Any]]:
    """Group schedule into opponent series for selection purposes."""
    if not team_abbr:
        return []

    games: List[Dict[str, Any]] = []
    use_mock = bool(app.config.get("USE_MOCK_SCHEDULE"))

    if use_mock:
        raw = _build_mock_upcoming_games(team_abbr, limit=20)
        for item in raw:
            try:
                games.append({
                    "game_date": item.get("game_date_iso") or item.get("date"),
                    "game_datetime": item.get("game_datetime_iso"),
                    "opponent_id": item.get("opponent_id"),
                    "opponent_name": item.get("opponent"),
                    "is_home": item.get("home"),
                    "venue": item.get("venue"),
                    "series_description": item.get("series"),
                    "status": item.get("status"),
                })
            except Exception:
                continue
    else:
        if not next_games:
            return []
        try:
            games = next_games(team_abbr, days_ahead=days_ahead, include_started=True)
        except Exception as exc:
            print(f"Warning fetching series schedule: {exc}")
            games = []

    if not games:
        return []

    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    last_opponent = None
    for game in games:
        opponent = game.get("opponent_id")
        if last_opponent is None or opponent == last_opponent:
            current_chunk.append(game)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [game]
        last_opponent = opponent
    if current_chunk:
        chunks.append(current_chunk)

    now_ts = datetime.now(timezone.utc).timestamp()
    out: List[Dict[str, Any]] = []

    for chunk in chunks:
        if not chunk:
            continue
        first_game = chunk[0]
        last_game = chunk[-1]

        start_dt = _coerce_utc_datetime(
            first_game.get("game_datetime")
            or first_game.get("game_datetime_iso")
            or first_game.get("game_date")
            or first_game.get("game_date_iso")
        )
        end_dt = _coerce_utc_datetime(
            last_game.get("game_datetime")
            or last_game.get("game_datetime_iso")
            or last_game.get("game_date")
            or last_game.get("game_date_iso")
        )
        if not start_dt or not end_dt:
            continue

        start_ts = start_dt.timestamp()
        end_ts = end_dt.timestamp()

        # Skip series that finished before the grace window
        if end_ts < now_ts - SERIES_AUTO_DELETE_GRACE_SECONDS:
            continue

        opponent_id = first_game.get("opponent_id")
        opponent_name = first_game.get("opponent_name")
        opponent_abbr = _team_abbr_from_id(opponent_id) or (opponent_name or "")
        home_label = "Home vs" if first_game.get("is_home") else "Road @"
        series_label = f"{home_label} {opponent_name}".strip()

        def _fmt_range(dt: datetime) -> str:
            return dt.astimezone().strftime("%b %d")

        range_label = _fmt_range(start_dt)
        if end_dt.date() != start_dt.date():
            range_label = f"{range_label} – {_fmt_range(end_dt)}"

        status_lower_values = [str(game.get("status") or "").lower() for game in chunk]
        if any("final" in s or "game over" in s for s in status_lower_values):
            status = "expired"
        elif any("in progress" in s or "pre-game" in s or "pregame" in s or "pre game" in s or "delayed" in s for s in status_lower_values):
            status = "current"
        elif start_ts <= now_ts <= end_ts:
            status = "current"
        elif end_ts < now_ts:
            status = "expired"
        else:
            status = "upcoming"

        out.append({
            "id": f"{opponent_id}_{int(start_ts)}",
            "opponent_id": opponent_id,
            "opponent_name": opponent_name,
            "opponent_abbr": opponent_abbr,
            "is_home": bool(first_game.get("is_home")),
            "series_label": series_label,
            "series_description": first_game.get("series_description"),
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "range": range_label,
            "status": status,
            "game_count": len(chunk),
        })

    out.sort(key=lambda item: item.get("start") or "")
    return out


def _collect_recent_reports(limit=5):
    """Return recent generated reports with metadata."""
    reports = []
    try:
        pdf_files = sorted(OUT_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception as exc:
        print(f"Warning reading reports directory: {exc}")
        return reports

    for pdf in pdf_files[:limit]:
        try:
            mtime = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%b %d, %Y %I:%M %p")
        except Exception:
            mtime = "Unknown"

        reports.append({
            "title": pdf.stem.replace("_", " "),
            "generated_at": mtime,
            "filename": pdf.name,
        })
    return reports


def _attach_reports_to_games(
    games: List[Dict[str, Any]],
    reports: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Annotate each upcoming game with any available scouting reports."""
    report_map: Dict[str, List[Dict[str, Any]]] = {}
    for report in reports:
        filename = report.get("filename") or ""
        match = _REPORT_OPP_PATTERN.search(filename)
        if not match:
            continue
        abbr = match.group(1).upper()
        payload = {
            "title": report.get("title"),
            "generated_at": report.get("generated_at"),
            "url": report.get("url"),
        }
        report_map.setdefault(abbr, []).append(payload)

    for game in games:
        abbr = game.get("opponent_abbr")
        game["reports"] = report_map.get(abbr, [])
    return games


def _format_staff_note(note: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a staff note row for JSON/template contexts."""
    created_at = note.get("created_at")
    updated_at = note.get("updated_at")

    def _fmt(ts):
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).strftime("%b %d, %Y %I:%M %p")
        except Exception:
            return None

    def _iso(ts):
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None

    return {
        "id": note.get("id"),
        "title": note.get("title"),
        "body": note.get("body"),
        "team_abbr": note.get("team_abbr"),
        "tags": note.get("tags") or [],
        "author": note.get("author_name"),
        "pinned": bool(note.get("pinned")),
        "created_at": _fmt(created_at),
        "updated_at": _fmt(updated_at),
        "created_at_iso": _iso(created_at),
        "updated_at_iso": _iso(updated_at),
        "created_at_raw": created_at,
        "updated_at_raw": updated_at,
    }


def _collect_staff_notes(team_abbr: Optional[str], limit: int = 10):
    """Retrieve staff notes for display on the Gameday hub."""
    if not PlayerDB:
        return []

    try:
        db = PlayerDB()
        notes = db.list_staff_notes(team_abbr=team_abbr, limit=limit)
        db.close()
    except Exception as exc:
        print(f"Warning fetching staff notes: {exc}")
        return []

    return [_format_staff_note(note) for note in notes]


def _parse_leader_lines(raw_text, max_entries=5):
    """Parse the text output from statsapi.league_leaders into structured rows."""
    entries = []
    if not raw_text:
        return entries

    for line in raw_text.splitlines():
        if line.startswith("Rank") or not line.strip():
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        rank, name, team, value = parts[:4]
        entries.append({
            "rank": rank.strip(),
            "player": name.strip(),
            "team": team.strip(),
            "value": value.strip(),
        })
        if len(entries) >= max_entries:
            break
    return entries


def _filter_leader_entries(entries, include_pitchers=False):
    """Filter leader entries by primary position."""
    filtered = []
    for entry in entries:
        primary_position = (entry.get("position") or "").upper()
        if include_pitchers:
            if primary_position in {"", "P"}:
                filtered.append(entry)
        else:
            if primary_position and primary_position != "P":
                filtered.append(entry)
            elif not primary_position:
                filtered.append(entry)

    return filtered[:5]


def _fetch_leader_entries(category, stat_group, limit=5):
    """Fetch structured leader data directly from the MLB stats API."""
    params = {
        "leaderCategories": category,
        "season": datetime.now().year,
        "sportId": 1,
        "statGroup": stat_group,
        "leaderGameTypes": "R",
        "leaderBoardType": "regularSeason",
        "limit": max(limit, 5),
    }
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/stats/leaders",
            params=params,
            timeout=6
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        print(f"Warning fetching leaders {category}/{stat_group}: {exc}")
        return []

    leaders_block = (payload.get("leagueLeaders") or [])
    if not leaders_block:
        return []

    leaders = []
    for entry in leaders_block[0].get("leaders", [])[:limit]:
        person = entry.get("person") or {}
        team = entry.get("team") or {}
        leaders.append({
            "rank": entry.get("rank"),
            "player": person.get("fullName"),
            "team": team.get("name"),
            "value": entry.get("value"),
            "player_id": person.get("id"),
            "position": ((person.get("primaryPosition") or {}).get("abbreviation") or "")
        })
    return leaders


def _collect_league_leaders():
    """Get curated hitting and pitching leaderboards."""
    cached = _cache_get(_LEAGUE_LEADERS_CACHE, "default")
    if cached is not None:
        return cached

    groups = [
        ("Hitting Leaders", [
            ("homeRuns", "Home Runs"),
            ("runsBattedIn", "Runs Batted In"),
            ("battingAverage", "Batting Average"),
        ]),
        ("Pitching Leaders", [
            ("era", "Earned Run Average"),
            ("strikeouts", "Strikeouts"),
            ("whip", "WHIP"),
        ])
    ]

    result = []
    for group_label, categories in groups:
        category_entries = []
        for stat_code, label in categories:
            stat_group = "hitting" if group_label.startswith("Hitting") else "pitching"
            entries = _fetch_leader_entries(stat_code, stat_group)
            if group_label.startswith("Hitting"):
                entries = _filter_leader_entries(entries, include_pitchers=False)
            else:
                entries = _filter_leader_entries(entries, include_pitchers=True)
            if entries:
                category_entries.append({
                    "label": label,
                    "abbr": LEADER_CATEGORY_ABBR.get(stat_code, "Value"),
                    "entries": entries
                })
        if category_entries:
            result.append({
                "group": group_label,
                "categories": category_entries
            })

    _cache_set(_LEAGUE_LEADERS_CACHE, "default", result, ttl_seconds=600)
    return result


def _get_team_metadata(team_abbr):
    """Fetch MLB metadata for a given team abbreviation."""
    team_id = TEAM_ABBR_TO_ID.get((team_abbr or "").upper())
    if not team_id:
        return {}
    cache_key = (team_abbr or "").upper()
    cached = _cache_get(_TEAM_METADATA_CACHE, cache_key)
    if cached is not None:
        return cached
    try:
        team_payload = statsapi.get("team", {"teamId": team_id})
        team_info = (team_payload.get("teams") or [{}])[0]
        division = team_info.get("division") or {}
        league = team_info.get("league") or {}
        payload = {
            "team_id": team_id,
            "team_name": team_info.get("name"),
            "division_id": division.get("id"),
            "division_name": division.get("name"),
            "league_id": league.get("id"),
            "league_name": league.get("name")
        }
        _cache_set(_TEAM_METADATA_CACHE, cache_key, payload, ttl_seconds=3600)
        return payload
    except Exception as exc:
        print(f"Warning fetching team metadata: {exc}")
        payload = {"team_id": team_id}
        _cache_set(_TEAM_METADATA_CACHE, cache_key, payload, ttl_seconds=900)
        return payload


def _collect_standings_data(view, team_metadata, division_id=None, league_id=None):
    """Return standings data for either a division or wildcard view."""
    season = datetime.now().year
    team_id = (team_metadata or {}).get("team_id")
    default_league_id = (team_metadata or {}).get("league_id")

    if view == "wildcard":
        league_id = int(league_id or default_league_id or 104)
        cache_key = ("wildcard", team_id, None, league_id)
        cached = _cache_get(_STANDINGS_CACHE, cache_key)
        if cached is not None:
            return cached
        try:
            standings_payload = statsapi.get("standings", {
                "leagueId": league_id,
                "season": season,
                "standingsType": "wildCard"
            })
        except Exception as exc:
            print(f"Warning fetching wildcard standings: {exc}")
            return None

        team_records = []
        for record in standings_payload.get("records", []):
            if record.get("league", {}).get("id") == league_id:
                team_records.extend(record.get("teamRecords", []))

        if not team_records:
            return None

        def rank_key(entry):
            try:
                return int(entry.get("wildCardRank", 999))
            except (TypeError, ValueError):
                return 999

        team_records.sort(key=rank_key)

        rows = []
        for entry in team_records:
            gb = entry.get("wildCardGamesBack")
            if gb in (None, "-", ""):
                gb = "0"
            rows.append({
                "team": entry.get("team", {}).get("name"),
                "wins": entry.get("wins"),
                "losses": entry.get("losses"),
                "games_back": gb,
                "is_user_team": entry.get("team", {}).get("id") == team_id
            })

        payload = {
            "title": f"{next((l['name'] for l in LEAGUE_OPTIONS if l['id'] == league_id), 'League')} Wild Card",
            "rows": rows,
            "view": "wildcard",
            "division_id": None,
            "league_id": league_id
        }
        _cache_set(_STANDINGS_CACHE, cache_key, payload, ttl_seconds=600)
        return payload

    # Division view
    resolved_division_id = division_id or (team_metadata or {}).get("division_id") or DIVISION_OPTIONS[0]["id"]
    try:
        division_id = int(resolved_division_id)
    except (TypeError, ValueError):
        division_id = DIVISION_OPTIONS[0]["id"]

    league_id = default_league_id or 104
    division_meta = next((opt for opt in DIVISION_OPTIONS if opt["id"] == division_id), None)
    if division_meta:
        league_id = division_meta.get("league_id") or league_id

    league_id = int(league_id)
    cache_key = ("division", team_id, division_id, league_id)
    cached = _cache_get(_STANDINGS_CACHE, cache_key)
    if cached is not None:
        return cached

    try:
        standings_payload = statsapi.get("standings", {
            "leagueId": league_id,
            "season": season
        })
    except Exception as exc:
        print(f"Warning fetching division standings: {exc}")
        return None

    division_record = None
    for record in standings_payload.get("records", []):
        if record.get("division", {}).get("id") == division_id:
            division_record = record
            break

    if not division_record:
        return None

    rows = []
    for entry in division_record.get("teamRecords", []):
        gb = entry.get("gamesBack")
        if gb in (None, "-", ""):
            gb = "0"
        rows.append({
            "team": entry.get("team", {}).get("name"),
            "wins": entry.get("wins"),
            "losses": entry.get("losses"),
            "games_back": gb,
            "is_user_team": entry.get("team", {}).get("id") == team_id
        })

    payload = {
        "title": division_record.get("division", {}).get("name", "Division"),
        "rows": rows,
        "view": "division",
        "division_id": division_id,
        "league_id": league_id
    }
    _cache_set(_STANDINGS_CACHE, cache_key, payload, ttl_seconds=600)
    return payload


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration endpoint."""
    if not PlayerDB:
        flash("Database unavailable. Please contact support.", "error")
        return redirect(url_for("home"))

    if request.method == 'POST':
        if not validate_csrf(request.form.get("csrf_token")):
            flash("Invalid form submission. Please try again.", "error")
            return redirect(url_for('register'))

        email = clean_str(request.form.get('email', '')).lower()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''
        first_name = clean_str(request.form.get('first_name', ''))
        last_name = clean_str(request.form.get('last_name', ''))

        errors = _validate_auth_form_fields(email, password, first_name, last_name, confirm_password)
        if errors:
            for error in errors:
                flash(error, "error")
            return redirect(url_for('register', next=request.form.get('next')))

        try:
            db = PlayerDB()
            existing = db.get_user_by_email(email)
            if existing:
                flash("An account with that email already exists. Please sign in.", "error")
                return redirect(url_for('login'))

            password_hash = generate_password_hash(password)
            user_id = db.create_user(email, password_hash, first_name, last_name, is_admin=False)
        except Exception as exc:
            flash(f"Could not create account: {exc}", "error")
            return redirect(url_for('register'))
        finally:
            try:
                db.close()
            except Exception:
                pass

        # Rotate CSRF token and set session values
        session.pop('csrf_token', None)
        session['user_id'] = user_id
        session['first_name'] = first_name
        session['last_name'] = last_name
        session['is_admin'] = False
        generate_csrf_token()

        flash("Welcome to Sequence BioLab!", "success")
        return redirect(get_safe_redirect())

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login endpoint."""
    if not PlayerDB:
        flash("Database unavailable. Please contact support.", "error")
        return redirect(url_for("home"))

    if request.method == 'POST':
        if not validate_csrf(request.form.get("csrf_token")):
            flash("Invalid form submission. Please try again.", "error")
            return redirect(url_for('login'))

        email = clean_str(request.form.get('email', '')).lower()
        password = request.form.get('password') or ''

        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for('login', next=request.form.get('next')))

        user = None
        try:
            db = PlayerDB()
            user = db.get_user_by_email(email)
        except Exception as exc:
            flash(f"Login failed: {exc}", "error")
            return redirect(url_for('login'))
        finally:
            try:
                db.close()
            except Exception:
                pass

        if not user or not check_password_hash(user['password_hash'], password):
            flash("Invalid email or password.", "error")
            return redirect(url_for('login', next=request.form.get('next')))

        session.pop('csrf_token', None)
        session['user_id'] = user['id']
        session['first_name'] = user['first_name']
        session['last_name'] = user['last_name']
        session['is_admin'] = bool(user.get('is_admin'))
        generate_csrf_token()

        flash("Signed in successfully.", "success")
        return redirect(get_safe_redirect())

    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    """Log out the current user."""
    if not validate_csrf(request.form.get("csrf_token")):
        flash("Invalid logout request.", "error")
        return redirect(url_for('home'))

    for key in ("user_id", "first_name", "last_name", "is_admin"):
        session.pop(key, None)
    session.pop('csrf_token', None)
    generate_csrf_token()

    flash("You have been logged out.", "info")
    return redirect(url_for('home'))


def generate_single_report(hitter_name, team="AUTO", season_start="2025-03-20", use_next_series=False, opponent_team=None, pdf_name=None):
    """Generate a single report and return the PDF path"""
    try:
        # Activate virtual environment and run the report generation
        venv_python = ROOT_DIR / "venv" / "bin" / "python3"
        if not venv_python.exists():
            venv_python = "python3"
        
        script_path = ROOT_DIR / "src" / "generate_report.py"
        template_path = ROOT_DIR / "src" / "templates" / "hitter_report.html"
        
        cmd = [
            str(venv_python),
            str(script_path),
            "--team", team,
            "--hitter", hitter_name,
            "--season_start", season_start,
            "--out", str(OUT_DIR),
            "--template", str(template_path)
        ]
        
        if pdf_name:
            cmd.extend(["--pdf_name", pdf_name])

        if opponent_team and opponent_team.strip():
            cmd.extend(["--opponent", opponent_team.strip()])
        
        if use_next_series:
            cmd.append("--use-next-series")
        
        # Run the command with environment variable to suppress urllib3 warnings
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::UserWarning:urllib3,ignore::Warning'
        
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR / "src"),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        # First, check if PDF was generated (even if returncode != 0, warnings might cause non-zero exit)
        output_lines = result.stdout.split('\n')
        pdf_path = None
        
        for line in output_lines:
            if "Saved report:" in line:
                pdf_path = line.split("Saved report:")[-1].strip()
                break
        
        # If we can't find it from output, try to find it by name
        if not pdf_path or not Path(pdf_path).exists():
            # Look for PDFs with the player's name
            safe_name = hitter_name.replace(' ', '_').replace('"', '')
            pdf_files = list(OUT_DIR.glob(f"*{safe_name}*.pdf"))
            if pdf_files:
                # Sort by modification time and get the most recent
                pdf_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                pdf_path = str(pdf_files[0])
        
        # If PDF was generated, return success regardless of returncode
        if pdf_path and Path(pdf_path).exists():
            return {
                "success": True,
                "pdf_path": pdf_path,
                "pdf_filename": Path(pdf_path).name
            }
        
        # If no PDF found and returncode != 0, report the error
        if result.returncode != 0:
            # Combine stderr and stdout to get full error
            error_msg = ""
            if result.stderr:
                error_msg += result.stderr
            if result.stdout and result.stdout.strip():
                error_msg += "\n" + result.stdout if error_msg else result.stdout
            
            if not error_msg.strip():
                error_msg = "Unknown error"
            
            # Filter out urllib3 warnings - they're not fatal errors
            lines = error_msg.split('\n')
            filtered_lines = []
            skip_next_n_lines = 0
            
            for i, line in enumerate(lines):
                # Skip lines after warnings.warn( calls
                if skip_next_n_lines > 0:
                    skip_next_n_lines -= 1
                    continue
                
                # Skip urllib3/OpenSSL warnings
                if any(keyword in line for keyword in ['NotOpenSSLWarning', 'urllib3', 'site-packages/urllib3', 'OpenSSL']):
                    # If we see warnings.warn(, skip the next few lines too
                    if 'warnings.warn(' in line:
                        skip_next_n_lines = 2
                    continue
                
                # Skip lines that are just file paths to urllib3
                if line.strip().startswith('/Users') and ('urllib3' in line or 'site-packages' in line):
                    continue
                
                # Skip warning-related lines
                if 'warnings.warn(' in line or '__init__.py' in line and 'site-packages' in line:
                    skip_next_n_lines = 1
                    continue
                
                # Keep non-empty lines that aren't warnings
                if line.strip() and not line.strip().startswith('warnings.'):
                    filtered_lines.append(line)
            
            # If we filtered everything, keep some context
            if filtered_lines:
                filtered_error = '\n'.join(filtered_lines)
            else:
                # If only warnings were present, check if stdout has useful info
                if result.stdout and result.stdout.strip():
                    filtered_error = result.stdout.strip()
                else:
                    filtered_error = "Report generation failed (check logs for details)"
            
            # Get the actual error message (last meaningful line or traceback)
            error_lines = filtered_error.split('\n')
            # Look for the actual exception message
            actual_error = None
            for i in range(len(error_lines) - 1, -1, -1):
                line = error_lines[i].strip()
                if line and not line.startswith('File ') and not line.startswith('Traceback'):
                    if 'Error' in line or 'Exception' in line or ':' in line:
                        actual_error = line
                        break
            
            if actual_error:
                filtered_error = actual_error + "\n\n" + filtered_error[:300]
            
            return {
                "success": False,
                "error": f"Error generating report: {filtered_error[:800]}"
            }
        
        # If returncode is 0 but no PDF found
        return {
            "success": False,
            "error": "Report generation completed but PDF file not found"
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Report generation timed out (over 5 minutes)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }

def generate_report(hitter_name, team="AUTO", season_start="2025-03-20", use_next_series=False, opponent_team=None, job_id=None):
    """Generate a single report in the background (for backward compatibility)"""
    result = generate_single_report(hitter_name, team, season_start, use_next_series, opponent_team)
    
    if result["success"]:
        job_status[job_id] = {
            "status": "completed",
            "message": "Report generated successfully!",
            "pdf_path": result["pdf_path"],
            "pdf_filename": result["pdf_filename"]
        }
    else:
        job_status[job_id] = {
            "status": "error",
            "message": result.get("error", "Unknown error")
        }

def parse_player_entry(entry):
    """Parse a player entry that may include team and opponent.
    
    Format: "Player Name | Team | Opponent"
    All parts are optional except player name.
    Returns: (player_name, team, opponent)
    """
    parts = [p.strip() for p in entry.split('|')]
    player_name = parts[0].strip() if parts else ""
    team = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    opponent = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    
    return player_name, team, opponent

def generate_batch_reports(player_entries, default_team, season_start, use_next_series, default_opponent, job_id):
    """Generate reports for multiple players with individual settings"""
    total = len(player_entries)
    completed = 0
    failed = 0
    pdfs = []
    errors = []
    
    for i, entry in enumerate(player_entries):
        # Parse player entry
        hitter_name, team, opponent = parse_player_entry(entry)
        
        if not hitter_name:
            errors.append({
                "player": entry,
                "error": "Invalid format: missing player name"
            })
            failed += 1
            continue
        
        # Use per-player settings if provided, otherwise use defaults
        player_team = team if team else default_team
        player_opponent = opponent if opponent else default_opponent
        
        try:
            # Update status with player-specific info
            status_msg = f"Generating report {i+1} of {total}: {hitter_name}"
            if player_opponent:
                status_msg += f" vs {player_opponent}"
            
            job_status[job_id] = {
                "status": "running",
                "message": status_msg,
                "total": total,
                "completed": completed,
                "failed": failed,
                "current": hitter_name,
                "current_index": i + 1,
                "pdfs": pdfs,
                "errors": errors
            }
            
            # Generate single report with this player's settings
            result = generate_single_report(hitter_name, player_team, season_start, use_next_series, player_opponent)
            
            if result["success"]:
                pdfs.append({
                    "player": hitter_name,
                    "team": player_team,
                    "opponent": player_opponent,
                    "path": result["pdf_path"],
                    "filename": result["pdf_filename"]
                })
                completed += 1
            else:
                errors.append({
                    "player": hitter_name,
                    "error": result.get("error", "Unknown error")
                })
                failed += 1
                
        except Exception as e:
            errors.append({
                "player": hitter_name,
                "error": str(e)
            })
            failed += 1
    
    # Update final status
    job_status[job_id] = {
        "status": "completed",
        "message": f"Completed {completed} of {total} reports" + (f" ({failed} failed)" if failed > 0 else ""),
        "total": total,
        "completed": completed,
        "failed": failed,
        "pdfs": pdfs,
        "errors": errors
    }

# ============================================================================
# PITCHER REPORT GENERATION FUNCTIONS (separate from hitter reports)
# ============================================================================

def generate_single_pitcher_report(pitcher_name, team="AUTO", season_start="2025-03-20", use_next_series=False, opponent_team=None):
    """Generate a single pitcher report and return the PDF path"""
    try:
        # Activate virtual environment and run the report generation
        venv_python = ROOT_DIR / "venv" / "bin" / "python3"
        if not venv_python.exists():
            venv_python = "python3"
        
        script_path = ROOT_DIR / "src" / "generate_pitcher_report.py"
        template_path = ROOT_DIR / "src" / "templates" / "pitcher_report.html"
        
        cmd = [
            str(venv_python),
            str(script_path),
            "--team", team,
            "--pitcher", pitcher_name,
            "--season_start", season_start,
            "--out", str(OUT_DIR),
            "--template", str(template_path)
        ]
        
        if opponent_team and opponent_team.strip():
            cmd.extend(["--opponent", opponent_team.strip()])
        
        if use_next_series:
            cmd.append("--use-next-series")
        
        # Run the command with environment variable to suppress urllib3 warnings
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::UserWarning:urllib3,ignore::Warning'
        
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR / "src"),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        # First, check if PDF was generated (even if returncode != 0, warnings might cause non-zero exit)
        output_lines = result.stdout.split('\n')
        pdf_path = None
        
        for line in output_lines:
            if "Saved report:" in line:
                pdf_path = line.split("Saved report:")[-1].strip()
                break
        
        # If we can't find it from output, try to find it by name
        if not pdf_path or not Path(pdf_path).exists():
            # Look for PDFs with the player's name
            safe_name = pitcher_name.replace(' ', '_').replace('"', '')
            pdf_files = list(OUT_DIR.glob(f"*{safe_name}*.pdf"))
            if pdf_files:
                # Sort by modification time and get the most recent
                pdf_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                pdf_path = str(pdf_files[0])
        
        # If PDF was generated, return success regardless of returncode
        if pdf_path and Path(pdf_path).exists():
            return {
                "success": True,
                "pdf_path": pdf_path,
                "pdf_filename": Path(pdf_path).name
            }
        
        # If no PDF found and returncode != 0, report the error
        if result.returncode != 0:
            # Combine stderr and stdout to get full error
            error_msg = ""
            if result.stderr:
                error_msg += result.stderr
            if result.stdout and result.stdout.strip():
                error_msg += "\n" + result.stdout if error_msg else result.stdout
            
            if not error_msg.strip():
                error_msg = "Unknown error"
            
            # Filter out urllib3 warnings - they're not fatal errors
            lines = error_msg.split('\n')
            filtered_lines = []
            skip_next_n_lines = 0
            
            for i, line in enumerate(lines):
                # Skip lines after warnings.warn( calls
                if skip_next_n_lines > 0:
                    skip_next_n_lines -= 1
                    continue
                
                # Skip urllib3/OpenSSL warnings
                if any(keyword in line for keyword in ['NotOpenSSLWarning', 'urllib3', 'site-packages/urllib3', 'OpenSSL']):
                    # If we see warnings.warn(, skip the next few lines too
                    if 'warnings.warn(' in line:
                        skip_next_n_lines = 2
                    continue
                
                # Skip lines that are just file paths to urllib3
                if line.strip().startswith('/Users') and ('urllib3' in line or 'site-packages' in line):
                    continue
                
                # Skip warning-related lines
                if 'warnings.warn(' in line or '__init__.py' in line and 'site-packages' in line:
                    skip_next_n_lines = 1
                    continue
                
                # Keep non-empty lines that aren't warnings
                if line.strip() and not line.strip().startswith('warnings.'):
                    filtered_lines.append(line)
            
            # If we filtered everything, keep some context
            if filtered_lines:
                filtered_error = '\n'.join(filtered_lines)
            else:
                # If only warnings were present, check if stdout has useful info
                if result.stdout and result.stdout.strip():
                    filtered_error = result.stdout.strip()
                else:
                    filtered_error = "Report generation failed (check logs for details)"
            
            # Get the actual error message (last meaningful line or traceback)
            error_lines = filtered_error.split('\n')
            # Look for the actual exception message
            actual_error = None
            for i in range(len(error_lines) - 1, -1, -1):
                line = error_lines[i].strip()
                if line and not line.startswith('File ') and not line.startswith('Traceback'):
                    if 'Error' in line or 'Exception' in line or ':' in line:
                        actual_error = line
                        break
            
            if actual_error:
                filtered_error = actual_error + "\n\n" + filtered_error[:300]
            
            return {
                "success": False,
                "error": f"Error generating report: {filtered_error[:800]}"
            }
        
        # If returncode is 0 but no PDF found
        return {
            "success": False,
            "error": "Report generation completed but PDF file not found"
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Report generation timed out (over 5 minutes)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }

def generate_pitcher_report(pitcher_name, team="AUTO", season_start="2025-03-20", use_next_series=False, opponent_team=None, job_id=None):
    """Generate a single pitcher report in the background"""
    result = generate_single_pitcher_report(pitcher_name, team, season_start, use_next_series, opponent_team)
    
    if result["success"]:
        job_status[job_id] = {
            "status": "completed",
            "message": "Report generated successfully!",
            "pdf_path": result["pdf_path"],
            "pdf_filename": result["pdf_filename"]
        }
    else:
        job_status[job_id] = {
            "status": "error",
            "message": result.get("error", "Unknown error")
        }

def parse_pitcher_entry(entry):
    """Parse a pitcher entry that may include team and opponent.
    
    Format: "Pitcher Name | Team | Opponent"
    All parts are optional except pitcher name.
    Returns: (pitcher_name, team, opponent)
    """
    parts = [p.strip() for p in entry.split('|')]
    pitcher_name = parts[0].strip() if parts else ""
    team = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    opponent = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    
    return pitcher_name, team, opponent

def generate_batch_pitcher_reports(pitcher_entries, default_team, season_start, use_next_series, default_opponent, job_id):
    """Generate pitcher reports for multiple players with individual settings"""
    total = len(pitcher_entries)
    completed = 0
    failed = 0
    pdfs = []
    errors = []
    
    for i, entry in enumerate(pitcher_entries):
        # Parse pitcher entry
        pitcher_name, team, opponent = parse_pitcher_entry(entry)
        
        if not pitcher_name:
            errors.append({
                "pitcher": entry,
                "player": entry,
                "error": "Invalid format: missing pitcher name"
            })
            failed += 1
            continue
        
        # Use per-player settings if provided, otherwise use defaults
        player_team = team if team else default_team
        player_opponent = opponent if opponent else default_opponent
        
        try:
            # Update status with player-specific info
            status_msg = f"Generating report {i+1} of {total}: {pitcher_name}"
            if player_opponent:
                status_msg += f" vs {player_opponent}"
            
            job_status[job_id] = {
                "status": "running",
                "message": status_msg,
                "total": total,
                "completed": completed,
                "failed": failed,
                "current": pitcher_name,
                "current_index": i + 1,
                "pdfs": pdfs,
                "errors": errors
            }
            
            # Generate single report with this pitcher's settings
            result = generate_single_pitcher_report(pitcher_name, player_team, season_start, use_next_series, player_opponent)
            
            if result["success"]:
                pdfs.append({
                    "pitcher": pitcher_name,
                    "player": pitcher_name,
                    "team": player_team,
                    "opponent": player_opponent,
                    "path": result["pdf_path"],
                    "filename": result["pdf_filename"]
                })
                completed += 1
            else:
                errors.append({
                    "pitcher": pitcher_name,
                    "player": pitcher_name,
                    "error": result.get("error", "Unknown error")
                })
                failed += 1
                
        except Exception as e:
            errors.append({
                "pitcher": pitcher_name,
                "player": pitcher_name,
                "error": str(e)
            })
            failed += 1
    
    # Update final status
    job_status[job_id] = {
        "status": "completed",
        "message": f"Completed {completed} of {total} reports" + (f" ({failed} failed)" if failed > 0 else ""),
        "total": total,
        "completed": completed,
        "failed": failed,
        "pdfs": pdfs,
        "errors": errors
    }

def _get_player_headshot_url(user: Dict[str, Any]) -> Optional[str]:
    """Get the player's headshot URL from MLB.com."""
    first_name = (user.get("first_name") or "").strip()
    last_name = (user.get("last_name") or "").strip()
    
    if not first_name or not last_name:
        return None
    
    player_name = f"{first_name} {last_name}"
    
    try:
        # Use the more reliable lookup_batter_id function
        from scrape_savant import lookup_batter_id
        player_id = lookup_batter_id(player_name)
        if player_id:
            # Return MLB headshot URL
            return f"https://img.mlbstatic.com/mlb-photos/image/upload/f_png,w_213,q_100/v1/people/{player_id}/headshot/67/current"
    except Exception as e:
        print(f"Warning: Could not get player headshot for {player_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def _build_player_home_context(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not user:
        return {}

    next_series = _load_next_series_snapshot(user)
    latest_document, deliverables, outstanding_count = _load_player_deliverables(user)
    performance = _build_performance_snapshot()
    journal_entries = _load_journal_preview(user)
    resources = _load_resource_links(user)
    support_team = _load_support_contacts()
    focus_highlights = _build_focus_highlights(next_series, latest_document, outstanding_count)
    player_news = _load_player_news(user)
    player_headshot_url = _get_player_headshot_url(user)
    # Use local MLB logo file
    mlb_logo_url = url_for('static', filename='MLB_Logo.png')
    schedule_calendar = _load_schedule_calendar(user)

    return {
        "hero_message": None,
        "next_series": next_series,
        "latest_document": latest_document,
        "deliverables": deliverables,
        "outstanding_count": outstanding_count,
        "performance": performance,
        "journal_entries": journal_entries,
        "resources": resources,
        "support_team": support_team,
        "focus_highlights": focus_highlights,
        "player_news": player_news,
        "player_headshot_url": player_headshot_url,
        "mlb_logo_url": mlb_logo_url,
        "schedule_calendar": schedule_calendar,
    }


def _load_next_series_snapshot(user: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        team_abbr = _determine_user_team(user)
        series_list = _collect_series_for_team(team_abbr, days_ahead=21)
    except Exception as exc:
        print(f"Warning building next series snapshot: {exc}")
        series_list = []

    next_active = None
    for series in series_list:
        if series.get("status") != "expired":
            next_active = series
            break

    if not next_active:
        return None

    start_dt = _coerce_utc_datetime(next_active.get("start"))
    end_dt = _coerce_utc_datetime(next_active.get("end"))
    now_local = datetime.now(timezone.utc).astimezone()
    start_local = start_dt.astimezone(now_local.tzinfo) if start_dt else None
    end_local = end_dt.astimezone(now_local.tzinfo) if end_dt else None
    if start_local:
        days_until = max(0, (start_local.date() - now_local.date()).days)
    else:
        days_until = None

    def _fmt(dt: Optional[datetime]) -> Optional[str]:
        if not dt:
            return None
        return dt.strftime("%b %d")

    return {
        "opponent_name": next_active.get("opponent_name"),
        "opponent_id": next_active.get("opponent_id"),
        "opponent_abbr": next_active.get("opponent_abbr"),
        "start_date": _fmt(start_local),
        "end_date": _fmt(end_local),
        "status": next_active.get("status"),
        "days_until": days_until,
    }


def _load_schedule_calendar(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load schedule data for calendar widget (full month)."""
    try:
        team_abbr = _determine_user_team(user)
        use_mock = bool(app.config.get("USE_MOCK_SCHEDULE"))
        
        # Get games - use mock or real data
        games = []
        if use_mock:
            # Get mock games with raw date data (same pattern as schedule page)
            now = datetime.now().astimezone()
            base_first_pitch = now.replace(hour=19, minute=10, second=0, microsecond=0)
            
            mock_blueprint = [
                {"days_offset": 0, "opponent": "Washington Nationals", "opponent_abbr": "WSH", "opponent_id": 120, "is_home": True, "status": "Scheduled"},
                {"days_offset": 1, "opponent": "Washington Nationals", "opponent_abbr": "WSH", "opponent_id": 120, "is_home": True, "status": "Scheduled"},
                {"days_offset": 3, "opponent": "Philadelphia Phillies", "opponent_abbr": "PHI", "opponent_id": 143, "is_home": False, "status": "Scheduled"},
                {"days_offset": 4, "opponent": "Philadelphia Phillies", "opponent_abbr": "PHI", "opponent_id": 143, "is_home": False, "status": "Scheduled"},
                {"days_offset": 6, "opponent": "Atlanta Braves", "opponent_abbr": "ATL", "opponent_id": 144, "is_home": True, "status": "Scheduled"},
                {"days_offset": 8, "opponent": "Miami Marlins", "opponent_abbr": "MIA", "opponent_id": 146, "is_home": True, "status": "Scheduled"},
                {"days_offset": 10, "opponent": "New York Yankees", "opponent_abbr": "NYY", "opponent_id": 147, "is_home": False, "status": "Scheduled"},
            ]
            
            # Generate games for the next 60 days to cover multiple months
            # Repeat the pattern every 14 days to create a realistic schedule (same as schedule page)
            for week_offset in range(0, 5):  # 5 weeks = ~35 days, extend to 60
                for entry in mock_blueprint:
                    days_offset = entry["days_offset"] + (week_offset * 14)
                    
                    game_dt = base_first_pitch + timedelta(days=days_offset)
                    games.append({
                        "game_date": game_dt.date().isoformat(),
                        "game_datetime": game_dt.isoformat(),
                        "opponent_name": entry["opponent"],
                        "opponent_abbr": entry["opponent_abbr"],
                        "opponent_id": entry["opponent_id"],
                        "is_home": entry["is_home"],
                        "status": entry.get("status", "Scheduled"),
                        "venue": "TBD",
                    })
        elif next_games:
            # Get 60 days of games to cover full months
            games = next_games(team_abbr, days_ahead=60, include_started=True)
        else:
            return []
        
        # Format for calendar display
        calendar_data = []
        for game in games:
            game_date = game.get("game_date")
            if game_date:
                try:
                    # Handle both ISO string and date object
                    if isinstance(game_date, str):
                        dt = datetime.fromisoformat(game_date)
                    else:
                        dt = game_date if isinstance(game_date, datetime) else datetime.combine(game_date, datetime.min.time())
                    
                    opponent_id = game.get("opponent_id")
                    opponent_abbr = game.get("opponent_abbr", "")
                    # Derive opponent_abbr from opponent_id if not present
                    if not opponent_abbr and opponent_id:
                        opponent_abbr = _team_abbr_from_id(opponent_id) or ""
                    
                    calendar_data.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "day": dt.strftime("%d"),
                        "day_name": dt.strftime("%a"),
                        "opponent": game.get("opponent_name", ""),
                        "opponent_abbr": opponent_abbr,
                        "opponent_id": opponent_id,
                        "is_home": game.get("is_home", False),
                        "venue": game.get("venue", ""),
                        "status": game.get("status", "Scheduled"),
                    })
                except Exception as e:
                    print(f"Warning formatting game date {game_date}: {e}")
                    continue
        
        return calendar_data
    except Exception as e:
        print(f"Warning loading schedule calendar: {e}")
        import traceback
        traceback.print_exc()
        return []


def _load_full_season_schedule(user: Dict[str, Any], start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load full season schedule, optionally filtered by date range."""
    try:
        team_abbr = _determine_user_team(user)
        use_mock = bool(app.config.get("USE_MOCK_SCHEDULE"))
        
        # Default to full season (March to October)
        if not start_date:
            current_year = datetime.now().year
            start_date = f"{current_year}-03-01"
        if not end_date:
            current_year = datetime.now().year
            end_date = f"{current_year}-10-31"
        
        # Calculate days between dates
        start_dt = datetime.fromisoformat(start_date).date()
        end_dt = datetime.fromisoformat(end_date).date()
        today = datetime.now().date()
        
        # Get games - use mock or real data
        games = []
        if use_mock:
            # Get mock games with raw date data (same as calendar widget)
            now = datetime.now().astimezone()
            base_first_pitch = now.replace(hour=19, minute=10, second=0, microsecond=0)
            
            mock_blueprint = [
                {"days_offset": 0, "opponent": "Washington Nationals", "opponent_abbr": "WSH", "opponent_id": 120, "is_home": True, "status": "Scheduled"},
                {"days_offset": 1, "opponent": "Washington Nationals", "opponent_abbr": "WSH", "opponent_id": 120, "is_home": True, "status": "Scheduled"},
                {"days_offset": 3, "opponent": "Philadelphia Phillies", "opponent_abbr": "PHI", "opponent_id": 143, "is_home": False, "status": "Scheduled"},
                {"days_offset": 4, "opponent": "Philadelphia Phillies", "opponent_abbr": "PHI", "opponent_id": 143, "is_home": False, "status": "Scheduled"},
                {"days_offset": 6, "opponent": "Atlanta Braves", "opponent_abbr": "ATL", "opponent_id": 144, "is_home": True, "status": "Scheduled"},
                {"days_offset": 8, "opponent": "Miami Marlins", "opponent_abbr": "MIA", "opponent_id": 146, "is_home": True, "status": "Scheduled"},
                {"days_offset": 10, "opponent": "New York Yankees", "opponent_abbr": "NYY", "opponent_id": 147, "is_home": False, "status": "Scheduled"},
            ]
            
            # Generate games for the next 60 days to cover multiple months
            # Repeat the pattern every 14 days to create a realistic schedule
            for week_offset in range(0, 5):  # 5 weeks = ~35 days, extend to 60
                for entry in mock_blueprint:
                    days_offset = entry["days_offset"] + (week_offset * 14)
                    
                    game_dt = base_first_pitch + timedelta(days=days_offset)
                    game_date = game_dt.date()
                    
                    # Only include if within date range
                    if start_dt <= game_date <= end_dt:
                        games.append({
                            "game_date": game_date.isoformat(),
                            "game_datetime": game_dt.isoformat(),
                            "opponent_name": entry["opponent"],
                            "opponent_abbr": entry["opponent_abbr"],
                            "opponent_id": entry["opponent_id"],
                            "is_home": entry["is_home"],
                            "status": entry.get("status", "Scheduled"),
                            "venue": "TBD",
                        })
        elif next_games:
            days_ahead = max((end_dt - today).days, 0)
            
            if days_ahead == 0:
                days_ahead = 365  # If end date is in past, get a full year
            
            # Get all games in range
            games = next_games(team_abbr, days_ahead=min(days_ahead, 365), include_started=True)
        else:
            return []
        
        # Filter by date range if provided
        filtered_games = []
        for game in games:
            game_date_str = game.get("game_date")
            if game_date_str:
                try:
                    if isinstance(game_date_str, str):
                        game_date = datetime.fromisoformat(game_date_str).date()
                    else:
                        game_date = game_date_str if isinstance(game_date_str, date) else datetime.combine(game_date_str, datetime.min.time()).date()
                    
                    if start_dt <= game_date <= end_dt:
                        filtered_games.append(game)
                except Exception:
                    continue
        
        return filtered_games
    except Exception as e:
        print(f"Warning loading full season schedule: {e}")
        import traceback
        traceback.print_exc()
        return []


def _load_player_deliverables(user: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], int]:
    if not PlayerDB or not user:
        return _sample_deliverables()

    rows: List[Dict[str, Any]] = []
    db = None
    try:
        db = PlayerDB()
        cursor = db.conn.cursor()
        cursor.execute(
            """
            SELECT d.*,
                   u.first_name AS uploader_first_name,
                   u.last_name AS uploader_last_name
            FROM player_documents AS d
            LEFT JOIN users AS u ON u.id = d.uploaded_by
            WHERE d.player_id = ?
            ORDER BY d.uploaded_at DESC
            LIMIT 6
            """,
            (int(user.get("id")),),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    except Exception as exc:
        print(f"Warning loading player deliverables for user {user.get('id')}: {exc}")
    finally:
        if db:
            db.close()

    if not rows:
        return _sample_deliverables()

    now_local = datetime.now(timezone.utc).astimezone()
    deliverables: List[Dict[str, Any]] = []

    for raw_doc in rows:
        raw_doc["category"] = (raw_doc.get("category") or "").strip().lower() or None
        uploader_name = "Sequence Staff"
        first = (raw_doc.get("uploader_first_name") or "").strip()
        last = (raw_doc.get("uploader_last_name") or "").strip()
        if first or last:
            uploader_name = f"{first} {last}".strip()

        formatted = _format_player_document(raw_doc)
        deliverable = _map_document_to_deliverable(raw_doc, formatted, uploader_name, now_local)
        deliverables.append(deliverable)

    deliverables.sort(key=lambda item: item.get("uploaded_ts") or 0, reverse=True)

    outstanding_count = sum(1 for item in deliverables if item.get("requires_ack"))
    latest = deliverables[0] if deliverables else None
    latest_document = None
    if latest:
        latest_document = {
            "title": latest["title"],
            "owner": latest["owner"],
            "time_ago": latest["time_ago"],
            "link": latest["link"],
        }

    for item in deliverables:
        item.pop("uploaded_ts", None)

    return latest_document, deliverables, outstanding_count


def _map_document_to_deliverable(
    raw_doc: Dict[str, Any],
    formatted: Dict[str, Any],
    uploader_name: str,
    reference_now: datetime,
) -> Dict[str, Any]:
    uploaded_ts = raw_doc.get("uploaded_at") or 0
    uploaded_dt = datetime.fromtimestamp(uploaded_ts, tz=timezone.utc).astimezone(reference_now.tzinfo)

    category = formatted.get("category")
    category_icon = {
        WORKOUT_CATEGORY: "fas fa-dumbbell",
        "scouting": "fas fa-clipboard-user",
        "report": "fas fa-chart-line",
        "video": "fas fa-film",
    }.get(category, "fas fa-file-lines")

    series_label = formatted.get("series_label")
    summary = series_label or (formatted.get("series_range_display") or "Shared document")

    title = _friendly_title(formatted.get("filename") or raw_doc.get("filename"))
    link = formatted.get("viewer_url") or formatted.get("download_url") or url_for("reports_library")
    requires_ack = bool(
        category != WORKOUT_CATEGORY
        and (reference_now - uploaded_dt).total_seconds() < 5 * 24 * 3600
    )
    if formatted.get("series_status") in {"current", "upcoming"}:
        requires_ack = True

    return {
        "id": raw_doc.get("id"),
        "icon": category_icon,
        "title": title,
        "summary": summary,
        "owner": uploader_name or "Sequence Staff",
        "time_ago": _humanize_time_ago(uploaded_dt, reference_now),
        "link": link,
        "requires_ack": requires_ack,
        "uploaded_ts": uploaded_ts,
    }


def _friendly_title(filename: Optional[str]) -> str:
    if not filename:
        return "Document"
    stem = os.path.splitext(filename)[0]
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned if cleaned else filename


def _humanize_time_ago(target: Optional[datetime], reference: Optional[datetime] = None) -> str:
    if not target:
        return "just now"
    if reference is None:
        reference = datetime.now(timezone.utc).astimezone()

    if target.tzinfo is None:
        target = target.replace(tzinfo=reference.tzinfo or timezone.utc)
    delta = reference - target
    seconds = delta.total_seconds()

    if seconds < 0:
        seconds = abs(seconds)
        prefix = "in "
    else:
        prefix = ""

    if seconds < 60:
        phrase = "moments"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        phrase = f"{minutes} min"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        phrase = f"{hours} hr"
    elif seconds < 7 * 86400:
        days = int(seconds // 86400)
        phrase = f"{days} day"
    else:
        return target.strftime("%b %d")

    if prefix:
        return f"{prefix}{phrase}"
    return f"{phrase} ago"


def _sample_deliverables():
    now_local = datetime.now(timezone.utc).astimezone()
    deliverables = [
        {
            "id": "sample-capsule",
            "icon": "fas fa-clipboard-user",
            "title": "Dodgers Series Capsule",
            "summary": "Matchup tendencies & red zone plan.",
            "owner": "Pro Scouting",
            "time_ago": "2 hr ago",
            "link": url_for("reports_library"),
            "requires_ack": True,
            "uploaded_ts": now_local.timestamp() - 2 * 3600,
        },
        {
            "id": "sample-workout",
            "icon": "fas fa-dumbbell",
            "title": "Wednesday Activation",
            "summary": "Mobility + lower half sequencing.",
            "owner": "Performance Team",
            "time_ago": "Yesterday",
            "link": url_for("workouts"),
            "requires_ack": False,
            "uploaded_ts": now_local.timestamp() - 27 * 3600,
        },
        {
            "id": "sample-video",
            "icon": "fas fa-film",
            "title": "LHP Slider Review",
            "summary": "High-leverage pitch shapes & cues.",
            "owner": "Video Coord.",
            "time_ago": "3 days ago",
            "link": url_for("visuals"),
            "requires_ack": True,
            "uploaded_ts": now_local.timestamp() - 3 * 86400,
        },
    ]
    outstanding = sum(1 for item in deliverables if item["requires_ack"])
    latest = deliverables[0]
    latest_document = {
        "title": latest["title"],
        "owner": latest["owner"],
        "time_ago": latest["time_ago"],
        "link": latest["link"],
    }
    for item in deliverables:
        item.pop("uploaded_ts", None)
    return latest_document, deliverables, outstanding


def _build_focus_highlights(
    next_series: Optional[Dict[str, Any]],
    latest_document: Optional[Dict[str, Any]],
    outstanding_count: int,
) -> List[Dict[str, Any]]:
    highlights: List[Dict[str, Any]] = []

    if next_series:
        start = next_series.get("start_date")
        days_until = next_series.get("days_until")
        detail_parts: List[str] = []
        if start:
            detail_parts.append(f"Starts {start}")
        if days_until is not None:
            detail_parts.append(f"{days_until} day{'s' if days_until != 1 else ''} out")
        highlights.append(
            {
                "icon": "fas fa-baseball",
                "label": "Next Series",
                "value": next_series.get("opponent_name") or "Opponent TBD",
                "detail": " • ".join(detail_parts) if detail_parts else None,
                "cta": {
                    "label": "Matchup Capsule",
                    "href": url_for("gameday"),
                },
            }
        )

    if latest_document:
        highlights.append(
            {
                "icon": "fas fa-file-lines",
                "label": "Latest Upload",
                "value": latest_document.get("title", "New document"),
                "detail": f"{latest_document.get('owner', 'Staff')} • {latest_document.get('time_ago', 'just now')}",
                "cta": {
                    "label": "Open File",
                    "href": latest_document.get("link") or url_for("reports_library"),
                },
            }
        )

    if outstanding_count:
        highlights.append(
            {
                "icon": "fas fa-bell",
                "label": "Action Needed",
                "value": f"{outstanding_count} item(s) awaiting acknowledgement",
                "detail": None,
                "cta": {
                    "label": "Review",
                    "href": url_for("reports_library"),
                },
            }
        )

    if not highlights:
        highlights.append(
            {
                "icon": "fas fa-check-circle",
                "label": "All Clear",
                "value": "You’re caught up on updates and action items.",
                "detail": None,
                "cta": None,
            }
        )

    return highlights


def _build_performance_snapshot() -> Dict[str, Any]:
    offense_values = [0.318, 0.327, 0.334, 0.329, 0.338, 0.345]
    training_values = [78, 80, 82, 84, 87, 90]

    offense_delta = offense_values[-1] - offense_values[-2]
    training_delta = training_values[-1] - training_values[-2]

    return {
        "offense_metric": {
            "value": f"{offense_values[-1]:.3f} xwOBA",
            "delta_label": f"{offense_delta:+.3f} vs last 7",
            "sparkline": _build_sparkline_svg(offense_values, "#f97316"),
        },
        "training_metric": {
            "value": f"{training_values[-1]:.0f}% readiness",
            "delta_label": f"{training_delta:+.0f}% vs last week",
            "sparkline": _build_sparkline_svg(training_values, "#22d3ee"),
        },
    }


def _build_sparkline_svg(values: List[float], stroke: str) -> str:
    if not values:
        return ""

    width = max(len(values) - 1, 1) * 20
    height = 48
    min_val = min(values)
    max_val = max(values)
    spread = max(max_val - min_val, 0.0001)

    coords = []
    for idx, value in enumerate(values):
        x = (width / (len(values) - 1)) * idx if len(values) > 1 else width / 2
        y = height - ((value - min_val) / spread) * height
        coords.append(f"{x:.2f},{y:.2f}")

    points = " ".join(coords)
    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">'
        f'<polyline fill="none" stroke="{stroke}" stroke-width="3" points="{points}" stroke-linecap="round"/>'
        "</svg>"
    )


def _load_journal_preview(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not PlayerDB or not user:
        return _sample_journal_entries()

    entries: List[Dict[str, Any]] = []
    db = None
    try:
        db = PlayerDB()
        rows = db.list_journal_entries(user_id=user.get("id"), limit=3)
    except Exception as exc:
        print(f"Warning fetching journal entries: {exc}")
        rows = []
    finally:
        if db:
            db.close()

    if not rows:
        return _sample_journal_entries()

    for row in rows:
        entry_date = row.get("entry_date") or ""
        try:
            parsed_date = datetime.strptime(entry_date, "%Y-%m-%d")
            date_label = parsed_date.strftime("%b %d")
        except Exception:
            date_label = entry_date or "Recent"

        body = (row.get("body") or "").strip()
        preview = textwrap.shorten(body, width=180, placeholder="…") if body else "Notes captured."
        title = (row.get("title") or "Training reflections").strip()

        entries.append({
            "title": title,
            "date": date_label,
            "preview": preview,
            "reply": None,
        })

    return entries


def _sample_journal_entries() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Game 3 vs LAD — at-bat notes",
            "date": "Jun 10",
            "preview": "Saw their setup man three times — slider start up, finish below barrel. Staying through the middle with shorter gather helped.",
            "reply": {
                "coach_name": "Ramirez",
                "text": "Keep that gather—add fastball machine reps tomorrow with same cue.",
            },
        },
        {
            "title": "Bullpen touch & feel",
            "date": "Jun 09",
            "preview": "Good carry when I stayed stacked. Focus for next pen: tempo after leg lift and staying tall through release.",
            "reply": None,
        },
    ]


def _load_resource_links(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        team_abbr = _determine_user_team(user)
        notes = _collect_staff_notes(team_abbr=team_abbr, limit=4)
    except Exception as exc:
        print(f"Warning loading resources: {exc}")
        notes = []

    if notes:
        resources = []
        for note in notes:
            tags = note.get("tags") or []
            category = " • ".join(tags) if tags else "Staff note"
            resources.append({
                "title": note.get("title") or "Staff update",
                "category": category,
                "link": url_for("gameday") + f"#staff-note-{note.get('id')}",
            })
        return resources

    return _sample_resources()


def _sample_resources() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Recovery: 24-hour travel reset",
            "category": "Recovery",
            "link": url_for("nutrition"),
        },
        {
            "title": "Mobility primer — lower half",
            "category": "Movement",
            "link": url_for("workouts"),
        },
        {
            "title": "Approach dashboard — LHP game plan",
            "category": "Video",
            "link": url_for("visuals"),
        },
    ]


def _load_player_news(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load 2 player-specific and 2 league-wide news articles."""
    first_name = (user.get("first_name") or "").strip()
    last_name = (user.get("last_name") or "").strip()
    
    if not first_name or not last_name:
        return []
    
    player_name = f"{first_name} {last_name}"
    player_news_items = []
    league_news_items = []
    
    # Helper function to check if image URL is good (not Google proxy, etc.)
    def is_good_image_url(url):
        if not url:
            return False
        # Skip Google proxy images
        if 'googleusercontent.com' in url or 'google.com' in url:
            return False
        # Skip data URIs (too small usually)
        if url.startswith('data:'):
            return False
        # Skip very small images
        if any(skip in url.lower() for skip in ['icon', 'logo', 'avatar', 'thumb', '16x16', '32x32']):
            # But allow if it's clearly an article image
            if any(allow in url.lower() for allow in ['article', 'news', 'story', 'feature', 'hero', 'main']):
                return True
            return False
        return True
    
    try:
        import feedparser
        from urllib.parse import quote
        from bs4 import BeautifulSoup
        
        # First, try Google News - fetch with requests first, then parse
        try:
            search_queries = [
                f"{player_name}",
                f"{last_name} baseball",
                f"{first_name} {last_name} MLB",
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            for query in search_queries:
                if len(player_news_items) >= 2:
                    break
                try:
                    # Fetch RSS feed with requests first
                    google_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en&gl=US&ceid=US:en"
                    response = requests.get(google_url, headers=headers, timeout=10, allow_redirects=True)
                    
                    if response.status_code == 200:
                        # Parse the XML content
                        feed = feedparser.parse(response.content)
                        
                        if feed.entries and len(feed.entries) > 0:
                            for entry in feed.entries[:20]:
                                if len(player_news_items) >= 2:
                                    break
                                
                                title = entry.get('title', '')
                                # Clean Google News title (remove " - Source" suffix)
                                if ' - ' in title:
                                    title_parts = title.split(' - ')
                                    title = ' - '.join(title_parts[:-1])  # Remove last part (source)
                                
                                summary = entry.get('summary', entry.get('description', ''))
                                link = entry.get('link', '#')
                                published = entry.get('published', entry.get('updated', ''))
                                
                                # More lenient matching
                                search_text = (title + " " + summary).lower()
                                player_lower = player_name.lower()
                                last_lower = last_name.lower()
                                
                                # Match if player name or last name appears
                                if (player_lower in search_text or 
                                    last_lower in search_text or
                                    (first_name.lower() in search_text and last_lower in search_text)):
                                    
                                    # Skip duplicates
                                    if any(existing.get('title', '').lower().startswith(title.lower()[:50]) for existing in player_news_items):
                                        continue
                                    
                                    # Determine category
                                    category = "MLB News"
                                    if any(word in search_text for word in ['video', 'highlight', 'watch', 'replay']):
                                        category = "Video Analysis"
                                    elif any(word in search_text for word in ['stat', 'performance', 'batting', 'hitting', 'home run', 'homer']):
                                        category = "Performance"
                                    elif any(word in search_text for word in ['injury', 'health', 'disabled list', 'dl']):
                                        category = "Health"
                                    
                                    # Try to fetch image from article page first (best quality)
                                    image_url = None
                                    if link and link != '#':
                                        try:
                                            article_response = requests.get(link, headers=headers, timeout=5, allow_redirects=True)
                                            if article_response.status_code == 200:
                                                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                                
                                                # Try og:image first (usually best quality)
                                                og_image = article_soup.find('meta', property='og:image')
                                                if og_image and og_image.get('content'):
                                                    candidate = og_image.get('content')
                                                    if is_good_image_url(candidate):
                                                        image_url = candidate
                                                
                                                # Try twitter:image as backup
                                                if not image_url:
                                                    twitter_image = article_soup.find('meta', attrs={'name': 'twitter:image'})
                                                    if twitter_image and twitter_image.get('content'):
                                                        candidate = twitter_image.get('content')
                                                        if is_good_image_url(candidate):
                                                            image_url = candidate
                                                
                                                # Try to find article hero/main image
                                                if not image_url:
                                                    # Look for images with specific classes/attributes that indicate article images
                                                    for img in article_soup.find_all('img', src=True):
                                                        img_src = img.get('src', '')
                                                        if not img_src:
                                                            continue
                                                        
                                                        # Handle relative URLs
                                                        if not img_src.startswith('http'):
                                                            if img_src.startswith('//'):
                                                                img_src = 'https:' + img_src
                                                            else:
                                                                try:
                                                                    from urllib.parse import urljoin
                                                                    img_src = urljoin(link, img_src)
                                                                except:
                                                                    continue
                                                        
                                                        if not is_good_image_url(img_src):
                                                            continue
                                                        
                                                        # Check for article image indicators
                                                        img_class = img.get('class', [])
                                                        img_id = img.get('id', '')
                                                        parent_class = ''
                                                        if img.parent:
                                                            parent_class = ' '.join(img.parent.get('class', []))
                                                        
                                                        search_text = ' '.join([
                                                            ' '.join(img_class) if isinstance(img_class, list) else str(img_class),
                                                            img_id,
                                                            parent_class,
                                                            img_src.lower()
                                                        ]).lower()
                                                        
                                                        # Prefer images that look like article images
                                                        if any(keyword in search_text for keyword in ['article', 'news', 'story', 'feature', 'hero', 'main', 'headline', 'lead', 'cover']):
                                                            image_url = img_src
                                                            break
                                                    
                                                    # If still no image, take first decent-sized image
                                                    if not image_url:
                                                        for img in article_soup.find_all('img', src=True):
                                                            img_src = img.get('src', '')
                                                            if not img_src:
                                                                continue
                                                            
                                                            # Handle relative URLs
                                                            if not img_src.startswith('http'):
                                                                if img_src.startswith('//'):
                                                                    img_src = 'https:' + img_src
                                                                else:
                                                                    try:
                                                                        from urllib.parse import urljoin
                                                                        img_src = urljoin(link, img_src)
                                                                    except:
                                                                        continue
                                                            
                                                            if is_good_image_url(img_src):
                                                                # Check image dimensions if available
                                                                width = img.get('width')
                                                                height = img.get('height')
                                                                if width and height:
                                                                    try:
                                                                        w, h = int(width), int(height)
                                                                        if w >= 200 and h >= 150:  # Reasonable size
                                                                            image_url = img_src
                                                                            break
                                                                    except:
                                                                        pass
                                                                else:
                                                                    # No dimensions, but URL looks good
                                                                    image_url = img_src
                                                                    break
                                        except Exception as e:
                                            pass  # Don't fail if we can't fetch the article
                                    
                                    # Fallback to RSS feed images only if we didn't get a good one from article
                                    if not image_url:
                                        # Try media_content
                                        if entry.get('media_content'):
                                            media = entry.get('media_content', [])
                                            if isinstance(media, list) and len(media) > 0:
                                                media_item = media[0]
                                                candidate = None
                                                if isinstance(media_item, dict):
                                                    candidate = media_item.get('url')
                                                elif isinstance(media_item, str):
                                                    candidate = media_item
                                                if candidate and is_good_image_url(candidate):
                                                    image_url = candidate
                                        
                                        # Try media_thumbnail
                                        if not image_url and entry.get('media_thumbnail'):
                                            thumb = entry.get('media_thumbnail', [])
                                            if isinstance(thumb, list) and len(thumb) > 0:
                                                thumb_item = thumb[0]
                                                candidate = None
                                                if isinstance(thumb_item, dict):
                                                    candidate = thumb_item.get('url')
                                                elif isinstance(thumb_item, str):
                                                    candidate = thumb_item
                                                if candidate and is_good_image_url(candidate):
                                                    image_url = candidate
                                        
                                        # Try to extract from summary/description HTML
                                        if not image_url:
                                            summary_html = entry.get('summary', entry.get('description', ''))
                                            if summary_html and '<img' in summary_html:
                                                try:
                                                    from bs4 import BeautifulSoup
                                                    soup = BeautifulSoup(summary_html, 'html.parser')
                                                    for img_tag in soup.find_all('img'):
                                                        candidate = img_tag.get('src')
                                                        if candidate:
                                                            # Handle relative URLs
                                                            if not candidate.startswith('http'):
                                                                if candidate.startswith('//'):
                                                                    candidate = 'https:' + candidate
                                                                elif candidate.startswith('/'):
                                                                    try:
                                                                        from urllib.parse import urlparse
                                                                        parsed = urlparse(link)
                                                                        candidate = f"{parsed.scheme}://{parsed.netloc}{candidate}"
                                                                    except:
                                                                        continue
                                                            
                                                            if is_good_image_url(candidate):
                                                                image_url = candidate
                                                                break
                                                except:
                                                    pass
                                    
                                    # If we only have a Google proxy image, set to None to use fallback icon
                                    if image_url and not is_good_image_url(image_url):
                                        image_url = None
                                    
                                    player_news_items.append({
                                        "category": category,
                                        "title": title[:100] + "..." if len(title) > 100 else title,
                                        "description": (summary[:150] + "..." if len(summary) > 150 else summary) or "Read more...",
                                        "time_ago": _format_news_time(published),
                                        "link": link,
                                        "icon": _get_news_icon(category),
                                        "image": image_url,
                                        "type": "player"
                                    })
                except Exception as e:
                    print(f"Warning: Could not fetch Google News for query '{query}': {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not fetch Google News: {e}")
        
        # Now fetch league-wide news (2 articles)
        try:
            # Fetch general MLB news
            league_queries = [
                "MLB news",
                "Major League Baseball",
                "MLB latest",
            ]
            
            for query in league_queries:
                if len(league_news_items) >= 2:
                    break
                try:
                    google_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en&gl=US&ceid=US:en"
                    response = requests.get(google_url, headers=headers, timeout=10, allow_redirects=True)
                    
                    if response.status_code == 200:
                        feed = feedparser.parse(response.content)
                        
                        if feed.entries and len(feed.entries) > 0:
                            for entry in feed.entries[:20]:
                                if len(league_news_items) >= 2:
                                    break
                                
                                title = entry.get('title', '')
                                # Clean Google News title
                                if ' - ' in title:
                                    title_parts = title.split(' - ')
                                    title = ' - '.join(title_parts[:-1])
                                
                                summary = entry.get('summary', entry.get('description', ''))
                                link = entry.get('link', '#')
                                published = entry.get('published', entry.get('updated', ''))
                                
                                # Skip if it's about the player (we want league-wide only)
                                search_text = (title + " " + summary).lower()
                                player_lower = player_name.lower()
                                last_lower = last_name.lower()
                                
                                if (player_lower in search_text or 
                                    last_lower in search_text or
                                    (first_name.lower() in search_text and last_lower in search_text)):
                                    continue  # Skip player-specific news
                                
                                # Skip duplicates
                                if any(existing.get('title', '').lower().startswith(title.lower()[:50]) for existing in league_news_items):
                                    continue
                                
                                # Determine category
                                category = "MLB News"
                                if any(word in search_text for word in ['video', 'highlight', 'watch', 'replay']):
                                    category = "Video Analysis"
                                elif any(word in search_text for word in ['stat', 'performance', 'batting', 'hitting']):
                                    category = "Performance"
                                
                                # Try to fetch image from article page
                                image_url = None
                                if link and link != '#':
                                    try:
                                        article_response = requests.get(link, headers=headers, timeout=5, allow_redirects=True)
                                        if article_response.status_code == 200:
                                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                            og_image = article_soup.find('meta', property='og:image')
                                            if og_image and og_image.get('content'):
                                                candidate = og_image.get('content')
                                                if is_good_image_url(candidate):
                                                    image_url = candidate
                                    except:
                                        pass
                                
                                # If we only have a Google proxy image, set to None
                                if image_url and not is_good_image_url(image_url):
                                    image_url = None
                                
                                league_news_items.append({
                                    "category": category,
                                    "title": title[:100] + "..." if len(title) > 100 else title,
                                    "description": (summary[:150] + "..." if len(summary) > 150 else summary) or "Read more...",
                                    "time_ago": _format_news_time(published),
                                    "link": link,
                                    "icon": _get_news_icon(category),
                                    "image": image_url,
                                    "type": "league"
                                })
                except Exception as e:
                    print(f"Warning: Could not fetch league news for query '{query}': {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not fetch league news: {e}")
        
        # Try multiple RSS feeds as backup for player news
        if len(player_news_items) < 2:
            rss_urls = [
                "https://www.espn.com/espn/rss/mlb/news",
                "https://feeds.feedburner.com/mlb/rss",
            ]
            
            # Use same headers for consistency
            rss_headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            for rss_url in rss_urls:
                if len(player_news_items) >= 2:
                    break
                try:
                    feed = feedparser.parse(rss_url)
                    if feed.entries and len(feed.entries) > 0:
                        for entry in feed.entries[:30]:
                            if len(player_news_items) >= 2:
                                break
                            
                            title = entry.get('title', '')
                            summary = entry.get('summary', entry.get('description', ''))
                            link = entry.get('link', '#')
                            published = entry.get('published', entry.get('updated', ''))
                            
                            search_text = (title + " " + summary).lower()
                            player_lower = player_name.lower()
                            last_lower = last_name.lower()
                            
                            if (player_lower in search_text or 
                                last_lower in search_text or
                                (first_name.lower() in search_text and last_lower in search_text)):
                                
                                # Skip duplicates
                                if any(existing.get('title', '').lower().startswith(title.lower()[:50]) for existing in player_news_items):
                                    continue
                                
                                category = "MLB News"
                                if any(word in search_text for word in ['video', 'highlight', 'watch', 'replay']):
                                    category = "Video Analysis"
                                elif any(word in search_text for word in ['stat', 'performance', 'batting', 'hitting', 'home run', 'homer']):
                                    category = "Performance"
                                
                                # Try to extract image
                                image_url = None
                                # Try to fetch from article page first
                                if link and link != '#':
                                    try:
                                        article_response = requests.get(link, headers=rss_headers, timeout=5, allow_redirects=True)
                                        if article_response.status_code == 200:
                                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                            og_image = article_soup.find('meta', property='og:image')
                                            if og_image and og_image.get('content'):
                                                candidate = og_image.get('content')
                                                if is_good_image_url(candidate):
                                                    image_url = candidate
                                    except:
                                        pass
                                
                                # Fallback to RSS feed images
                                if not image_url:
                                    if entry.get('media_content'):
                                        media = entry.get('media_content', [])
                                        if isinstance(media, list) and len(media) > 0:
                                            candidate = media[0].get('url') if isinstance(media[0], dict) else None
                                            if candidate and is_good_image_url(candidate):
                                                image_url = candidate
                                    if not image_url and entry.get('media_thumbnail'):
                                        thumb = entry.get('media_thumbnail', [])
                                        if isinstance(thumb, list) and len(thumb) > 0:
                                            candidate = thumb[0].get('url') if isinstance(thumb[0], dict) else None
                                            if candidate and is_good_image_url(candidate):
                                                image_url = candidate
                                
                                # If we only have a bad image, set to None to use fallback icon
                                if image_url and not is_good_image_url(image_url):
                                    image_url = None
                                
                                player_news_items.append({
                                    "category": category,
                                    "title": title[:100] + "..." if len(title) > 100 else title,
                                    "description": (summary[:150] + "..." if len(summary) > 150 else summary) or "Read more...",
                                    "time_ago": _format_news_time(published),
                                    "link": link,
                                    "icon": _get_news_icon(category),
                                    "image": image_url,
                                    "type": "player"
                                })
                except Exception as e:
                    print(f"Warning: Could not parse RSS feed {rss_url}: {e}")
                    continue
        
        # If we didn't get enough player news, try web scraping MLB.com search
        if len(player_news_items) < 2:
            try:
                # Try MLB.com search with better headers
                search_url = f"https://www.mlb.com/search?q={quote(player_name)}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
                response = requests.get(search_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Look for various article patterns
                    articles = soup.find_all(['article', 'div', 'a'], class_=lambda x: x and isinstance(x, str) and ('article' in x.lower() or 'news' in x.lower() or 'story' in x.lower()))[:15]
                    
                    for article in articles:
                        if len(player_news_items) >= 2:
                            break
                        
                        # Try to find link and title
                        link_elem = article.find('a', href=True) if article.name != 'a' else article
                        if not link_elem or not link_elem.get('href'):
                            continue
                            
                        title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'span'], class_=lambda x: x and isinstance(x, str) and ('title' in x.lower() or 'headline' in x.lower()))
                        if not title_elem:
                            title_elem = article.find(['h1', 'h2', 'h3', 'h4'])
                        
                        if link_elem and title_elem:
                            title = title_elem.get_text(strip=True)
                            link = link_elem.get('href', '#')
                            if not link.startswith('http'):
                                link = f"https://www.mlb.com{link}" if link.startswith('/') else f"https://www.mlb.com/{link}"
                            
                            # Get description
                            desc_elem = article.find('p') or article.find('div', class_=lambda x: x and isinstance(x, str) and 'description' in x.lower())
                            description = desc_elem.get_text(strip=True) if desc_elem else ""
                            
                            # More flexible matching
                            title_lower = title.lower()
                            if (player_name.lower() in title_lower or 
                                (first_name.lower() in title_lower and last_name.lower() in title_lower) or
                                (last_name.lower() in title_lower and len(last_name) > 3)):
                                
                                # Try to extract image from article
                                image_url = None
                                # Try to fetch from article page
                                if link and link != '#':
                                    try:
                                        article_response = requests.get(link, headers=headers, timeout=5, allow_redirects=True)
                                        if article_response.status_code == 200:
                                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                            og_image = article_soup.find('meta', property='og:image')
                                            if og_image and og_image.get('content'):
                                                candidate = og_image.get('content')
                                                if is_good_image_url(candidate):
                                                    image_url = candidate
                                    except:
                                        pass
                                
                                # Fallback to scraping search results
                                if not image_url:
                                    try:
                                        img_elem = article.find('img')
                                        if img_elem and img_elem.get('src'):
                                            candidate = img_elem.get('src')
                                            if not candidate.startswith('http'):
                                                candidate = f"https://www.mlb.com{candidate}" if candidate.startswith('/') else f"https://www.mlb.com/{candidate}"
                                            if is_good_image_url(candidate):
                                                image_url = candidate
                                    except:
                                        pass
                                
                                # If we only have a bad image, set to None to use fallback icon
                                if image_url and not is_good_image_url(image_url):
                                    image_url = None
                                
                                player_news_items.append({
                                    "category": "MLB News",
                                    "title": title[:100] + "..." if len(title) > 100 else title,
                                    "description": description[:150] + "..." if len(description) > 150 else (description or "Read more..."),
                                    "time_ago": "Recently",
                                    "link": link,
                                    "icon": "fas fa-newspaper",
                                    "image": image_url,
                                    "type": "player"
                                })
            except Exception as e:
                print(f"Warning: Could not scrape MLB.com: {e}")
        
        # Try NewsAPI if available for league news (requires API key)
        if len(league_news_items) < 2:
            newsapi_key = os.environ.get('NEWSAPI_KEY')
            if newsapi_key:
                try:
                    newsapi_url = f"https://newsapi.org/v2/everything?q={quote('MLB baseball')}&apiKey={newsapi_key}&sortBy=publishedAt&language=en&pageSize=10"
                    response = requests.get(newsapi_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])
                        for article in articles[:10]:
                            if len(league_news_items) >= 2:
                                break
                            
                            # Skip if it's about the player
                            article_text = (article.get('title', '') + " " + article.get('description', '')).lower()
                            if (player_name.lower() in article_text or 
                                last_name.lower() in article_text):
                                continue
                            
                            # Get image and filter if needed
                            image_url = article.get('urlToImage') or article.get('image')
                            if image_url and not is_good_image_url(image_url):
                                image_url = None
                            
                            league_news_items.append({
                                "category": "MLB News",
                                "title": article.get('title', '')[:100] + "..." if len(article.get('title', '')) > 100 else article.get('title', ''),
                                "description": article.get('description', '')[:150] + "..." if len(article.get('description', '')) > 150 else (article.get('description', '') or "Read more..."),
                                "time_ago": _format_news_time(article.get('publishedAt', '')),
                                "link": article.get('url', '#'),
                                "icon": "fas fa-newspaper",
                                "image": image_url,
                                "type": "league"
                            })
                except Exception as e:
                    print(f"Warning: Could not fetch from NewsAPI: {e}")
        
    except ImportError as e:
        print(f"Warning: Missing required library for news fetching: {e}")
    except Exception as e:
        print(f"Warning: Error loading player news for {player_name}: {e}")
        import traceback
        traceback.print_exc()
    
    # Combine player and league news: 2 player-specific, 2 league-wide
    combined_news = player_news_items[:2] + league_news_items[:2]
    return combined_news if combined_news else []


def _format_news_time(date_str: Optional[str]) -> str:
    """Format news date to human-readable time ago."""
    if not date_str:
        return "Recently"
    
    try:
        # Try parsing various date formats
        news_date = None
        # Try dateutil first
        try:
            from dateutil import parser
            news_date = parser.parse(date_str)
        except (ImportError, Exception):
            # Fallback to datetime.strptime for common formats
            date_formats = [
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y",
            ]
            for fmt in date_formats:
                try:
                    news_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
        
        if not news_date:
            return "Recently"
        
        now = datetime.now(timezone.utc)
        if news_date.tzinfo is None:
            news_date = news_date.replace(tzinfo=timezone.utc)
        
        delta = now - news_date
        days = delta.days
        hours = delta.seconds // 3600
        
        if days == 0:
            if hours == 0:
                return "Just now"
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif days == 1:
            return "Yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        else:
            months = days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
    except Exception:
        return "Recently"


def _get_news_icon(category: str) -> str:
    """Get icon class based on news category."""
    category_lower = category.lower()
    if "video" in category_lower or "highlight" in category_lower:
        return "fas fa-video"
    elif "injury" in category_lower or "health" in category_lower:
        return "fas fa-heartbeat"
    elif "trade" in category_lower or "transaction" in category_lower:
        return "fas fa-exchange-alt"
    elif "performance" in category_lower or "stats" in category_lower:
        return "fas fa-trophy"
    elif "interview" in category_lower:
        return "fas fa-microphone"
    else:
        return "fas fa-newspaper"


def _sample_player_news(player_name: str) -> List[Dict[str, Any]]:
    """Return sample news items when real news is unavailable."""
    return [
        {
            "category": "MLB News",
            "title": f"{player_name} Continues Strong Season",
            "description": f"Latest updates and highlights from {player_name}'s recent performances",
            "time_ago": "Today",
            "link": "#",
            "icon": "fas fa-newspaper"
        },
        {
            "category": "Video Analysis",
            "title": "Recent Game Highlights",
            "description": f"Watch {player_name}'s standout moments from recent games",
            "time_ago": "2 days ago",
            "link": "#",
            "icon": "fas fa-video"
        },
        {
            "category": "Performance",
            "title": "Season Statistics Update",
            "description": f"Review {player_name}'s latest stats and performance metrics",
            "time_ago": "1 week ago",
            "link": "#",
            "icon": "fas fa-trophy"
        },
        {
            "category": "Team News",
            "title": "Team Updates",
            "description": f"Latest news and updates from {player_name}'s team",
            "time_ago": "1 week ago",
            "link": "#",
            "icon": "fas fa-users"
        }
    ]


def _load_support_contacts() -> List[Dict[str, Any]]:
    return _sample_support_contacts()


def _sample_support_contacts() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Jordan Lee",
            "role": "Hitting Coach",
            "contact_label": "Message",
            "contact_link": "mailto:jlee@sequencebiolab.com",
            "photo": None,
        },
        {
            "name": "Morgan Patel",
            "role": "Performance Lead",
            "contact_label": "Call",
            "contact_link": "tel:+15555551212",
            "photo": None,
        },
        {
            "name": "Avery Chen",
            "role": "Nutrition",
            "contact_label": "Check-in",
            "contact_link": url_for("nutrition"),
            "photo": None,
        },
    ]


@app.route('/')
def home():
    """Landing/home page"""
    viewer_user = getattr(g, "user", None)
    target_user = viewer_user
    admin_user_options: List[Dict[str, Any]] = []
    requested_user_id = request.args.get("user_id", type=int)

    def _format_user_label(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "Unknown User"
        first = (record.get("first_name") or "").strip()
        last = (record.get("last_name") or "").strip()
        full_name = f"{first} {last}".strip()
        if full_name:
            return full_name
        email = (record.get("email") or "").strip()
        if email:
            return email
        return f"User #{record.get('id')}"

    if session.get("is_admin") and PlayerDB:
        user_rows: List[Dict[str, Any]] = []
        db = None
        try:
            db = PlayerDB()
            user_rows = db.list_users()
        except Exception as exc:
            print(f"Warning fetching users for admin home selector: {exc}")
        finally:
            try:
                if db:
                    db.close()
            except Exception:
                pass

        admin_user_options = [
            {"id": row["id"], "label": _format_user_label(row)}
            for row in user_rows
        ]

        if requested_user_id:
            for row in user_rows:
                if row["id"] == requested_user_id:
                    target_user = row
                    break

    context = _build_player_home_context(target_user)
    context["admin_user_options"] = admin_user_options
    context["selected_user_id"] = (target_user.get("id") if target_user else None)
    context["selected_user"] = target_user  # Pass selected user for display
    return render_template('home.html', **context)


@app.route('/schedule')
@login_required
def schedule():
    """Full schedule page with month-by-month navigation."""
    viewer_user = getattr(g, "user", None)
    if not viewer_user:
        return redirect(url_for('login'))
    
    target_user = viewer_user
    admin_user_options: List[Dict[str, Any]] = []
    requested_user_id = request.args.get("user_id", type=int)
    
    def _format_user_label(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "Unknown User"
        first = (record.get("first_name") or "").strip()
        last = (record.get("last_name") or "").strip()
        full_name = f"{first} {last}".strip()
        if full_name:
            return full_name
        email = (record.get("email") or "").strip()
        if email:
            return email
        return f"User #{record.get('id')}"
    
    if session.get("is_admin") and PlayerDB:
        user_rows: List[Dict[str, Any]] = []
        db = None
        try:
            db = PlayerDB()
            user_rows = db.list_users()
        except Exception as exc:
            print(f"Warning fetching users for admin schedule selector: {exc}")
        finally:
            try:
                if db:
                    db.close()
            except Exception:
                pass
        
        admin_user_options = [
            {"id": row["id"], "label": _format_user_label(row)}
            for row in user_rows
        ]
        
        if requested_user_id:
            for row in user_rows:
                if row["id"] == requested_user_id:
                    target_user = row
                    break
    
    # Get month/year from query params, default to current month
    month = request.args.get('month', type=int)
    year = request.args.get('year', type=int)
    
    if not month or not year:
        now = datetime.now()
        month = now.month
        year = now.year
    
    # Calculate date range for the month
    start_date = datetime(year, month, 1).date()
    if month == 12:
        end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
    
    # Load games for this month using target_user (selected user for admin)
    games = _load_full_season_schedule(
        target_user, 
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )
    
    # Group games by date and format dates
    games_by_date = {}
    for game in games:
        game_date = game.get("game_date")
        if game_date:
            try:
                dt = datetime.fromisoformat(game_date)
                date_key = dt.strftime("%Y-%m-%d")
                if date_key not in games_by_date:
                    games_by_date[date_key] = {
                        "date_obj": dt,
                        "day_name": dt.strftime("%A"),
                        "date_formatted": dt.strftime("%B %d, %Y"),
                        "games": []
                    }
                
                # Format game time if available
                formatted_game = dict(game)
                if game.get("game_datetime"):
                    try:
                        game_dt = datetime.fromisoformat(game["game_datetime"].replace('Z', '+00:00'))
                        formatted_game["time_formatted"] = game_dt.strftime("%I:%M %p")
                    except Exception:
                        formatted_game["time_formatted"] = "TBD"
                else:
                    formatted_game["time_formatted"] = "TBD"
                
                games_by_date[date_key]["games"].append(formatted_game)
            except Exception:
                continue
    
    # Calculate previous/next month
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year
    
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year
    
    # Get all months with games for quick navigation
    all_games = _load_full_season_schedule(target_user)
    months_with_games = set()
    for game in all_games:
        game_date = game.get("game_date")
        if game_date:
            try:
                dt = datetime.fromisoformat(game_date)
                months_with_games.add((dt.year, dt.month))
            except Exception:
                continue
    
    # Add current month if no games found (so user can still navigate)
    months_with_games.add((year, month))
    
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    return render_template(
        'schedule.html',
        games_by_date=games_by_date,
        current_month=month,
        current_year=year,
        month_name=month_names[month - 1],
        prev_month=prev_month,
        prev_year=prev_year,
        next_month=next_month,
        next_year=next_year,
        months_with_games=months_with_games,
        month_names=month_names,
        admin_user_options=admin_user_options,
        selected_user_id=(target_user.get("id") if target_user else None)
    )

@app.route('/scouting-report')
def scouting_report():
    """Scouting report generator page"""
    return render_template('scouting_report.html')

@app.route('/pitchers-report')
def pitchers_report():
    """Pitcher's reports page"""
    return render_template('pitchers_report.html')

@app.route('/mocap')
@login_required
def mocap():
    """Mocap analysis page"""
    # Check if user is admin
    is_admin = session.get('is_admin', False)
    
    # Build the motion webapp URL
    # Admins use a different base URL and mode
    if is_admin:
        base_url = "https://cooper-710.github.io/motion-webapp"
        mode = "admin"
    else:
        base_url = "https://motion-webapp.pages.dev"
        mode = "player"
    
    # Check if user info is in session (for future authentication)
    player_name = request.args.get('player')
    if not player_name:
        user = getattr(g, "user", None)
        if user and user.get('first_name') and user.get('last_name'):
            player_name = f"{user['first_name']} {user['last_name']}"
        elif session.get('first_name') and session.get('last_name'):
            player_name = f"{session['first_name']} {session['last_name']}"
        else:
            # Fallback example player (should be rare with login_required)
            player_name = "Pete Alonso"
    
    # URL encode the player name (spaces become +)
    encoded_player_name = quote_plus(player_name)
    
    # Get session date from query params or use default
    session_date = request.args.get('session', '2025-08-27')
    
    # Build the full URL with parameters
    if is_admin:
        motion_app_url = f"{base_url}/?mode={mode}&player={encoded_player_name}&session={session_date}"
    else:
        motion_app_url = f"{base_url}/?mode={mode}&player={encoded_player_name}&session={session_date}&lock=1"
    
    return render_template('mocap.html', motion_app_url=motion_app_url)

@app.route('/pitchviz')
def pitchviz():
    """PitchViz visualization page"""
    # Base URL for the PitchViz webapp
    base_url = "https://cooper-710.github.io/NEWPV-main_with_orbit/"
    
    # Get parameters from query string or use defaults
    team = request.args.get('team', 'ARI')
    pitcher = request.args.get('pitcher', 'Backhus, Kyle')
    view = request.args.get('view', 'catcher')
    trail = request.args.get('trail', '0')
    orbit = request.args.get('orbit', '1')
    
    # URL encode the pitcher name (handles commas and spaces)
    encoded_pitcher = quote_plus(pitcher)
    
    # Build the full URL with parameters
    pitchviz_url = f"{base_url}?team={team}&pitcher={encoded_pitcher}&view={view}&trail={trail}&orbit={orbit}"
    
    return render_template('pitchviz.html', pitchviz_url=pitchviz_url)

@app.route('/contractviz')
def contractviz():
    """ContractViz analysis page"""
    # Base URL for the ContractViz webapp
    base_url = "https://contract-viz-boras.vercel.app/"
    
    # Build the URL (can add query parameters here if needed in the future)
    contractviz_url = base_url
    
    return render_template('contractviz.html', contractviz_url=contractviz_url)

@app.route('/player-database')
def player_database():
    """Player database page"""
    return render_template('player_database.html')

@app.route('/player/<player_id>')
def player_profile(player_id):
    """Player profile page"""
    return render_template('player_profile.html', player_id=player_id)

# Player Database API Routes
@app.route('/api/players', methods=['GET'])
def api_players():
    """List/search players with filters"""
    if not PlayerDB:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        db = PlayerDB()
        search = request.args.get('search', '').strip() or None
        team = request.args.get('team', '').strip() or None
        position = request.args.get('position', '').strip() or None
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 25))
        offset = (page - 1) * limit
        
        players = db.search_players(search=search, team=team, position=position, limit=limit, offset=offset)
        total = db.count_players(search=search, team=team, position=position)
        
        db.close()
        
        return jsonify({
            "players": players,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/players/<player_id>', methods=['GET'])
def api_player_detail(player_id):
    """Get detailed player profile"""
    if not PlayerDB:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        db = PlayerDB()
        player = db.get_player(player_id)
        
        if not player:
            db.close()
            return jsonify({"error": "Player not found"}), 404
        
        # Get current season stats (default to 2024)
        season = request.args.get('season', '2024')
        current_season = db.get_player_current_season(player_id, season)
        
        db.close()
        
        return jsonify({
            "player": player,
            "current_season": current_season
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/players/<player_id>/stats', methods=['GET'])
def api_player_stats(player_id):
    """Get player stats history"""
    if not PlayerDB:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        db = PlayerDB()
        seasons = db.get_player_seasons(player_id)
        db.close()
        
        return jsonify({
            "player_id": player_id,
            "seasons": seasons
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/players/<player_id>/seasons', methods=['GET'])
def api_player_seasons(player_id):
    """Get season-by-season stats (alias for stats)"""
    return api_player_stats(player_id)

@app.route('/api/teams', methods=['GET'])
def api_teams():
    """Get all teams for filter dropdown"""
    if not PlayerDB:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        db = PlayerDB()
        teams = db.get_all_teams()
        db.close()
        
        return jsonify({"teams": teams})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# CSV-based Player Search API Routes
@app.route('/api/csv/search', methods=['GET'])
def api_csv_search():
    """Search for players by name in CSV files"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        search_term = request.args.get('q', '').strip()
        if not search_term:
            return jsonify({"players": []})
        
        players = csv_loader.search_players(search_term)
        return jsonify({"players": players})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/csv/player/<player_name>/seasons', methods=['GET'])
def api_csv_player_seasons(player_name):
    """Get available seasons for a specific player"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        # Decode URL-encoded player name
        from urllib.parse import unquote
        player_name = unquote(player_name)
        
        # Try direct player data lookup first (more reliable)
        player_data = csv_loader.get_player_data(player_name)
        if player_data and player_data.get('fangraphs'):
            # Extract unique seasons from fangraphs data
            seasons = set()
            for row in player_data['fangraphs']:
                if 'Season' in row and row['Season'] is not None:
                    try:
                        seasons.add(int(row['Season']))
                    except (ValueError, TypeError):
                        pass
            
            if seasons:
                seasons_str = sorted([str(s) for s in seasons], reverse=True)
                return jsonify({
                    "player": player_name,
                    "seasons": seasons_str
                })
        
        # Fallback: Get all players summary to find this player
        players = csv_loader.get_all_players_summary()
        player_name_lower = player_name.lower().strip()
        
        for player in players:
            if player.get('name', '').lower().strip() == player_name_lower:
                seasons = player.get('seasons', [])
                # Ensure seasons are strings for the dropdown
                seasons_str = [str(s) for s in seasons] if seasons else []
                return jsonify({
                    "player": player.get('name'),
                    "seasons": seasons_str
                })
        
        return jsonify({"error": f"Player '{player_name}' not found"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/csv/player/<player_name>', methods=['GET'])
def api_csv_player_data(player_name):
    """Get all data for a specific player from CSV files"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        # Decode URL-encoded player name
        from urllib.parse import unquote
        player_name = unquote(player_name)
        
        player_data = csv_loader.get_player_data(player_name)
        return jsonify(player_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/players/sync', methods=['POST'])
def api_sync_players():
    """Trigger database sync from Sportradar (background job)"""
    # This would trigger a background sync job
    # For now, return a message indicating it needs to be run manually
    return jsonify({
        "message": "Sync initiated. This is a long-running operation. Run src/populate_players.py manually.",
        "status": "queued"
    }), 202

@app.route('/visuals')
def visuals():
    """Visuals page"""
    return render_template('visuals.html')

@app.route('/heatmaps')
def heatmaps():
    """Heatmaps visualization page"""
    return render_template('heatmaps.html')

@app.route('/spraychart')
def spraychart():
    """Spray chart visualization page"""
    return render_template('spraychart.html')

@app.route('/timeline')
def timeline():
    """Performance Timeline visualization page"""
    return render_template('timeline.html')

@app.route('/pitchplots')
def pitchplots():
    """Pitch Plots visualization page"""
    return render_template('pitchplots.html')

@app.route('/velocity_trends')
def velocity_trends():
    """Velocity Trends visualization page"""
    return render_template('velocity_trends.html')

@app.route('/pitch-mix-analysis')
def pitch_mix_analysis():
    """Pitch Mix Analysis visualization page"""
    return render_template('pitch_mix_analysis.html')

@app.route('/count-performance')
def count_performance():
    """Count Performance Breakdown visualization page"""
    return render_template('count_performance.html')

@app.route('/zone-contact-rates')
def zone_contact_rates():
    """Zone Contact Rates visualization page"""
    return render_template('zone_contact_rates.html')

@app.route('/plate-discipline-matrix')
def plate_discipline_matrix():
    """Plate Discipline Matrix visualization page"""
    return render_template('plate_discipline_matrix.html')

@app.route('/expected-stats-comparison')
def expected_stats_comparison():
    """Expected Stats Comparison visualization page"""
    return render_template('expected_stats_comparison.html')

@app.route('/pitch-tunnel')
def pitch_tunnel():
    """Pitch Tunnel Analysis visualization page"""
    return render_template('pitch_tunnel.html')

@app.route('/barrel-quality-contact')
def barrel_quality_contact():
    """Barrel Rate & Quality of Contact visualization page"""
    return render_template('barrel_quality_contact.html')

@app.route('/swing-decision-matrix')
def swing_decision_matrix():
    """Swing Decision Matrix visualization page"""
    return render_template('swing_decision_matrix.html')

@app.route('/pitch-arsenal-effectiveness')
def pitch_arsenal_effectiveness():
    """Pitch Arsenal Effectiveness visualization page"""
    return render_template('pitch_arsenal_effectiveness.html')

# Analytics API Endpoints
@app.route('/api/analytics/players', methods=['GET'])
def api_analytics_players():
    """Get all players with basic stats for filtering"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        players = csv_loader.get_all_players_summary()
        
        # Get unique teams and positions for filters
        teams = sorted(list(set(p['team'] for p in players if p['team'])))
        positions = sorted(list(set(p['position'] for p in players if p['position'])))
        all_seasons = sorted(list(set(season for p in players for season in p['seasons'])))
        
        return jsonify({
            "players": players,
            "filters": {
                "teams": teams,
                "positions": positions,
                "seasons": all_seasons
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/trends', methods=['GET'])
def api_analytics_trends():
    """Get player performance trends over seasons"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        player_name = request.args.get('player', '').strip()
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        stats = request.args.getlist('stats')
        if not stats:
            stats = None
        
        season_start = request.args.get('season_start', type=int)
        season_end = request.args.get('season_end', type=int)
        
        trends = csv_loader.get_player_trends(
            player_name=player_name,
            stats=stats,
            season_start=season_start,
            season_end=season_end
        )
        
        return jsonify(trends)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/comparisons', methods=['GET'])
def api_analytics_comparisons():
    """Compare multiple players across selected metrics"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        players = request.args.getlist('players')
        if not players:
            return jsonify({"error": "At least one player is required"}), 400
        
        stats = request.args.getlist('stats')
        if not stats:
            return jsonify({"error": "At least one stat is required"}), 400
        
        season = request.args.get('season', type=int)
        
        comparison = csv_loader.compare_players(
            player_names=players,
            stats=stats,
            season=season
        )
        
        return jsonify(comparison)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/leaders', methods=['GET'])
def api_analytics_leaders():
    """Get league leaders for various stats"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        stat = request.args.get('stat', '').strip()
        if not stat:
            return jsonify({"error": "Stat is required"}), 400
        
        limit = request.args.get('limit', default=10, type=int)
        season = request.args.get('season', type=int)
        position = request.args.get('position', '').strip() or None
        team = request.args.get('team', '').strip() or None
        
        leaders = csv_loader.get_league_leaders(
            stat=stat,
            limit=limit,
            season=season,
            position=position,
            team=team
        )
        
        return jsonify({"leaders": leaders, "stat": stat})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/distributions', methods=['GET'])
def api_analytics_distributions():
    """Get statistical distributions for metrics"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        stat = request.args.get('stat', '').strip()
        if not stat:
            return jsonify({"error": "Stat is required"}), 400
        
        season = request.args.get('season', type=int)
        position = request.args.get('position', '').strip() or None
        team = request.args.get('team', '').strip() or None
        bins = request.args.get('bins', default=20, type=int)
        
        distribution = csv_loader.get_stat_distribution(
            stat=stat,
            season=season,
            position=position,
            team=team,
            bins=bins
        )
        
        return jsonify(distribution)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/heatmap', methods=['GET'])
def api_visuals_heatmap():
    """Get heatmap data for visualization - returns location-based heatmap data"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        player_name = request.args.get('player', '').strip()
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        metric = request.args.get('metric', '').strip()
        if not metric:
            return jsonify({"error": "Metric is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        team = request.args.get('team', '').strip() or None
        position = request.args.get('position', '').strip() or None
        count = request.args.get('count', '').strip() or None
        pitcher_hand = request.args.get('pitcher_hand', '').strip() or None
        pitch_type = request.args.get('pitch_type', '').strip() or None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, fetch_pitcher_statcast, lookup_batter_id
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np
        
        # Get player data filtered by criteria
        try:
            players = csv_loader.get_all_players_summary()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error loading player data: {str(e)}"}), 500
        
        if not players:
            return jsonify({"error": "No players found in database"}), 404
        
        # Find the selected player (case-insensitive match with whitespace handling)
        selected_player_data = None
        player_name_normalized = player_name.strip().lower()
        for player in players:
            player_name_from_data = player.get('name', '').strip().lower()
            if player_name_from_data == player_name_normalized:
                selected_player_data = player
                break
        
        # If exact match not found, try partial match
        if not selected_player_data:
            for player in players:
                player_name_from_data = player.get('name', '').strip().lower()
                if player_name_normalized in player_name_from_data or player_name_from_data in player_name_normalized:
                    selected_player_data = player
                    break
        
        if not selected_player_data:
            # Return more helpful error with available players for debugging
            similar_names = [p['name'] for p in players if player_name_normalized[:3] in p['name'].lower()][:5]
            error_msg = f"Player '{player_name}' not found"
            if similar_names:
                error_msg += f". Similar names: {', '.join(similar_names)}"
            return jsonify({"error": error_msg}), 404
        
        # Use the actual player name from the database (to handle case differences)
        actual_player_name = selected_player_data.get('name')
        
        # Determine if player is a pitcher or hitter
        player_position = selected_player_data.get('position', '').upper()
        is_pitcher = 'P' in player_position or 'PITCHER' in player_position
        
        # Additional filters (if they match player's data)
        # Note: Team and position filters apply to the player's overall data
        if team and selected_player_data.get('team') != team:
            return jsonify({
                "error": f"Player '{actual_player_name}' does not match team filter '{team}'",
                "metric": metric,
                "data": [],
                "summary": {}
            }), 200
        
        if position and selected_player_data.get('position') != position:
            return jsonify({
                "error": f"Player '{actual_player_name}' does not match position filter '{position}'",
                "metric": metric,
                "data": [],
                "summary": {}
            }), 200
        
        # Get player ID (works for both batters and pitchers)
        try:
            player_id = lookup_batter_id(actual_player_name)
        except Exception as e:
            return jsonify({"error": f"Could not find player ID for '{actual_player_name}': {str(e)}"}), 404
        
        # Calculate date range for the season
        # If season is specified, use it; otherwise fetch all available data
        if season:
            start_date = f"{season}-03-01"  # Start of season
            end_date = f"{season}-11-30"    # End of season
            filter_by_season = True
        else:
            # Fetch data from a wide range to get all available seasons
            # Statcast data goes back to 2008, but we'll use a reasonable range
            current_year = datetime.now().year
            start_date = "2008-03-01"  # Statcast started in 2008
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data based on player type
        # Try to determine player type by attempting to fetch both types of data
        statcast_df = None
        is_pitcher_confirmed = False
        
        if is_pitcher:
            # Try pitcher data first
            try:
                statcast_df = fetch_pitcher_statcast(player_id, start_date, end_date)
                if statcast_df is not None and not statcast_df.empty:
                    is_pitcher_confirmed = True
            except Exception:
                pass
        
        # If not confirmed as pitcher or no pitcher data, try batter data
        if statcast_df is None or statcast_df.empty:
            try:
                statcast_df = fetch_batter_statcast(player_id, start_date, end_date)
                if statcast_df is not None and not statcast_df.empty:
                    is_pitcher_confirmed = False
            except Exception:
                pass
        
        # If still no data, try the other type
        if (statcast_df is None or statcast_df.empty) and not is_pitcher:
            try:
                statcast_df = fetch_pitcher_statcast(player_id, start_date, end_date)
                if statcast_df is not None and not statcast_df.empty:
                    is_pitcher_confirmed = True
            except Exception:
                pass
        
        if statcast_df is None or statcast_df.empty:
            return jsonify({
                "error": f"No statcast data found for {actual_player_name}",
                "metric": metric,
                "data": [],
                "grid": []
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            # First, try to filter by year column if it exists (most reliable)
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in statcast_df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                # Filter by year (this is the most reliable method)
                # Convert to int if needed for comparison
                if not pd.api.types.is_integer_dtype(statcast_df[year_column]):
                    statcast_df[year_column] = pd.to_numeric(statcast_df[year_column], errors='coerce')
                statcast_df = statcast_df[
                    (statcast_df[year_column].notna()) & 
                    (statcast_df[year_column] == int(season))
                ]
            else:
                # Fallback: try filtering by date column
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in statcast_df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    # Convert date column to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(statcast_df[date_column]):
                        statcast_df[date_column] = pd.to_datetime(statcast_df[date_column], errors='coerce')
                    
                    # Filter to the specific season
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    # Drop rows where date conversion failed
                    statcast_df = statcast_df[statcast_df[date_column].notna()]
                    
                    # Filter by date range
                    statcast_df = statcast_df[
                        (statcast_df[date_column] >= season_start) & 
                        (statcast_df[date_column] <= season_end)
                    ]
            
            # Check if we have data after season filtering
            if statcast_df.empty:
                return jsonify({
                    "error": f"No statcast data found for {actual_player_name} for season {season}",
                    "metric": metric,
                    "data": [],
                    "grid": []
                }), 200
        
        # Filter by count if specified
        if count:
            count_parts = count.split('-')
            if len(count_parts) == 2:
                try:
                    balls_filter = int(count_parts[0])
                    strikes_filter = int(count_parts[1])
                    if 'balls' in statcast_df.columns and 'strikes' in statcast_df.columns:
                        statcast_df = statcast_df[
                            (statcast_df['balls'] == balls_filter) & 
                            (statcast_df['strikes'] == strikes_filter)
                        ]
                except ValueError:
                    pass
        
        # Filter by pitcher/batter handedness if specified
        if is_pitcher_confirmed:
            # For pitchers, filter by batter handedness (stand column)
            if pitcher_hand and 'stand' in statcast_df.columns:
                statcast_df = statcast_df[statcast_df['stand'] == pitcher_hand]
        else:
            # For hitters, filter by pitcher handedness (p_throws column)
            if pitcher_hand and 'p_throws' in statcast_df.columns:
                statcast_df = statcast_df[statcast_df['p_throws'] == pitcher_hand]
        
        # Filter by pitch type if specified
        if pitch_type and 'pitch_type' in statcast_df.columns:
            statcast_df = statcast_df[statcast_df['pitch_type'] == pitch_type]
        
        # Filter out rows without location data
        if 'plate_x' not in statcast_df.columns or 'plate_z' not in statcast_df.columns:
            return jsonify({
                "error": "Location data (plate_x, plate_z) not available in statcast data",
                "metric": metric,
                "data": [],
                "grid": []
            }), 200
        
        statcast_df = statcast_df[
            statcast_df['plate_x'].notna() & 
            statcast_df['plate_z'].notna()
        ]
        
        if statcast_df.empty:
            return jsonify({
                "error": "No valid location data found",
                "metric": metric,
                "data": [],
                "grid": []
            }), 200
        
        # Helper function to calculate SLG from events
        def calculate_slg(cell_data):
            """Calculate slugging percentage from events column"""
            if 'events' not in cell_data.columns:
                return None
            
            # Filter to only at-bats - need events that are not NaN
            # At-bats exclude walks, hit-by-pitch, etc.
            ab_data = cell_data[
                cell_data['events'].notna() & 
                cell_data['events'].isin([
                    'single', 'double', 'triple', 'home_run', 'field_out', 
                    'strikeout', 'force_out', 'grounded_into_double_play',
                    'fielders_choice', 'field_error', 'double_play', 'triple_play',
                    'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'catcher_interf'
                ])
            ]
            
            if len(ab_data) == 0:
                return None
            
            # Calculate total bases
            bases_map = {
                'single': 1,
                'double': 2,
                'triple': 3,
                'home_run': 4
            }
            
            total_bases = 0
            for event in ab_data['events']:
                total_bases += bases_map.get(event, 0)
            
            return total_bases / len(ab_data)
        
        # Helper function to calculate OBP from events
        def calculate_obp(cell_data):
            """Calculate on-base percentage from events column"""
            if 'events' not in cell_data.columns:
                return None
            
            # Plate appearances = all events that are not NaN (actual outcomes)
            # Exclude stolen bases and caught stealing
            pa_data = cell_data[
                cell_data['events'].notna() & 
                ~cell_data['events'].isin([
                    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home',
                    'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home'
                ])
            ]
            
            if len(pa_data) == 0:
                return None
            
            # On-base events: hits, walks, hit-by-pitch
            on_base_events = ['single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch']
            on_base_count = pa_data['events'].isin(on_base_events).sum()
            
            return on_base_count / len(pa_data)
        
        # Helper function to calculate HR rate from events
        def calculate_hr_rate(cell_data):
            """Calculate home run rate (HR per plate appearance)"""
            if 'events' not in cell_data.columns:
                return None
            
            # Count plate appearances (events that are not NaN)
            pa_data = cell_data[
                cell_data['events'].notna() & 
                ~cell_data['events'].isin([
                    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home',
                    'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home'
                ])
            ]
            
            if len(pa_data) == 0:
                return None
            
            hr_count = (pa_data['events'] == 'home_run').sum()
            return hr_count / len(pa_data)
        
        # Helper function to calculate RBI rate from events (approximate)
        def calculate_rbi_rate(cell_data):
            """Calculate approximate RBI rate - home runs always have RBI, others may vary"""
            if 'events' not in cell_data.columns:
                return None
            
            # Count plate appearances (events that are not NaN)
            pa_data = cell_data[
                cell_data['events'].notna() & 
                ~cell_data['events'].isin([
                    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home',
                    'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home'
                ])
            ]
            
            if len(pa_data) == 0:
                return None
            
            # Home runs always have at least 1 RBI (often more)
            # For other hits, we'll estimate based on hit type
            rbi_estimate = 0
            for event in pa_data['events']:
                if event == 'home_run':
                    rbi_estimate += 1.5  # Average ~1.5 RBI per HR
                elif event == 'triple':
                    rbi_estimate += 0.8  # High probability of scoring runner
                elif event == 'double':
                    rbi_estimate += 0.6
                elif event == 'single':
                    rbi_estimate += 0.4
            
            return rbi_estimate / len(pa_data)
        
        # Map metric to calculation method
        metric_upper = metric.upper()
        calculate_metric = None
        metric_column = None
        
        # Pitcher-specific metrics
        if is_pitcher_confirmed:
            if metric_upper in ['WHIFF_RATE', 'WHIFF RATE', 'WHIFF', 'WHIFF%']:
                def calculate_whiff_rate(cell_data):
                    """Calculate whiff rate (swinging strikes / swings)"""
                    if 'description' not in cell_data.columns:
                        return None
                    desc = cell_data['description'].astype(str).str.lower()
                    swings = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play']).sum()
                    whiffs = desc.isin(['swinging_strike', 'swinging_strike_blocked']).sum()
                    return (whiffs / swings) if swings > 0 else None
                calculate_metric = calculate_whiff_rate
            elif metric_upper in ['STRIKE_RATE', 'STRIKE RATE', 'STRIKE', 'STRIKE%']:
                def calculate_strike_rate(cell_data):
                    """Calculate strike rate (strikes / total pitches)"""
                    total = len(cell_data)
                    if total == 0:
                        return None
                    # Check for type column (S = strike, X = in play, B = ball)
                    if 'type' in cell_data.columns:
                        strikes = cell_data['type'].isin(['S', 'X']).sum()
                    elif 'description' in cell_data.columns:
                        # Fallback: count non-ball descriptions as strikes
                        desc = cell_data['description'].astype(str).str.lower()
                        strikes = (~desc.isin(['ball', 'blocked_ball', 'intent_ball'])).sum()
                    else:
                        return None
                    return strikes / total
                calculate_metric = calculate_strike_rate
            elif metric_upper in ['XWOBA', 'XWOBA_ALLOWED']:
                metric_column = 'estimated_woba_using_speedangle'
            elif metric_upper in ['XBA', 'XBA_ALLOWED']:
                metric_column = 'estimated_ba_using_speedangle'
            elif metric_upper in ['XSLG', 'XSLG_ALLOWED']:
                metric_column = 'estimated_slg_using_speedangle'
            else:
                # Default to xwOBA for pitchers
                metric_column = 'estimated_woba_using_speedangle'
        else:
            # Hitter-specific metrics
            if metric_upper in ['SLG', 'SLUGGING']:
                calculate_metric = calculate_slg
            elif metric_upper in ['OBP', 'ON_BASE_PERCENTAGE']:
                calculate_metric = calculate_obp
            elif metric_upper in ['OPS', 'ON_BASE_PLUS_SLUGGING']:
                # OPS = OBP + SLG
                def calculate_ops(cell_data):
                    obp = calculate_obp(cell_data)
                    slg = calculate_slg(cell_data)
                    if obp is None or slg is None:
                        return None
                    return obp + slg
                calculate_metric = calculate_ops
            elif metric_upper in ['HR', 'HOME_RUNS', 'HOME_RUN']:
                calculate_metric = calculate_hr_rate
            elif metric_upper in ['RBI', 'RUNS_BATTED_IN']:
                calculate_metric = calculate_rbi_rate
            elif metric_upper in ['WRC+', 'WRC', 'WRC_PLUS']:
                # wRC+ is complex, but we can use wOBA as a proxy since it's closely related
                # For a proper wRC+ calculation, we'd need league averages, park factors, etc.
                # We'll use woba_value if available, otherwise estimated_woba
                if 'woba_value' in statcast_df.columns:
                    metric_column = 'woba_value'
                else:
                    metric_column = 'estimated_woba_using_speedangle'
            elif metric_upper in ['WAR']:
                # WAR cannot be calculated at the pitch level - it's a cumulative stat
                return jsonify({
                    "error": "WAR is a cumulative statistic and cannot be visualized as a location-based heatmap. Please select a different metric.",
                    "metric": metric,
                    "data": [],
                    "grid": []
                }), 200
            else:
                # Direct column mappings for hitters
                metric_map = {
                    'XWOBA': 'estimated_woba_using_speedangle',
                    'XBA': 'estimated_ba_using_speedangle',
                    'XSLG': 'estimated_slg_using_speedangle',
                    'WOBA': 'woba_value',
                    'BA': 'hit',
                    'AVG': 'hit',
                    'AVERAGE': 'hit',
                }
                
                metric_column = metric_map.get(metric_upper, 'estimated_woba_using_speedangle')
        
        # If using direct column, verify it exists
        if calculate_metric is None:
            if metric_column not in statcast_df.columns:
                # Try alternative columns
                for alt_col in ['estimated_woba_using_speedangle', 'woba_value', 'launch_speed', 'launch_angle']:
                    if alt_col in statcast_df.columns:
                        metric_column = alt_col
                        break
                else:
                    return jsonify({
                        "error": f"Metric '{metric}' cannot be calculated from available statcast data",
                        "metric": metric,
                        "data": [],
                        "grid": []
                    }), 200
        
        # Create grid for strike zone (10x10 grid)
        # Strike zone: x from -1.5 to 1.5 feet, z from 1.5 to 3.5 feet
        grid_size = 12
        x_min, x_max = -2.0, 2.0
        z_min, z_max = 0.5, 4.5
        
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        z_bins = np.linspace(z_min, z_max, grid_size + 1)
        
        # Bin the data
        statcast_df['x_bin'] = pd.cut(statcast_df['plate_x'], bins=x_bins, labels=False)
        statcast_df['z_bin'] = pd.cut(statcast_df['plate_z'], bins=z_bins, labels=False)
        
        # Calculate average metric for each grid cell
        grid_data = []
        for x_idx in range(grid_size):
            for z_idx in range(grid_size):
                cell_data = statcast_df[
                    (statcast_df['x_bin'] == x_idx) & 
                    (statcast_df['z_bin'] == z_idx)
                ]
                
                pitch_count = len(cell_data)
                
                if not cell_data.empty:
                    # Use calculation function if available, otherwise use direct column
                    if calculate_metric is not None:
                        # Calculate metric from events/outcomes
                        avg_value = calculate_metric(cell_data)
                        if avg_value is not None:
                            avg_value = float(avg_value)
                        else:
                            avg_value = None
                    elif metric_column in cell_data.columns:
                        # Calculate average from direct column
                        values = cell_data[metric_column].dropna()
                        if len(values) > 0:
                            avg_value = float(values.mean())
                        else:
                            avg_value = None
                    else:
                        avg_value = None
                else:
                    avg_value = None
                
                # Calculate center coordinates of the cell
                x_center = (x_bins[x_idx] + x_bins[x_idx + 1]) / 2
                z_center = (z_bins[z_idx] + z_bins[z_idx + 1]) / 2
                
                grid_data.append({
                    'x': x_idx,
                    'y': grid_size - 1 - z_idx,  # Flip y-axis for display
                    'x_center': float(x_center),
                    'z_center': float(z_center),
                    'value': avg_value,
                    'count': pitch_count
                })
        
        # Filter out cells with no data
        grid_data = [cell for cell in grid_data if cell['value'] is not None and cell['count'] > 0]
        
        if not grid_data:
            return jsonify({
                "error": f"No {metric} data available for the selected filters",
                "metric": metric,
                "data": [],
                "grid": []
            }), 200
        
        # Calculate summary statistics
        values = [cell['value'] for cell in grid_data if cell['value'] is not None]
        
        # Determine batter handedness from statcast data
        batter_hand = 'R'  # Default to right-handed
        if 'stand' in statcast_df.columns:
            stand_values = statcast_df['stand'].dropna()
            if len(stand_values) > 0:
                # Use the most common value
                batter_hand = str(stand_values.mode().iloc[0]) if len(stand_values.mode()) > 0 else 'R'
        
        heatmap_data = {
            'player': actual_player_name,
            'metric': metric,
            'batter_hand': batter_hand,  # 'R' or 'L'
            'filters': {
                'season': season,
                'team': team,
                'position': position,
                'count': count,
                'pitcher_hand': pitcher_hand,
                'pitch_type': pitch_type
            },
            'grid': grid_data,
            'grid_size': grid_size,
            'x_range': [float(x_min), float(x_max)],
            'z_range': [float(z_min), float(z_max)],
            'summary': {
                'total_cells': len(grid_data),
                'min_value': float(min(values)) if values else 0,
                'max_value': float(max(values)) if values else 0,
                'avg_value': float(np.mean(values)) if values else 0,
                'total_pitches': sum(cell['count'] for cell in grid_data)
            }
        }
        
        return jsonify(heatmap_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/heatmap/player-info', methods=['GET'])
def api_visuals_heatmap_player_info():
    """Get player type and available metrics"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        player_name = request.args.get('player', '').strip()
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        players = csv_loader.get_all_players_summary()
        player_name_normalized = player_name.strip().lower()
        
        selected_player_data = None
        for player in players:
            if player.get('name', '').strip().lower() == player_name_normalized:
                selected_player_data = player
                break
        
        if not selected_player_data:
            return jsonify({"error": "Player not found"}), 404
        
        player_position = selected_player_data.get('position', '').upper()
        is_pitcher = 'P' in player_position or 'PITCHER' in player_position
        
        # Define sorted metrics
        hitter_metrics = [
            {'value': 'xwOBA', 'label': 'xwOBA'},
            {'value': 'xSLG', 'label': 'xSLG'},
            {'value': 'xBA', 'label': 'xBA'},
            {'value': 'wRC+', 'label': 'wRC+'},
            {'value': 'OPS', 'label': 'OPS'},
            {'value': 'SLG', 'label': 'SLG'},
            {'value': 'OBP', 'label': 'OBP'},
            {'value': 'AVG', 'label': 'Batting Average'},
            {'value': 'HR', 'label': 'Home Runs'},
            {'value': 'RBI', 'label': 'RBI'},
        ]
        
        pitcher_metrics = [
            {'value': 'xwOBA', 'label': 'xwOBA Allowed'},
            {'value': 'xBA', 'label': 'xBA Allowed'},
            {'value': 'xSLG', 'label': 'xSLG Allowed'},
            {'value': 'Whiff Rate', 'label': 'Whiff Rate'},
            {'value': 'Strike Rate', 'label': 'Strike Rate'},
        ]
        
        return jsonify({
            'is_pitcher': is_pitcher,
            'metrics': pitcher_metrics if is_pitcher else hitter_metrics
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/spraychart', methods=['GET'])
def api_visuals_spraychart():
    """Get spray chart data for visualization - returns batted ball locations"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        player_name = request.args.get('player', '').strip()
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Filter options
        event_type = request.args.get('event_type', '').strip() or None  # e.g., 'single', 'double', 'home_run'
        min_launch_speed = request.args.get('min_launch_speed', type=float)
        max_launch_speed = request.args.get('max_launch_speed', type=float)
        min_launch_angle = request.args.get('min_launch_angle', type=float)
        max_launch_angle = request.args.get('max_launch_angle', type=float)
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, fetch_pitcher_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get player data
        try:
            players = csv_loader.get_all_players_summary()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error loading player data: {str(e)}"}), 500
        
        if not players:
            return jsonify({"error": "No players found in database"}), 404
        
        # Find the selected player
        selected_player_data = None
        player_name_normalized = player_name.strip().lower()
        for player in players:
            player_name_from_data = player.get('name', '').strip().lower()
            if player_name_from_data == player_name_normalized:
                selected_player_data = player
                break
        
        if not selected_player_data:
            for player in players:
                player_name_from_data = player.get('name', '').strip().lower()
                if player_name_normalized in player_name_from_data or player_name_from_data in player_name_normalized:
                    selected_player_data = player
                    break
        
        if not selected_player_data:
            similar_names = [p['name'] for p in players if player_name_normalized[:3] in p['name'].lower()][:5]
            error_msg = f"Player '{player_name}' not found"
            if similar_names:
                error_msg += f". Similar names: {', '.join(similar_names)}"
            return jsonify({"error": error_msg}), 404
        
        actual_player_name = selected_player_data.get('name')
        
        # Get player ID (works for both batters and pitchers)
        try:
            player_id = lookup_batter_id(actual_player_name)
        except Exception as e:
            return jsonify({"error": f"Could not find player ID for '{actual_player_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Try to fetch statcast data - try pitcher first (for spray chart, we want batted balls hit OFF of pitchers)
        # Then try batter if pitcher fails (for batters, we want batted balls they hit)
        statcast_df = None
        player_type = None
        
        # First, try as pitcher (batted balls hit off of them)
        try:
            pitcher_df = fetch_pitcher_statcast(player_id, start_date, end_date)
            if pitcher_df is not None and not pitcher_df.empty:
                # Check if there are any batted balls in the pitcher data
                if 'events' in pitcher_df.columns:
                    batted_balls = pitcher_df[pitcher_df['events'].notna()]
                    if not batted_balls.empty:
                        statcast_df = pitcher_df
                        player_type = 'pitcher'
        except Exception as e:
            pass  # Try batter below
        
        # If pitcher lookup failed or returned no batted ball data, try as batter
        if statcast_df is None or (statcast_df is not None and statcast_df.empty):
            try:
                batter_df = fetch_batter_statcast(player_id, start_date, end_date)
                if batter_df is not None and not batter_df.empty:
                    # Check if there are any batted balls in the batter data
                    if 'events' in batter_df.columns:
                        batted_balls = batter_df[batter_df['events'].notna()]
                        if not batted_balls.empty:
                            statcast_df = batter_df
                            player_type = 'batter'
            except Exception as e:
                pass  # Will return error below
        
        if statcast_df is None or (statcast_df is not None and statcast_df.empty):
            return jsonify({
                "error": f"No statcast data found for {actual_player_name}",
                "data": [],
                "summary": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in statcast_df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(statcast_df[year_column]):
                    statcast_df[year_column] = pd.to_numeric(statcast_df[year_column], errors='coerce')
                statcast_df = statcast_df[
                    (statcast_df[year_column].notna()) & 
                    (statcast_df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in statcast_df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(statcast_df[date_column]):
                        statcast_df[date_column] = pd.to_datetime(statcast_df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    statcast_df = statcast_df[statcast_df[date_column].notna()]
                    statcast_df = statcast_df[
                        (statcast_df[date_column] >= season_start) & 
                        (statcast_df[date_column] <= season_end)
                    ]
            
            if statcast_df.empty:
                return jsonify({
                    "error": f"No statcast data found for {actual_player_name} for season {season}",
                    "data": [],
                    "summary": {}
                }), 200
        
        # Filter to only regular season games (exclude postseason, spring training, etc.)
        if 'game_type' in statcast_df.columns:
            statcast_df = statcast_df[statcast_df['game_type'] == 'R']
        
        # Filter to only events that are batted balls (have events)
        statcast_df = statcast_df[statcast_df['events'].notna()]
        
        # Filter by event type if specified (normalize to lowercase for matching)
        if event_type:
            statcast_df = statcast_df[statcast_df['events'].str.lower() == event_type.lower()]
        
        # Filter by launch speed if specified
        if min_launch_speed is not None and 'launch_speed' in statcast_df.columns:
            statcast_df = statcast_df[
                (statcast_df['launch_speed'].notna()) & 
                (statcast_df['launch_speed'] >= min_launch_speed)
            ]
        
        if max_launch_speed is not None and 'launch_speed' in statcast_df.columns:
            statcast_df = statcast_df[
                (statcast_df['launch_speed'].notna()) & 
                (statcast_df['launch_speed'] <= max_launch_speed)
            ]
        
        # Filter by launch angle if specified
        if min_launch_angle is not None and 'launch_angle' in statcast_df.columns:
            statcast_df = statcast_df[
                (statcast_df['launch_angle'].notna()) & 
                (statcast_df['launch_angle'] >= min_launch_angle)
            ]
        
        if max_launch_angle is not None and 'launch_angle' in statcast_df.columns:
            statcast_df = statcast_df[
                (statcast_df['launch_angle'].notna()) & 
                (statcast_df['launch_angle'] <= max_launch_angle)
            ]
        
        if statcast_df.empty:
            return jsonify({
                "error": "No batted ball data available for the selected filters",
                "data": [],
                "summary": {}
            }), 200
        
        # Count all events (including those without coordinates) for accurate statistics
        event_counts_all = {}
        for _, row in statcast_df.iterrows():
            event_name = str(row['events']).lower() if pd.notna(row['events']) else 'unknown'
            event_counts_all[event_name] = event_counts_all.get(event_name, 0) + 1
        
        # Now filter to only batted balls with hit coordinates for visualization
        statcast_df_with_coords = statcast_df[
            statcast_df['hc_x'].notna() & 
            statcast_df['hc_y'].notna()
        ]
        
        # Prepare spray chart data (only for entries with coordinates)
        spray_data = []
        for _, row in statcast_df_with_coords.iterrows():
            # Normalize event name to lowercase for consistent counting
            event_name = str(row['events']).lower() if pd.notna(row['events']) else 'unknown'
            
            spray_data.append({
                'x': float(row['hc_x']) if pd.notna(row['hc_x']) else None,
                'y': float(row['hc_y']) if pd.notna(row['hc_y']) else None,
                'event': event_name,
                'launch_speed': float(row['launch_speed']) if 'launch_speed' in row and pd.notna(row['launch_speed']) else None,
                'launch_angle': float(row['launch_angle']) if 'launch_angle' in row and pd.notna(row['launch_angle']) else None,
                'hit_distance': float(row['hit_distance_sc']) if 'hit_distance_sc' in row and pd.notna(row['hit_distance_sc']) else None,
                'is_barrel': bool(row['barrel']) if 'barrel' in row and pd.notna(row['barrel']) else None,
            })
        
        # Filter out entries without valid coordinates (shouldn't happen, but just in case)
        spray_data = [d for d in spray_data if d['x'] is not None and d['y'] is not None]
        
        # Calculate summary statistics from all data (not just those with coordinates)
        launch_speeds_all = statcast_df['launch_speed'].dropna().tolist() if 'launch_speed' in statcast_df.columns else []
        launch_angles_all = statcast_df['launch_angle'].dropna().tolist() if 'launch_angle' in statcast_df.columns else []
        
        # Use event counts from all data (accurate counts)
        event_counts = event_counts_all
        
        summary = {
            'total_batted_balls': len(statcast_df),  # Count all batted balls, not just those with coordinates
            'total_visualized': len(spray_data),  # Count of those with coordinates for visualization
            'avg_launch_speed': float(np.mean(launch_speeds_all)) if launch_speeds_all else None,
            'avg_launch_angle': float(np.mean(launch_angles_all)) if launch_angles_all else None,
            'min_launch_speed': float(min(launch_speeds_all)) if launch_speeds_all else None,
            'max_launch_speed': float(max(launch_speeds_all)) if launch_speeds_all else None,
            'min_launch_angle': float(min(launch_angles_all)) if launch_angles_all else None,
            'max_launch_angle': float(max(launch_angles_all)) if launch_angles_all else None,
            'event_counts': event_counts
        }
        
        return jsonify({
            'player': actual_player_name,
            'filters': {
                'season': season,
                'event_type': event_type,
                'min_launch_speed': min_launch_speed,
                'max_launch_speed': max_launch_speed,
                'min_launch_angle': min_launch_angle,
                'max_launch_angle': max_launch_angle
            },
            'data': spray_data,
            'summary': summary
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/barrel-quality-contact', methods=['GET'])
def api_visuals_barrel_quality_contact():
    """Get barrel rate and quality of contact data for visualization"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        player_name = request.args.get('player', '').strip()
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get player data
        try:
            players = csv_loader.get_all_players_summary()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Error loading player data: {str(e)}"}), 500
        
        if not players:
            return jsonify({"error": "No players found in database"}), 404
        
        # Find the selected player
        selected_player_data = None
        player_name_normalized = player_name.strip().lower()
        for player in players:
            player_name_from_data = player.get('name', '').strip().lower()
            if player_name_from_data == player_name_normalized:
                selected_player_data = player
                break
        
        if not selected_player_data:
            for player in players:
                player_name_from_data = player.get('name', '').strip().lower()
                if player_name_normalized in player_name_from_data or player_name_from_data in player_name_normalized:
                    selected_player_data = player
                    break
        
        if not selected_player_data:
            similar_names = [p['name'] for p in players if player_name_normalized[:3] in p['name'].lower()][:5]
            error_msg = f"Player '{player_name}' not found"
            if similar_names:
                error_msg += f". Similar names: {', '.join(similar_names)}"
            return jsonify({"error": error_msg}), 404
        
        actual_player_name = selected_player_data.get('name')
        
        # Get batter ID
        try:
            batter_id = lookup_batter_id(actual_player_name)
        except Exception as e:
            return jsonify({"error": f"Could not find player ID for '{actual_player_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            statcast_df = fetch_batter_statcast(batter_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if statcast_df is None or statcast_df.empty:
            return jsonify({
                "error": f"No statcast data found for {actual_player_name}",
                "data": [],
                "summary": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in statcast_df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(statcast_df[year_column]):
                    statcast_df[year_column] = pd.to_numeric(statcast_df[year_column], errors='coerce')
                statcast_df = statcast_df[
                    (statcast_df[year_column].notna()) & 
                    (statcast_df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in statcast_df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(statcast_df[date_column]):
                        statcast_df[date_column] = pd.to_datetime(statcast_df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    statcast_df = statcast_df[statcast_df[date_column].notna()]
                    statcast_df = statcast_df[
                        (statcast_df[date_column] >= season_start) & 
                        (statcast_df[date_column] <= season_end)
                    ]
            
            if statcast_df.empty:
                return jsonify({
                    "error": f"No statcast data found for {actual_player_name} for season {season}",
                    "data": [],
                    "summary": {}
                }), 200
        
        # Filter to only regular season games
        if 'game_type' in statcast_df.columns:
            statcast_df = statcast_df[statcast_df['game_type'] == 'R']
        
        # Filter to only events that are batted balls (have events)
        statcast_df = statcast_df[statcast_df['events'].notna()]
        
        if statcast_df.empty:
            return jsonify({
                "error": "No batted ball data available for the selected filters",
                "data": [],
                "summary": {}
            }), 200
        
        # Helper function to classify barrel (based on Statcast barrel definition)
        def classify_barrel(la, ev):
            """Classify if a batted ball is a barrel based on launch angle and exit velocity
            Uses Statcast's barrel definition with varying exit velocity thresholds by launch angle"""
            if pd.isna(la) or pd.isna(ev) or la < -50 or la > 50:
                return False
            
            # Barrel classification: 8-50° launch angle with exit velocity thresholds
            # The minimum exit velocity varies based on launch angle
            if la >= 8 and la <= 50:
                # Calculate minimum exit velocity based on launch angle
                # Formula: min_ev increases as launch angle decreases below 26°
                # and decreases as launch angle increases above 26°
                if la <= 8:
                    min_ev = 98
                elif la <= 12:
                    min_ev = 98
                elif la <= 14:
                    min_ev = 98
                elif la <= 16:
                    min_ev = 97
                elif la <= 18:
                    min_ev = 96
                elif la <= 20:
                    min_ev = 95
                elif la <= 22:
                    min_ev = 94
                elif la <= 24:
                    min_ev = 93
                elif la <= 26:
                    min_ev = 92
                elif la <= 28:
                    min_ev = 91
                elif la <= 30:
                    min_ev = 90
                elif la <= 32:
                    min_ev = 89
                elif la <= 34:
                    min_ev = 88
                elif la <= 36:
                    min_ev = 87
                elif la <= 38:
                    min_ev = 86
                elif la <= 40:
                    min_ev = 85
                elif la <= 42:
                    min_ev = 84
                elif la <= 44:
                    min_ev = 83
                elif la <= 46:
                    min_ev = 82
                elif la <= 48:
                    min_ev = 81
                else:  # 50°
                    min_ev = 80
                
                # Maximum exit velocity (typically 117 mph for most angles, but can be higher)
                max_ev = 117
                
                if ev >= min_ev and ev <= max_ev:
                    return True
            
            return False
        
        # Helper function to classify quality of contact
        def classify_quality(la, ev):
            """Classify quality of contact"""
            if pd.isna(la) or pd.isna(ev):
                return 'unknown'
            
            # Check for barrel first
            if classify_barrel(la, ev):
                return 'barrel'
            
            # Solid contact: hard hit (≥95 mph) with good launch angle (8-32°)
            if ev >= 95 and la >= 8 and la <= 32:
                return 'solid'
            
            # Flares/burners: hard hit (≥95 mph) but launch angle outside sweet spot
            if ev >= 95 and (la < 8 or la > 32):
                return 'flare'
            
            # Topped: negative or very low launch angle (ground balls)
            if la < 8:
                return 'topped'
            
            # Weak contact: low exit velocity
            if ev < 95:
                return 'weak'
            
            return 'other'
        
        # Prepare batted ball data
        batted_ball_data = []
        quality_counts = {
            'barrel': 0,
            'solid': 0,
            'flare': 0,
            'topped': 0,
            'weak': 0,
            'other': 0,
            'unknown': 0
        }
        
        total_batted_balls = len(statcast_df)
        barrels = 0
        hard_hits = 0  # ≥95 mph
        sweet_spots = 0  # 8-32° launch angle
        
        for _, row in statcast_df.iterrows():
            la = row.get('launch_angle')
            ev = row.get('launch_speed')
            events = row.get('events', '')
            
            # Use Statcast barrel classification if available
            is_barrel = False
            if 'barrel' in row and pd.notna(row['barrel']):
                is_barrel = bool(row['barrel'])
            elif not pd.isna(la) and not pd.isna(ev):
                is_barrel = classify_barrel(la, ev)
            
            if is_barrel:
                barrels += 1
            
            if not pd.isna(ev) and ev >= 95:
                hard_hits += 1
            
            if not pd.isna(la) and la >= 8 and la <= 32:
                sweet_spots += 1
            
            # Classify quality of contact
            quality = classify_quality(la, ev) if not pd.isna(la) and not pd.isna(ev) else 'unknown'
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            batted_ball_data.append({
                'launch_angle': float(la) if not pd.isna(la) else None,
                'exit_velocity': float(ev) if not pd.isna(ev) else None,
                'events': str(events) if pd.notna(events) else None,
                'is_barrel': is_barrel,
                'quality': quality,
                'estimated_woba_using_speedangle': float(row['estimated_woba_using_speedangle']) if 'estimated_woba_using_speedangle' in row and pd.notna(row['estimated_woba_using_speedangle']) else None,
                'estimated_ba_using_speedangle': float(row['estimated_ba_using_speedangle']) if 'estimated_ba_using_speedangle' in row and pd.notna(row['estimated_ba_using_speedangle']) else None,
                'estimated_slg_using_speedangle': float(row['estimated_slg_using_speedangle']) if 'estimated_slg_using_speedangle' in row and pd.notna(row['estimated_slg_using_speedangle']) else None,
            })
        
        # Calculate summary statistics
        launch_speeds = statcast_df['launch_speed'].dropna().tolist() if 'launch_speed' in statcast_df.columns else []
        launch_angles = statcast_df['launch_angle'].dropna().tolist() if 'launch_angle' in statcast_df.columns else []
        
        barrel_rate = (barrels / total_batted_balls * 100) if total_batted_balls > 0 else 0
        hard_hit_pct = (hard_hits / total_batted_balls * 100) if total_batted_balls > 0 else 0
        sweet_spot_pct = (sweet_spots / total_batted_balls * 100) if total_batted_balls > 0 else 0
        
        summary = {
            'total_batted_balls': total_batted_balls,
            'barrel_rate': barrel_rate,
            'barrels': barrels,
            'hard_hit_pct': hard_hit_pct,
            'sweet_spot_pct': sweet_spot_pct,
            'avg_exit_velocity': float(np.mean(launch_speeds)) if launch_speeds else None,
            'avg_launch_angle': float(np.mean(launch_angles)) if launch_angles else None,
            'quality_counts': quality_counts
        }
        
        return jsonify({
            'player': actual_player_name,
            'filters': {
                'season': season
            },
            'data': batted_ball_data,
            'summary': summary
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating barrel analysis: {str(e)}"}), 500

@app.route('/api/visuals/pitchplots', methods=['GET'])
def api_visuals_pitchplots():
    """Get pitch plots data for visualization - returns Trackman-style movement data"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        pitcher_name = request.args.get('pitcher', '').strip()
        if not pitcher_name:
            return jsonify({"error": "Pitcher name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Filter options
        pitch_type = request.args.get('pitch_type', '').strip() or None
        batter_hand = request.args.get('batter_hand', '').strip() or None
        count = request.args.get('count', '').strip() or None
        min_velocity = request.args.get('min_velocity', type=float)
        max_velocity = request.args.get('max_velocity', type=float)
        min_hb = request.args.get('min_hb', type=float)
        max_hb = request.args.get('max_hb', type=float)
        min_vb = request.args.get('min_vb', type=float)
        max_vb = request.args.get('max_vb', type=float)
        normalize_arm_side = request.args.get('normalize_arm_side', '0') == '1'
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_pitcher_statcast
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Create lookup function for pitcher ID (similar to batter lookup)
        def lookup_pitcher_id(name: str) -> int:
            """Look up MLBAM ID for a pitcher by name"""
            from scrape_savant import lookup_batter_id
            # The lookup_batter_id function works for both batters and pitchers
            return lookup_batter_id(name)
        
        # Get pitcher ID
        try:
            pitcher_id = lookup_pitcher_id(pitcher_name)
        except Exception as e:
            return jsonify({"error": f"Could not find pitcher ID for '{pitcher_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            statcast_df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if statcast_df is None or statcast_df.empty:
            return jsonify({
                "error": f"No statcast data found for {pitcher_name}",
                "pitcher": pitcher_name,
                "pitches": [],
                "summary": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in statcast_df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(statcast_df[year_column]):
                    statcast_df[year_column] = pd.to_numeric(statcast_df[year_column], errors='coerce')
                statcast_df = statcast_df[
                    (statcast_df[year_column].notna()) & 
                    (statcast_df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in statcast_df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(statcast_df[date_column]):
                        statcast_df[date_column] = pd.to_datetime(statcast_df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    statcast_df = statcast_df[statcast_df[date_column].notna()]
                    statcast_df = statcast_df[
                        (statcast_df[date_column] >= season_start) & 
                        (statcast_df[date_column] <= season_end)
                    ]
            
            if statcast_df.empty:
                return jsonify({
                    "error": f"No statcast data found for {pitcher_name} for season {season}",
                    "pitcher": pitcher_name,
                    "pitches": [],
                    "summary": {}
                }), 200
        
        # Filter by pitch type if specified
        if pitch_type and 'pitch_type' in statcast_df.columns:
            statcast_df = statcast_df[statcast_df['pitch_type'] == pitch_type]
        
        # Filter by batter handedness if specified
        if batter_hand and 'stand' in statcast_df.columns:
            statcast_df = statcast_df[statcast_df['stand'] == batter_hand]
        
        # Filter by count if specified
        if count:
            count_parts = count.split('-')
            if len(count_parts) == 2:
                try:
                    balls_filter = int(count_parts[0])
                    strikes_filter = int(count_parts[1])
                    if 'balls' in statcast_df.columns and 'strikes' in statcast_df.columns:
                        statcast_df = statcast_df[
                            (statcast_df['balls'] == balls_filter) & 
                            (statcast_df['strikes'] == strikes_filter)
                        ]
                except ValueError:
                    pass
        
        # Filter by velocity if specified
        if min_velocity is not None and 'release_speed' in statcast_df.columns:
            statcast_df = statcast_df[
                (statcast_df['release_speed'].notna()) & 
                (statcast_df['release_speed'] >= min_velocity)
            ]
        
        if max_velocity is not None and 'release_speed' in statcast_df.columns:
            statcast_df = statcast_df[
                (statcast_df['release_speed'].notna()) & 
                (statcast_df['release_speed'] <= max_velocity)
            ]
        
        # Filter out rows without movement data
        if 'pfx_x' not in statcast_df.columns or 'pfx_z' not in statcast_df.columns:
            return jsonify({
                "error": "Movement data (pfx_x, pfx_z) not available in statcast data",
                "pitcher": pitcher_name,
                "pitches": [],
                "summary": {}
            }), 200
        
        statcast_df = statcast_df[
            statcast_df['pfx_x'].notna() & 
            statcast_df['pfx_z'].notna()
        ]
        
        if statcast_df.empty:
            return jsonify({
                "error": "No valid movement data found",
                "pitcher": pitcher_name,
                "pitches": [],
                "summary": {}
            }), 200
        
        # Get pitcher handedness
        pitcher_hand = 'R'  # Default
        if 'p_throws' in statcast_df.columns:
            throws_values = statcast_df['p_throws'].dropna()
            if len(throws_values) > 0:
                pitcher_hand = str(throws_values.mode().iloc[0]) if len(throws_values.mode()) > 0 else 'R'
        
        # Convert movement to inches and normalize if requested
        if normalize_arm_side:
            sign = statcast_df['p_throws'].map({"R": -1, "L": 1}).fillna(-1)
        else:
            sign = -1
        
        statcast_df['pfx_x_inches'] = statcast_df['pfx_x'] * 12 * sign
        statcast_df['pfx_z_inches'] = statcast_df['pfx_z'] * 12
        
        # Filter by horizontal break if specified
        if min_hb is not None:
            statcast_df = statcast_df[statcast_df['pfx_x_inches'] >= min_hb]
        
        if max_hb is not None:
            statcast_df = statcast_df[statcast_df['pfx_x_inches'] <= max_hb]
        
        # Filter by vertical break if specified
        if min_vb is not None:
            statcast_df = statcast_df[statcast_df['pfx_z_inches'] >= min_vb]
        
        if max_vb is not None:
            statcast_df = statcast_df[statcast_df['pfx_z_inches'] <= max_vb]
        
        if statcast_df.empty:
            return jsonify({
                "error": "No pitch data available for the selected filters",
                "pitcher": pitcher_name,
                "pitches": [],
                "summary": {}
            }), 200
        
        # Prepare pitch data
        pitches = []
        for _, row in statcast_df.iterrows():
            pitch_data = {
                'pitch_type': str(row.get('pitch_type', 'UN')).upper() if pd.notna(row.get('pitch_type')) else 'UN',
                'horizontal_break': float(row['pfx_x_inches']) if pd.notna(row['pfx_x_inches']) else None,
                'vertical_break': float(row['pfx_z_inches']) if pd.notna(row['pfx_z_inches']) else None,
                'velocity': float(row['release_speed']) if 'release_speed' in row and pd.notna(row['release_speed']) else None,
                'count': f"{int(row['balls'])}-{int(row['strikes'])}" if 'balls' in row and 'strikes' in row and pd.notna(row['balls']) and pd.notna(row['strikes']) else None,
                'release_height': float(row['release_pos_z']) if 'release_pos_z' in row and pd.notna(row['release_pos_z']) else None,
                'release_side': float(-row['release_pos_x']) if 'release_pos_x' in row and pd.notna(row['release_pos_x']) else None,
                'release_extension': float(row['release_extension']) if 'release_extension' in row and pd.notna(row['release_extension']) else None,
                'spin_rate': float(row['release_spin_rate']) if 'release_spin_rate' in row and pd.notna(row['release_spin_rate']) else None,
                'spin_axis': float(row['spin_axis']) if 'spin_axis' in row and pd.notna(row['spin_axis']) else None
            }
            
            # Only include pitches with valid movement data
            if pitch_data['horizontal_break'] is not None and pitch_data['vertical_break'] is not None:
                pitches.append(pitch_data)
        
        # Calculate overall summary statistics
        velocities = [p['velocity'] for p in pitches if p['velocity'] is not None]
        horizontal_breaks = [p['horizontal_break'] for p in pitches if p['horizontal_break'] is not None]
        vertical_breaks = [p['vertical_break'] for p in pitches if p['vertical_break'] is not None]
        
        # Calculate per-pitch-type statistics
        pitch_type_stats = []
        total_pitches = len(statcast_df)
        
        # Normalize pitch type labels (similar to plots_movement.py)
        def normalize_pitch_type(pt):
            mapping = {
                "FA": "FF", "FO": "FF", "SV": "SL", "ST": "SL",
                "KC": "CU", "CS": "CU", "UN": "FF"
            }
            pt_str = str(pt).upper() if pd.notna(pt) else "UN"
            return mapping.get(pt_str, pt_str)
        
        statcast_df['pitch_type_normalized'] = statcast_df['pitch_type'].apply(normalize_pitch_type)
        
        # Group by normalized pitch type
        for pitch_type in statcast_df['pitch_type_normalized'].unique():
            if pd.isna(pitch_type):
                continue
                
            pitch_type_df = statcast_df[statcast_df['pitch_type_normalized'] == pitch_type]
            pitch_count = len(pitch_type_df)
            
            if pitch_count < 1:
                continue
            
            # Calculate metrics
            pitch_stats = {
                'pitch_type': str(pitch_type),
                'count': pitch_count,
                'usage_pct': float((pitch_count / total_pitches * 100)) if total_pitches > 0 else 0.0
            }
            
            # Velocity stats
            if 'release_speed' in pitch_type_df.columns:
                velo_data = pitch_type_df['release_speed'].dropna()
                if len(velo_data) > 0:
                    pitch_stats['velocity_avg'] = float(velo_data.mean())
                    pitch_stats['velocity_min'] = float(velo_data.min())
                    pitch_stats['velocity_max'] = float(velo_data.max())
            
            # Movement stats (already in inches)
            hb_data = pitch_type_df['pfx_x_inches'].dropna()
            if len(hb_data) > 0:
                pitch_stats['horizontal_break_avg'] = float(hb_data.mean())
            
            vb_data = pitch_type_df['pfx_z_inches'].dropna()
            if len(vb_data) > 0:
                pitch_stats['vertical_break_avg'] = float(vb_data.mean())
            
            # Release position
            if 'release_pos_z' in pitch_type_df.columns:
                release_h_data = pitch_type_df['release_pos_z'].dropna()
                if len(release_h_data) > 0:
                    pitch_stats['release_height'] = float(release_h_data.mean())
            
            if 'release_pos_x' in pitch_type_df.columns:
                release_x_data = pitch_type_df['release_pos_x'].dropna()
                if len(release_x_data) > 0:
                    # Negate to match convention (left side positive)
                    pitch_stats['release_side'] = float(-release_x_data.mean())
            
            # Release extension
            if 'release_extension' in pitch_type_df.columns:
                ext_data = pitch_type_df['release_extension'].dropna()
                if len(ext_data) > 0:
                    pitch_stats['release_extension'] = float(ext_data.mean())
            
            # Spin rate
            if 'release_spin_rate' in pitch_type_df.columns:
                spin_data = pitch_type_df['release_spin_rate'].dropna()
                if len(spin_data) > 0:
                    pitch_stats['spin_rate_avg'] = float(spin_data.mean())
            
            # Spin axis
            if 'spin_axis' in pitch_type_df.columns:
                axis_data = pitch_type_df['spin_axis'].dropna()
                if len(axis_data) > 0:
                    # Calculate circular mean for spin axis (0-360 degrees)
                    axis_rad = np.deg2rad(axis_data)
                    mean_cos = np.cos(axis_rad).mean()
                    mean_sin = np.sin(axis_rad).mean()
                    mean_axis = np.rad2deg(np.arctan2(mean_sin, mean_cos))
                    if mean_axis < 0:
                        mean_axis += 360
                    pitch_stats['spin_axis_avg'] = float(mean_axis)
            
            # Release velocity components (for axis calculation if needed)
            if 'vx0' in pitch_type_df.columns and 'vy0' in pitch_type_df.columns and 'vz0' in pitch_type_df.columns:
                vx_data = pitch_type_df['vx0'].dropna()
                vy_data = pitch_type_df['vy0'].dropna()
                vz_data = pitch_type_df['vz0'].dropna()
                if len(vx_data) > 0 and len(vy_data) > 0 and len(vz_data) > 0:
                    pitch_stats['release_vx_avg'] = float(vx_data.mean())
                    pitch_stats['release_vy_avg'] = float(vy_data.mean())
                    pitch_stats['release_vz_avg'] = float(vz_data.mean())
            
            pitch_type_stats.append(pitch_stats)
        
        # Sort by usage percentage (descending)
        pitch_type_stats.sort(key=lambda x: x.get('usage_pct', 0), reverse=True)
        
        summary = {
            'total_pitches': len(pitches),
            'avg_velocity': float(np.mean(velocities)) if velocities else None,
            'min_velocity': float(min(velocities)) if velocities else None,
            'max_velocity': float(max(velocities)) if velocities else None,
            'avg_hb': float(np.mean(horizontal_breaks)) if horizontal_breaks else None,
            'avg_vb': float(np.mean(vertical_breaks)) if vertical_breaks else None,
            'pitcher_hand': pitcher_hand,
            'pitch_type_stats': pitch_type_stats
        }
        
        return jsonify({
            'pitcher': pitcher_name,
            'pitches': pitches,
            'summary': summary,
            'filters': {
                'season': season,
                'pitch_type': pitch_type,
                'batter_hand': batter_hand,
                'count': count,
                'min_velocity': min_velocity,
                'max_velocity': max_velocity,
                'min_hb': min_hb,
                'max_hb': max_hb,
                'min_vb': min_vb,
                'max_vb': max_vb,
                'normalize_arm_side': normalize_arm_side
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/pitch-mix-analysis', methods=['GET'])
def api_visuals_pitch_mix_analysis():
    """Get comprehensive pitch mix analysis data with breakdowns by count, situation, and batter handedness"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        pitcher_name = request.args.get('pitcher', '').strip()
        if not pitcher_name:
            return jsonify({"error": "Pitcher name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_pitcher_statcast
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Create lookup function for pitcher ID
        def lookup_pitcher_id(name: str) -> int:
            """Look up MLBAM ID for a pitcher by name"""
            from scrape_savant import lookup_batter_id
            return lookup_batter_id(name)
        
        # Get pitcher ID
        try:
            pitcher_id = lookup_pitcher_id(pitcher_name)
        except Exception as e:
            return jsonify({"error": f"Could not find pitcher ID for '{pitcher_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {pitcher_name}",
                "pitcher": pitcher_name,
                "data": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
            
            if df.empty:
                return jsonify({
                    "error": f"No statcast data found for {pitcher_name} for season {season}",
                    "pitcher": pitcher_name,
                    "data": {}
                }), 200
        
        # Filter out rows without necessary data
        df = df.dropna(subset=['pitch_type', 'balls', 'strikes'])
        if df.empty:
            return jsonify({
                "error": "No valid pitch data available",
                "pitcher": pitcher_name,
                "data": {}
            }), 200
        
        # Create count column
        df['count'] = df['balls'].astype(int).astype(str) + '-' + df['strikes'].astype(int).astype(str)
        
        # Define situations
        def get_situation(row):
            """Determine game situation based on base runners and inning"""
            inning = row.get('inning', 0) if pd.notna(row.get('inning')) else 0
            on_1b = row.get('on_1b', pd.NA) if 'on_1b' in row else pd.NA
            on_2b = row.get('on_2b', pd.NA) if 'on_2b' in row else pd.NA
            on_3b = row.get('on_3b', pd.NA) if 'on_3b' in row else pd.NA
            
            runners_on = 0
            if pd.notna(on_1b):
                runners_on += 1
            if pd.notna(on_2b):
                runners_on += 1
            if pd.notna(on_3b):
                runners_on += 1
            
            if runners_on == 0:
                return "Bases Empty"
            elif runners_on == 1:
                if pd.notna(on_3b):
                    return "Runner on 3rd"
                elif pd.notna(on_2b):
                    return "Runner on 2nd"
                else:
                    return "Runner on 1st"
            elif runners_on == 2:
                if pd.notna(on_2b) and pd.notna(on_3b):
                    return "Runners on 2nd & 3rd"
                elif pd.notna(on_1b) and pd.notna(on_3b):
                    return "Runners on 1st & 3rd"
                else:
                    return "Runners on 1st & 2nd"
            else:
                return "Bases Loaded"
        
        df['situation'] = df.apply(get_situation, axis=1)
        
        # Get batter handedness
        if 'stand' not in df.columns:
            df['stand'] = 'R'  # Default
        
        # Calculate effectiveness metrics
        desc = df['description'].astype(str).str.lower()
        df['is_strike'] = desc.isin([
            'called_strike', 'foul', 'foul_tip', 'swinging_strike', 
            'swinging_strike_blocked', 'foul_bunt', 'hit_into_play'
        ])
        df['is_swing'] = desc.isin([
            'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 
            'missed_bunt', 'foul_bunt', 'hit_into_play'
        ])
        df['is_whiff'] = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'missed_bunt'])
        
        # Determine if pitch is in zone
        if 'zone' in df.columns:
            # Zone column exists - zone values 1-9 are in zone, others are not
            df['is_zone'] = df['zone'].between(1, 9, inclusive='both')
        elif 'plate_x' in df.columns and 'plate_z' in df.columns:
            # Approximate strike zone: plate_x from -0.85 to 0.85, plate_z from 1.5 to 3.5
            df['is_zone'] = (
                df['plate_x'].between(-0.85, 0.85, inclusive='both') &
                df['plate_z'].between(1.5, 3.5, inclusive='both')
            )
        else:
            df['is_zone'] = False
        
        # Helper function to calculate metrics for a group
        def calculate_metrics(group_df):
            total = len(group_df)
            if total == 0:
                return {}
            
            strikes = group_df['is_strike'].sum()
            swings = group_df['is_swing'].sum()
            whiffs = group_df['is_whiff'].sum()
            in_zone = group_df['is_zone'].sum() if 'is_zone' in group_df.columns else 0
            
            metrics = {
                'usage_pct': 100.0,
                'strike_rate': (strikes / total * 100) if total > 0 else 0.0,
                'swing_rate': (swings / total * 100) if total > 0 else 0.0,
                'whiff_rate': (whiffs / swings * 100) if swings > 0 else None,
                'zone_rate': (in_zone / total * 100) if total > 0 else 0.0,
            }
            
            # xwOBA if available
            if 'estimated_woba_using_speedangle' in group_df.columns:
                xwoba_values = group_df['estimated_woba_using_speedangle'].dropna()
                if len(xwoba_values) > 0:
                    metrics['xwoba'] = float(xwoba_values.mean())
                else:
                    metrics['xwoba'] = None
            elif 'woba_value' in group_df.columns:
                woba_values = group_df['woba_value'].dropna()
                if len(woba_values) > 0:
                    metrics['xwoba'] = float(woba_values.mean())
                else:
                    metrics['xwoba'] = None
            else:
                metrics['xwoba'] = None
            
            # Average velocity
            if 'release_speed' in group_df.columns:
                velo_values = group_df['release_speed'].dropna()
                if len(velo_values) > 0:
                    metrics['avg_velocity'] = float(velo_values.mean())
                else:
                    metrics['avg_velocity'] = None
            else:
                metrics['avg_velocity'] = None
            
            return metrics
        
        # Breakdown by count
        count_data = {}
        for count in df['count'].unique():
            count_df = df[df['count'] == count]
            total_pitches = len(count_df)
            
            if total_pitches == 0:
                continue
            
            count_breakdown = {
                'total_pitches': total_pitches,
                'by_pitch_type': {}
            }
            
            for pitch_type in count_df['pitch_type'].unique():
                pitch_df = count_df[count_df['pitch_type'] == pitch_type]
                pitch_count = len(pitch_df)
                
                if pitch_count == 0:
                    continue
                
                metrics = calculate_metrics(pitch_df)
                metrics['usage_pct'] = (pitch_count / total_pitches * 100) if total_pitches > 0 else 0.0
                metrics['pitch_count'] = pitch_count
                
                count_breakdown['by_pitch_type'][pitch_type] = metrics
            
            count_data[count] = count_breakdown
        
        # Breakdown by situation
        situation_data = {}
        for situation in df['situation'].unique():
            situation_df = df[df['situation'] == situation]
            total_pitches = len(situation_df)
            
            if total_pitches == 0:
                continue
            
            situation_breakdown = {
                'total_pitches': total_pitches,
                'by_pitch_type': {}
            }
            
            for pitch_type in situation_df['pitch_type'].unique():
                pitch_df = situation_df[situation_df['pitch_type'] == pitch_type]
                pitch_count = len(pitch_df)
                
                if pitch_count == 0:
                    continue
                
                metrics = calculate_metrics(pitch_df)
                metrics['usage_pct'] = (pitch_count / total_pitches * 100) if total_pitches > 0 else 0.0
                metrics['pitch_count'] = pitch_count
                
                situation_breakdown['by_pitch_type'][pitch_type] = metrics
            
            situation_data[situation] = situation_breakdown
        
        # Breakdown by batter handedness
        batter_hand_data = {}
        for stand in ['R', 'L']:
            if stand not in df['stand'].values:
                continue
            
            hand_df = df[df['stand'] == stand]
            total_pitches = len(hand_df)
            
            if total_pitches == 0:
                continue
            
            hand_breakdown = {
                'total_pitches': total_pitches,
                'by_pitch_type': {}
            }
            
            for pitch_type in hand_df['pitch_type'].unique():
                pitch_df = hand_df[hand_df['pitch_type'] == pitch_type]
                pitch_count = len(pitch_df)
                
                if pitch_count == 0:
                    continue
                
                metrics = calculate_metrics(pitch_df)
                metrics['usage_pct'] = (pitch_count / total_pitches * 100) if total_pitches > 0 else 0.0
                metrics['pitch_count'] = pitch_count
                
                hand_breakdown['by_pitch_type'][pitch_type] = metrics
            
            batter_hand_data[stand] = hand_breakdown
        
        # Overall pitch mix
        overall_total = len(df)
        overall_data = {
            'total_pitches': overall_total,
            'by_pitch_type': {}
        }
        
        for pitch_type in df['pitch_type'].unique():
            pitch_df = df[df['pitch_type'] == pitch_type]
            pitch_count = len(pitch_df)
            
            if pitch_count == 0:
                continue
            
            metrics = calculate_metrics(pitch_df)
            metrics['usage_pct'] = (pitch_count / overall_total * 100) if overall_total > 0 else 0.0
            metrics['pitch_count'] = pitch_count
            
            overall_data['by_pitch_type'][pitch_type] = metrics
        
        # Get pitcher handedness
        pitcher_hand = 'R'  # Default
        if 'p_throws' in df.columns:
            throws_values = df['p_throws'].dropna()
            if len(throws_values) > 0:
                pitcher_hand = str(throws_values.mode().iloc[0]) if len(throws_values.mode()) > 0 else 'R'
        
        return jsonify({
            'pitcher': pitcher_name,
            'pitcher_hand': pitcher_hand,
            'season': season,
            'total_pitches': overall_total,
            'breakdowns': {
                'by_count': count_data,
                'by_situation': situation_data,
                'by_batter_hand': batter_hand_data,
                'overall': overall_data
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/pitch-arsenal-effectiveness', methods=['GET'])
def api_visuals_pitch_arsenal_effectiveness():
    """Get comprehensive pitch arsenal effectiveness data with run value, whiff rates, ground ball rates, and putaway percentages"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        pitcher_name = request.args.get('pitcher', '').strip()
        if not pitcher_name:
            return jsonify({"error": "Pitcher name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        batter_hand = request.args.get('batter_hand', '').strip() or None
        count_filter = request.args.get('count_filter', '').strip() or None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_pitcher_statcast
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Create lookup function for pitcher ID
        def lookup_pitcher_id(name: str) -> int:
            """Look up MLBAM ID for a pitcher by name"""
            from scrape_savant import lookup_batter_id
            return lookup_batter_id(name)
        
        # Get pitcher ID
        try:
            pitcher_id = lookup_pitcher_id(pitcher_name)
        except Exception as e:
            return jsonify({"error": f"Could not find pitcher ID for '{pitcher_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {pitcher_name}",
                "pitcher": pitcher_name,
                "data": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
            
            if df.empty:
                return jsonify({
                    "error": f"No statcast data found for {pitcher_name} for season {season}",
                    "pitcher": pitcher_name,
                    "data": {}
                }), 200
        
        # Filter by batter handedness if specified
        if batter_hand and 'stand' in df.columns:
            df = df[df['stand'] == batter_hand]
        
        # Filter by count if specified
        if count_filter:
            if count_filter == '2-strikes':
                df = df[df['strikes'] == 2]
            else:
                balls, strikes = map(int, count_filter.split('-'))
                df = df[(df['balls'] == balls) & (df['strikes'] == strikes)]
        
        # Filter out rows without necessary data
        df = df.dropna(subset=['pitch_type'])
        if df.empty:
            return jsonify({
                "error": "No valid pitch data available",
                "pitcher": pitcher_name,
                "data": {}
            }), 200
        
        # Calculate run value (delta_run_exp if available, otherwise use estimated_woba_against)
        if 'delta_run_exp' in df.columns:
            df['run_value'] = df['delta_run_exp']
        elif 'estimated_woba_using_speedangle' in df.columns:
            # Approximate run value from wOBA (rough conversion: wOBA * 1.2 - 0.3)
            df['run_value'] = df['estimated_woba_using_speedangle'] * 1.2 - 0.3
        else:
            df['run_value'] = 0.0
        
        # Calculate whiff rates
        desc = df['description'].astype(str).str.lower()
        df['is_strike'] = desc.isin([
            'called_strike', 'foul', 'foul_tip', 'swinging_strike', 
            'swinging_strike_blocked', 'foul_bunt', 'hit_into_play'
        ])
        df['is_whiff'] = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'missed_bunt'])
        df['is_swing'] = desc.isin([
            'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 
            'missed_bunt', 'foul_bunt', 'hit_into_play'
        ])
        
        # Calculate ground ball rates (launch angle < 10 degrees)
        if 'launch_angle' in df.columns:
            df['is_ground_ball'] = (df['launch_angle'].notna()) & (df['launch_angle'] < 10)
            df['is_batted_ball'] = df['launch_angle'].notna()
        else:
            df['is_ground_ball'] = False
            df['is_batted_ball'] = False
        
        # Calculate strikeouts on 2-strike counts
        df['is_two_strike'] = df['strikes'] == 2
        df['is_strikeout'] = desc.isin(['strikeout', 'strikeout_double_play'])
        
        # Helper function to create zone heatmap data
        def create_zone_heatmap(group_df, metric_col, min_samples=5):
            """Create heatmap data for a zone-based metric"""
            if len(group_df) < min_samples:
                return {'zones': []}
            
            # Filter to pitches with location data
            loc_df = group_df[group_df['plate_x'].notna() & group_df['plate_z'].notna()]
            if len(loc_df) < min_samples:
                return {'zones': []}
            
            # Create grid zones
            zones = []
            x_bins = np.linspace(-2, 2, 10)  # 10 bins from -2 to 2 feet
            z_bins = np.linspace(0, 4, 10)   # 10 bins from 0 to 4 feet
            
            for i in range(len(x_bins) - 1):
                for j in range(len(z_bins) - 1):
                    x_min, x_max = x_bins[i], x_bins[i + 1]
                    z_min, z_max = z_bins[j], z_bins[j + 1]
                    
                    zone_df = loc_df[
                        (loc_df['plate_x'] >= x_min) & (loc_df['plate_x'] < x_max) &
                        (loc_df['plate_z'] >= z_min) & (loc_df['plate_z'] < z_max)
                    ]
                    
                    if len(zone_df) >= min_samples:
                        if metric_col == 'run_value':
                            value = zone_df['run_value'].mean()
                        elif metric_col == 'whiff_rate':
                            swings = zone_df['is_swing'].sum()
                            whiffs = zone_df['is_whiff'].sum()
                            value = (whiffs / swings * 100) if swings > 0 else 0
                        elif metric_col == 'ground_ball_rate':
                            batted = zone_df['is_batted_ball'].sum()
                            ground_balls = zone_df['is_ground_ball'].sum()
                            value = (ground_balls / batted * 100) if batted > 0 else 0
                        else:
                            value = 0
                        
                        zones.append({
                            'x': (x_min + x_max) / 2,
                            'z': (z_min + z_max) / 2,
                            'value': float(value),
                            'samples': len(zone_df)
                        })
            
            return {'zones': zones}
        
        # Get unique pitch types
        pitch_types = df['pitch_type'].unique().tolist()
        
        # Calculate heatmaps for each metric
        run_value_heatmaps = {}
        whiff_rate_heatmaps = {}
        ground_ball_heatmaps = {}
        putaway_data = {}
        summary = {}
        
        for pitch_type in pitch_types:
            pitch_df = df[df['pitch_type'] == pitch_type]
            
            if len(pitch_df) < 10:  # Skip if too few pitches
                continue
            
            # Run value heatmap
            run_value_heatmaps[pitch_type] = create_zone_heatmap(pitch_df, 'run_value')
            
            # Whiff rate heatmap
            swing_df = pitch_df[pitch_df['is_swing']]
            if len(swing_df) >= 10:
                whiff_rate_heatmaps[pitch_type] = create_zone_heatmap(swing_df, 'whiff_rate')
            else:
                whiff_rate_heatmaps[pitch_type] = {'zones': []}
            
            # Ground ball rate heatmap
            batted_df = pitch_df[pitch_df['is_batted_ball']]
            if len(batted_df) >= 10:
                ground_ball_heatmaps[pitch_type] = create_zone_heatmap(batted_df, 'ground_ball_rate')
            else:
                ground_ball_heatmaps[pitch_type] = {'zones': []}
            
            # Putaway percentage (strikeouts on 2-strike counts)
            two_strike_df = pitch_df[pitch_df['is_two_strike']]
            two_strike_count = len(two_strike_df)
            strikeouts = two_strike_df['is_strikeout'].sum()
            
            putaway_data[pitch_type] = {
                'two_strike_pitches': int(two_strike_count),
                'strikeouts': int(strikeouts),
                'putaway_pct': float((strikeouts / two_strike_count * 100) if two_strike_count > 0 else 0)
            }
            
            # Summary statistics
            total_pitches = len(pitch_df)
            overall_total = len(df)
            
            strikes = pitch_df['is_strike'].sum() if 'is_strike' in pitch_df.columns else 0
            swings = pitch_df['is_swing'].sum()
            whiffs = pitch_df['is_whiff'].sum()
            
            # Calculate metrics
            avg_run_value = pitch_df['run_value'].mean() if 'run_value' in pitch_df.columns else 0
            whiff_rate = (whiffs / swings * 100) if swings > 0 else None
            strike_rate = (strikes / total_pitches * 100) if total_pitches > 0 else 0
            
            # Ground ball rate
            batted = pitch_df['is_batted_ball'].sum()
            ground_balls = pitch_df['is_ground_ball'].sum()
            ground_ball_pct = (ground_balls / batted * 100) if batted > 0 else None
            
            # xwOBA
            if 'estimated_woba_using_speedangle' in pitch_df.columns:
                xwoba_values = pitch_df['estimated_woba_using_speedangle'].dropna()
                xwoba = float(xwoba_values.mean()) if len(xwoba_values) > 0 else None
            else:
                xwoba = None
            
            summary[pitch_type] = {
                'usage_pct': (total_pitches / overall_total * 100) if overall_total > 0 else 0,
                'avg_run_value': float(avg_run_value),
                'whiff_rate': float(whiff_rate) if whiff_rate is not None else None,
                'ground_ball_pct': float(ground_ball_pct) if ground_ball_pct is not None else None,
                'putaway_pct': putaway_data[pitch_type]['putaway_pct'],
                'strike_rate': float(strike_rate),
                'xwoba': xwoba
            }
        
        # Get pitcher handedness
        pitcher_hand = 'R'
        if 'p_throws' in df.columns:
            throws_values = df['p_throws'].dropna()
            if len(throws_values) > 0:
                pitcher_hand = str(throws_values.mode().iloc[0]) if len(throws_values.mode()) > 0 else 'R'
        
        return jsonify({
            'pitcher': pitcher_name,
            'pitcher_hand': pitcher_hand,
            'season': season,
            'total_pitches': len(df),
            'pitch_types': pitch_types,
            'run_value_heatmaps': run_value_heatmaps,
            'whiff_rate_heatmaps': whiff_rate_heatmaps,
            'ground_ball_heatmaps': ground_ball_heatmaps,
            'putaway_data': putaway_data,
            'summary': summary
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/count-performance', methods=['GET'])
def api_visuals_count_performance():
    """Get batter performance breakdown by count"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        batter_name = request.args.get('batter', '').strip()
        if not batter_name:
            return jsonify({"error": "Batter name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get batter ID
        try:
            batter_id = lookup_batter_id(batter_name)
        except Exception as e:
            return jsonify({"error": f"Could not find batter ID for '{batter_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            df = fetch_batter_statcast(batter_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {batter_name}",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
            
            if df.empty:
                return jsonify({
                    "error": f"No statcast data found for {batter_name} for season {season}",
                    "batter": batter_name,
                    "data": {}
                }), 200
        
        # Filter out rows without necessary data
        df = df.dropna(subset=['balls', 'strikes'])
        if df.empty:
            return jsonify({
                "error": "No valid pitch data available",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Create count column
        df['count'] = df['balls'].astype(int).astype(str) + '-' + df['strikes'].astype(int).astype(str)
        
        # Define count order
        COUNT_ORDER = ['0-0', '1-0', '0-1', '1-1', '2-0', '2-1', '1-2', '2-2', '3-0', '3-1', '3-2', '0-2']
        df = df[df['count'].isin(COUNT_ORDER)]
        
        # Calculate outcome metrics
        desc = df['description'].astype(str).str.lower()
        
        # Use events column if available (this is the actual at-bat outcome)
        # Otherwise fall back to description
        if 'events' in df.columns:
            events = df['events'].astype(str).str.lower()
            # Hits from events
            df['is_hit'] = events.isin(['single', 'double', 'triple', 'home_run'])
            # Outs from events (field outs, strikeouts, etc.)
            df['is_out'] = events.isin([
                'strikeout', 'strikeout_double_play', 'field_out', 'force_out', 
                'grounded_into_double_play', 'double_play', 'triple_play',
                'fielders_choice', 'fielders_choice_out', 'sac_fly', 'sac_fly_double_play',
                'sac_bunt', 'sac_bunt_double_play', 'bunt_groundout', 'bunt_popout'
            ])
            # Walks from events
            df['is_walk'] = events.isin(['walk', 'intent_walk', 'hit_by_pitch'])
            # Strikeouts from events
            df['is_strikeout'] = events.isin(['strikeout', 'strikeout_double_play'])
            # PA ending indicator
            df['is_pa_ending'] = events.notna() & (events != 'nan') & (events != '')
        else:
            # Fallback to description if events column not available
            df['is_hit'] = desc.isin(['single', 'double', 'triple', 'home_run'])
            df['is_out'] = desc.isin(['strikeout', 'strikeout_double_play', 'field_out', 'force_out', 'grounded_into_double_play', 'double_play', 'triple_play'])
            df['is_walk'] = desc.isin(['walk', 'intent_walk', 'hit_by_pitch'])
            df['is_strikeout'] = desc.isin(['strikeout', 'strikeout_double_play'])
            df['is_pa_ending'] = (
                df['is_hit'] | df['is_out'] | df['is_walk'] | 
                desc.str.contains('hit_into_play', na=False)
            )
        
        # Swing and whiff rates (these are pitch-level, so use description)
        df['is_swing'] = desc.isin(['foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'foul_bunt', 'hit_into_play'])
        df['is_whiff'] = desc.isin(['swinging_strike', 'swinging_strike_blocked', 'missed_bunt'])
        
        # Helper function to calculate metrics for a count
        def calculate_count_metrics(count_df):
            total_pitches = len(count_df)
            if total_pitches == 0:
                return {}
            
            # For at-bat outcomes (hits, outs, walks), only count pitches where at-bat ended
            # (i.e., where events is not null, or is_pa_ending is true)
            if 'events' in count_df.columns:
                # Filter to only pitches where events occurred (at-bat ending pitches)
                pa_df = count_df[count_df['events'].notna() & (count_df['events'].astype(str) != 'nan') & (count_df['events'].astype(str) != '')]
            else:
                # Fallback: use is_pa_ending flag
                pa_df = count_df[count_df['is_pa_ending']]
            
            # Plate appearances ending at this count
            pa_ending = len(pa_df)
            
            # Hits, walks, strikeouts (from at-bat ending pitches only)
            hits = pa_df['is_hit'].sum() if len(pa_df) > 0 else 0
            walks = pa_df['is_walk'].sum() if len(pa_df) > 0 else 0
            strikeouts = pa_df['is_strikeout'].sum() if len(pa_df) > 0 else 0
            outs = pa_df['is_out'].sum() if len(pa_df) > 0 else 0
            
            # Swing and whiff rates
            swings = count_df['is_swing'].sum()
            whiffs = count_df['is_whiff'].sum()
            
            # Batting average (hits / at-bats) - standard format (.XXX)
            # At-bats = hits + outs (excludes walks, hit-by-pitch, sacrifices)
            # We need to make sure we're only counting actual at-bats
            at_bats = hits + outs
            # Only calculate if we have actual at-bats
            if at_bats > 0:
                batting_avg = float(hits / at_bats)
            else:
                batting_avg = None
            
            # On-base percentage (hits + walks) / PA
            obp = ((hits + walks) / pa_ending * 100) if pa_ending > 0 else None
            
            # Strikeout rate
            k_rate = (strikeouts / pa_ending * 100) if pa_ending > 0 else None
            
            # Walk rate
            bb_rate = (walks / pa_ending * 100) if pa_ending > 0 else None
            
            # Swing rate
            swing_rate = (swings / total_pitches * 100) if total_pitches > 0 else 0.0
            
            # Whiff rate
            whiff_rate = (whiffs / swings * 100) if swings > 0 else None
            
            # xwOBA if available
            xwoba = None
            if 'estimated_woba_using_speedangle' in count_df.columns:
                xwoba_values = count_df['estimated_woba_using_speedangle'].dropna()
                if len(xwoba_values) > 0:
                    xwoba = float(xwoba_values.mean())
            elif 'woba_value' in count_df.columns:
                woba_values = count_df['woba_value'].dropna()
                if len(woba_values) > 0:
                    xwoba = float(woba_values.mean())
            
            # Average exit velocity
            avg_ev = None
            if 'launch_speed' in count_df.columns:
                ev_values = count_df['launch_speed'].dropna()
                if len(ev_values) > 0:
                    avg_ev = float(ev_values.mean())
            
            # Hard hit rate (95+ mph)
            hard_hit_rate = None
            if 'launch_speed' in count_df.columns:
                ev_values = count_df['launch_speed'].dropna()
                hard_hits = (ev_values >= 95).sum()
                hard_hit_rate = (hard_hits / len(ev_values) * 100) if len(ev_values) > 0 else None
            
            # Pitch types seen
            pitch_types = {}
            if 'pitch_type' in count_df.columns:
                pitch_type_counts = count_df['pitch_type'].value_counts()
                total_seen = pitch_type_counts.sum()
                for pt, count in pitch_type_counts.items():
                    if pd.notna(pt):
                        pitch_types[str(pt)] = {
                            'count': int(count),
                            'percentage': float(count / total_seen * 100) if total_seen > 0 else 0.0
                        }
            
            return {
                'total_pitches': total_pitches,
                'pa_ending': int(pa_ending),
                'hits': int(hits),
                'walks': int(walks),
                'strikeouts': int(strikeouts),
                'outs': int(outs),
                'at_bats': int(at_bats),
                'batting_avg': batting_avg,
                'obp': obp,
                'k_rate': k_rate,
                'bb_rate': bb_rate,
                'swing_rate': swing_rate,
                'whiff_rate': whiff_rate,
                'xwoba': xwoba,
                'avg_ev': avg_ev,
                'hard_hit_rate': hard_hit_rate,
                'pitch_types_seen': pitch_types
            }
        
        # Calculate metrics for each count
        count_data = {}
        for count in COUNT_ORDER:
            count_df = df[df['count'] == count]
            if len(count_df) == 0:
                continue
            count_data[count] = calculate_count_metrics(count_df)
        
        # Get batter handedness
        batter_hand = 'R'  # Default
        if 'stand' in df.columns:
            stand_values = df['stand'].dropna()
            if len(stand_values) > 0:
                batter_hand = str(stand_values.mode().iloc[0]) if len(stand_values.mode()) > 0 else 'R'
        
        # Overall stats
        overall_metrics = calculate_count_metrics(df)
        
        return jsonify({
            'batter': batter_name,
            'batter_hand': batter_hand,
            'season': season,
            'total_pitches': len(df),
            'overall': overall_metrics,
            'by_count': count_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/velocity-trends', methods=['GET'])
def api_visuals_velocity_trends():
    """Get velocity trends data for pitchers or batters"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        player_name = request.args.get('player', '').strip()
        if not player_name:
            return jsonify({"error": "Player name is required"}), 400
        
        player_type = request.args.get('player_type', 'pitcher').strip().lower()
        view_type = request.args.get('view_type', 'career').strip().lower()
        season = request.args.get('season', '').strip() or None
        game_date = request.args.get('game_date', '').strip() or None
        pitch_type = request.args.get('pitch_type', '').strip() or None
        
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_pitcher_statcast, fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get player ID
        try:
            player_id = lookup_batter_id(player_name)
        except Exception as e:
            return jsonify({"error": f"Could not find player ID for '{player_name}': {str(e)}"}), 404
        
        # Calculate date range
        if view_type == 'career':
            # Get all available data
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        elif season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            # Default to current season
            current_year = datetime.now().year
            start_date = f"{current_year}-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = True
            season = current_year
        
        # Fetch statcast data
        try:
            if player_type == 'pitcher':
                df = fetch_pitcher_statcast(player_id, start_date, end_date)
                velocity_column = 'release_speed'
            else:
                df = fetch_batter_statcast(player_id, start_date, end_date)
                velocity_column = 'launch_speed'
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {player_name}",
                "player": player_name,
                "trends": []
            }), 200
        
        # Filter by season if specified
        if filter_by_season and season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
        
        # Filter by pitch type if specified (pitchers only)
        if pitch_type and player_type == 'pitcher' and 'pitch_type' in df.columns:
            df = df[df['pitch_type'] == pitch_type]
        
        # Filter out rows without velocity data
        if velocity_column not in df.columns:
            return jsonify({
                "error": f"Velocity data ({velocity_column}) not available",
                "player": player_name,
                "trends": []
            }), 200
        
        df = df[df[velocity_column].notna()]
        
        if df.empty:
            return jsonify({
                "error": "No valid velocity data found",
                "player": player_name,
                "trends": []
            }), 200
        
        # Prepare trends data based on view type
        trends = []
        
        if view_type == 'game':
            # Game-level fatigue: group by game and pitch number
            if 'game_date' in df.columns and 'pitch_number' in df.columns:
                # Get game_date column name variations
                date_col = 'game_date'
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Filter by specific game date if provided
                if game_date:
                    try:
                        filter_date = pd.to_datetime(game_date)
                        df = df[df[date_col].dt.date == filter_date.date()]
                    except Exception:
                        pass  # If date parsing fails, don't filter
                
                df = df[df[date_col].notna() & df['pitch_number'].notna()]
                
                for _, row in df.iterrows():
                    game_date_val = row[date_col]
                    if pd.notna(game_date_val):
                        if isinstance(game_date_val, pd.Timestamp):
                            game_date_str = game_date_val.strftime('%Y-%m-%d')
                        else:
                            game_date_str = str(game_date_val)
                    else:
                        game_date_str = 'Unknown'
                    
                    trends.append({
                        'game_date': game_date_str,
                        'pitch_number': int(row['pitch_number']) if pd.notna(row['pitch_number']) else None,
                        'velocity': float(row[velocity_column]) if pd.notna(row[velocity_column]) else None,
                        'season': int(row['game_year']) if 'game_year' in row and pd.notna(row.get('game_year')) else season
                    })
        elif view_type == 'season':
            # Season trends: group by game date
            if 'game_date' in df.columns:
                date_col = 'game_date'
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                df = df[df[date_col].notna()]
                
                for game_date, group in df.groupby(date_col):
                    velocities = group[velocity_column].dropna()
                    if len(velocities) > 0:
                        if isinstance(game_date, pd.Timestamp):
                            game_date_str = game_date.strftime('%Y-%m-%d')
                        else:
                            game_date_str = str(game_date)
                        
                        trends.append({
                            'game_date': game_date_str,
                            'velocity': float(velocities.mean()),
                            'season': int(group['game_year'].iloc[0]) if 'game_year' in group.columns and pd.notna(group['game_year'].iloc[0]) else season
                        })
        else:
            # Career trends: group by season
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[df[year_column].notna()]
                
                for year, group in df.groupby(year_column):
                    velocities = group[velocity_column].dropna()
                    if len(velocities) > 0:
                        trends.append({
                            'season': int(year),
                            'velocity': float(velocities.mean()),
                            'game_date': None
                        })
            else:
                # Fallback: try to extract year from game_date
                if 'game_date' in df.columns:
                    date_col = 'game_date'
                    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df[df[date_col].notna()]
                    df['year'] = df[date_col].dt.year
                    
                    for year, group in df.groupby('year'):
                        velocities = group[velocity_column].dropna()
                        if len(velocities) > 0:
                            trends.append({
                                'season': int(year),
                                'velocity': float(velocities.mean()),
                                'game_date': None
                            })
        
        # Sort trends appropriately
        if view_type == 'career':
            trends.sort(key=lambda x: x['season'] if x['season'] else 0)
        elif view_type == 'season':
            trends.sort(key=lambda x: x['game_date'] if x['game_date'] else '')
        else:
            # Game-level: sort by game_date then pitch_number
            trends.sort(key=lambda x: (x['game_date'] if x['game_date'] else '', x['pitch_number'] if x['pitch_number'] is not None else 0))
        
        return jsonify({
            'player': player_name,
            'player_type': player_type,
            'view_type': view_type,
            'season': season,
            'pitch_type': pitch_type,
            'trends': trends
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/swing-decision-matrix', methods=['GET'])
def api_visuals_swing_decision_matrix():
    """Get swing decision matrix data: run values, optimal decision zones, chase rates, and decision quality metrics"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        batter_name = request.args.get('batter', '').strip()
        if not batter_name:
            return jsonify({"error": "Batter name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        count = request.args.get('count', '').strip() or None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get batter ID
        try:
            batter_id = lookup_batter_id(batter_name)
        except Exception as e:
            return jsonify({"error": f"Could not find batter ID for '{batter_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            df = fetch_batter_statcast(batter_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {batter_name}",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
            
            if df.empty:
                return jsonify({
                    "error": f"No statcast data found for {batter_name} for season {season}",
                    "batter": batter_name,
                    "data": {}
                }), 200
        
        # Filter by count if specified
        if count:
            if 'balls' in df.columns and 'strikes' in df.columns:
                b, s = count.split('-')
                df = df[(df['balls'] == int(b)) & (df['strikes'] == int(s))]
        
        # Filter out rows without necessary data
        df = df.dropna(subset=['pitch_type'])
        if df.empty:
            return jsonify({
                "error": "No valid pitch data available",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Calculate swing, contact, and zone indicators
        desc = df['description'].astype(str).str.lower()
        df['is_swing'] = desc.isin([
            'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 
            'missed_bunt', 'foul_bunt', 'hit_into_play'
        ])
        
        df['is_take'] = ~df['is_swing']
        
        df['is_contact'] = desc.isin([
            'foul', 'foul_tip', 'foul_bunt', 'hit_into_play'
        ])
        
        # Determine strike zone
        df['is_in_zone'] = False
        if 'zone' in df.columns:
            df['is_in_zone'] = df['zone'].between(1, 9, inclusive='both')
            if 'plate_x' in df.columns and 'plate_z' in df.columns:
                zone_na_mask = df['zone'].isna()
                if zone_na_mask.any():
                    df.loc[zone_na_mask, 'is_in_zone'] = (
                        (df.loc[zone_na_mask, 'plate_x'].abs() <= 0.855) &
                        (df.loc[zone_na_mask, 'plate_z'] >= 1.5) &
                        (df.loc[zone_na_mask, 'plate_z'] <= 3.5)
                    )
        elif 'plate_x' in df.columns and 'plate_z' in df.columns:
            df['is_in_zone'] = (
                (df['plate_x'].abs() <= 0.855) &
                (df['plate_z'] >= 1.5) &
                (df['plate_z'] <= 3.5)
            )
        
        # Chase = swing on pitch outside the zone
        df['is_chase'] = df['is_swing'] & ~df['is_in_zone']
        
        # Assign zone numbers (1-9) based on plate_x and plate_z
        def assign_zone(row):
            if pd.isna(row.get('zone')):
                # Estimate zone from plate_x and plate_z
                if 'plate_x' not in row or 'plate_z' not in row:
                    return None
                if pd.isna(row['plate_x']) or pd.isna(row['plate_z']):
                    return None
                
                x = row['plate_x']
                z = row['plate_z']
                
                # Zone boundaries (approximate)
                # Horizontal: -0.855 to 0.855 (3 zones)
                # Vertical: 1.5 to 3.5 (3 zones)
                if x < -0.285:  # Left column
                    col = 0
                elif x < 0.285:  # Middle column
                    col = 1
                else:  # Right column
                    col = 2
                
                if z < 2.17:  # Bottom row
                    row_num = 2
                elif z < 2.83:  # Middle row
                    row_num = 1
                else:  # Top row
                    row_num = 0
                
                return row_num * 3 + col + 1
            else:
                zone_val = row['zone']
                if pd.isna(zone_val) or zone_val < 1 or zone_val > 9:
                    return None
                return int(zone_val)
        
        df['zone_num'] = df.apply(assign_zone, axis=1)
        df = df[df['zone_num'].notna()]
        
        if df.empty:
            return jsonify({
                "error": "No valid zone data available",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Run expectancy matrix (simplified - approximate values)
        # Based on count situation
        run_expectancy = {
            '0-0': 0.475, '1-0': 0.525, '0-1': 0.260,
            '2-0': 0.575, '1-1': 0.350, '0-2': 0.100,
            '3-0': 0.625, '2-1': 0.425, '1-2': 0.175,
            '3-1': 0.550, '2-2': 0.275, '3-2': 0.300
        }
        
        # Calculate run values for each pitch
        def calculate_run_value(row):
            # Get count
            b = int(row.get('balls', 0)) if pd.notna(row.get('balls')) else 0
            s = int(row.get('strikes', 0)) if pd.notna(row.get('strikes')) else 0
            count_str = f"{b}-{s}"
            
            base_re = run_expectancy.get(count_str, 0.35)
            
            # Calculate outcome value
            outcome = row['description'].lower()
            
            # Simplified run values by outcome
            if 'hit_into_play' in outcome:
                # Calculate from wOBA or estimate
                if 'woba_value' in row and pd.notna(row['woba_value']):
                    rv = float(row['woba_value']) - base_re
                elif 'estimated_woba_using_speedangle' in row and pd.notna(row['estimated_woba_using_speedangle']):
                    rv = float(row['estimated_woba_using_speedangle']) - base_re
                else:
                    # Estimate based on outcome
                    if 'single' in outcome or 'double' in outcome or 'triple' in outcome or 'home_run' in outcome:
                        rv = 0.1  # Positive value for hits
                    else:
                        rv = -0.05  # Out
                return rv
            elif 'ball' in outcome:
                # Walk increases run expectancy
                next_b = min(3, b + 1)
                if next_b == 4:
                    rv = 0.3  # Walk
                else:
                    next_count = f"{next_b}-{s}"
                    next_re = run_expectancy.get(next_count, 0.35)
                    rv = next_re - base_re
                return rv
            elif 'called_strike' in outcome:
                # Strike reduces run expectancy
                next_s = min(2, s + 1)
                if next_s == 3:
                    rv = -0.25  # Strikeout
                else:
                    next_count = f"{b}-{next_s}"
                    next_re = run_expectancy.get(next_count, 0.35)
                    rv = next_re - base_re
                return rv
            elif 'swinging_strike' in outcome:
                next_s = min(2, s + 1)
                if next_s == 3:
                    rv = -0.25  # Strikeout
                else:
                    next_count = f"{b}-{next_s}"
                    next_re = run_expectancy.get(next_count, 0.35)
                    rv = next_re - base_re
                return rv
            elif 'foul' in outcome:
                # Foul ball with 2 strikes has no effect, otherwise reduces RE slightly
                if s == 2:
                    rv = 0.0  # No change
                else:
                    next_s = min(2, s + 1)
                    next_count = f"{b}-{next_s}"
                    next_re = run_expectancy.get(next_count, 0.35)
                    rv = next_re - base_re
                return rv
            else:
                return 0.0
        
        df['run_value'] = df.apply(calculate_run_value, axis=1)
        
        # Calculate metrics by zone
        zone_data = []
        for zone_num in range(1, 10):
            zone_df = df[df['zone_num'] == zone_num]
            
            if len(zone_df) == 0:
                continue
            
            total_pitches = len(zone_df)
            swings = zone_df['is_swing'].sum()
            takes = zone_df['is_take'].sum()
            chases = zone_df['is_chase'].sum()
            in_zone = zone_df['is_in_zone'].sum()
            
            swing_rate = (swings / total_pitches * 100) if total_pitches > 0 else 0.0
            chase_rate = (chases / total_pitches * 100) if total_pitches > 0 else 0.0
            
            # Calculate average run value for swings and takes in this zone
            swing_rv = zone_df[zone_df['is_swing']]['run_value'].mean() if swings > 0 else 0.0
            take_rv = zone_df[zone_df['is_take']]['run_value'].mean() if takes > 0 else 0.0
            
            # Optimal decision: swing if swing_rv > take_rv
            is_optimal_swing = swing_rv > take_rv if (swings > 0 and takes > 0) else (swing_rv > 0 if swings > 0 else True)
            
            # Overall run value (weighted average)
            overall_rv = zone_df['run_value'].mean()
            
            # Decision quality: how often batter makes optimal decision
            optimal_decisions = 0
            if is_optimal_swing:
                optimal_decisions = swings
            else:
                optimal_decisions = takes
            
            decision_quality = (optimal_decisions / total_pitches * 100) if total_pitches > 0 else 0.0
            
            zone_data.append({
                'zone': int(zone_num),
                'total_pitches': int(total_pitches),
                'swings': int(swings),
                'takes': int(takes),
                'chases': int(chases),
                'in_zone': int(in_zone),
                'swing_rate': round(swing_rate, 1),
                'chase_rate': round(chase_rate, 1),
                'run_value': round(overall_rv, 4),
                'swing_run_value': round(swing_rv, 4),
                'take_run_value': round(take_rv, 4),
                'is_optimal_swing': bool(is_optimal_swing),
                'decision_quality': round(decision_quality, 1)
            })
        
        # Calculate overall metrics
        total_pitches = len(df)
        overall_swings = df['is_swing'].sum()
        overall_swing_rate = (overall_swings / total_pitches * 100) if total_pitches > 0 else 0.0
        
        zone_pitches = df[df['is_in_zone']]
        zone_swings = zone_pitches['is_swing'].sum() if len(zone_pitches) > 0 else 0
        zone_swing_rate = (zone_swings / len(zone_pitches) * 100) if len(zone_pitches) > 0 else 0.0
        
        chase_pitches = df[~df['is_in_zone']]
        chases = chase_pitches['is_swing'].sum() if len(chase_pitches) > 0 else 0
        chase_rate = (chases / len(chase_pitches) * 100) if len(chase_pitches) > 0 else 0.0
        
        take_rate_in_zone = ((len(zone_pitches) - zone_swings) / len(zone_pitches) * 100) if len(zone_pitches) > 0 else 0.0
        
        # Calculate optimal decision rate
        optimal_decisions_total = 0
        for zd in zone_data:
            if zd['is_optimal_swing']:
                optimal_decisions_total += zd['swings']
            else:
                optimal_decisions_total += zd['takes']
        
        optimal_decision_rate = (optimal_decisions_total / total_pitches * 100) if total_pitches > 0 else 0.0
        
        # Average run value
        avg_run_value = df['run_value'].mean() if len(df) > 0 else 0.0
        
        # Decision quality score (0-100)
        decision_quality_score = optimal_decision_rate
        
        # Quality breakdown
        quality_breakdown = {
            'optimal': optimal_decisions_total,
            'suboptimal': total_pitches - optimal_decisions_total,
            'poor': 0  # Could be enhanced with more sophisticated logic
        }
        
        # Get batter handedness
        batter_hand = 'R'
        if 'stand' in df.columns:
            stand_values = df['stand'].dropna()
            if len(stand_values) > 0:
                batter_hand = str(stand_values.mode().iloc[0]) if len(stand_values.mode()) > 0 else 'R'
        
        return jsonify({
            'batter': batter_name,
            'batter_hand': batter_hand,
            'season': season,
            'count': count,
            'total_pitches': total_pitches,
            'overall_swing_rate': round(overall_swing_rate, 1),
            'zone_swing_rate': round(zone_swing_rate, 1),
            'chase_rate': round(chase_rate, 1),
            'take_rate_in_zone': round(take_rate_in_zone, 1),
            'optimal_decision_rate': round(optimal_decision_rate, 1),
            'avg_run_value': round(avg_run_value, 4),
            'decision_quality_score': round(decision_quality_score, 1),
            'zone_data': zone_data,
            'quality_breakdown': quality_breakdown
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/zone-contact-rates', methods=['GET'])
def api_visuals_zone_contact_rates():
    """Get strike zone contact and swing rate data for a batter"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        batter_name = request.args.get('batter', '').strip()
        if not batter_name:
            return jsonify({"error": "Batter name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get batter ID
        try:
            batter_id = lookup_batter_id(batter_name)
        except Exception as e:
            return jsonify({"error": f"Could not find batter ID for '{batter_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            df = fetch_batter_statcast(batter_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {batter_name}",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
            
            if df.empty:
                return jsonify({
                    "error": f"No statcast data found for {batter_name} for season {season}",
                    "batter": batter_name,
                    "data": {}
                }), 200
        
        # Filter out rows without location data
        if 'plate_x' not in df.columns or 'plate_z' not in df.columns:
            return jsonify({
                "error": "Location data (plate_x, plate_z) not available",
                "batter": batter_name,
                "data": {}
            }), 200
        
        df = df[df['plate_x'].notna() & df['plate_z'].notna()]
        if df.empty:
            return jsonify({
                "error": "No valid location data found",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Calculate swing and contact indicators
        desc = df['description'].astype(str).str.lower()
        df['is_swing'] = desc.isin([
            'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 
            'missed_bunt', 'foul_bunt', 'hit_into_play'
        ])
        
        # Contact made (swing that resulted in contact, not a whiff)
        df['is_contact'] = desc.isin([
            'foul', 'foul_tip', 'foul_bunt', 'hit_into_play'
        ])
        
        # Whiff (swing and miss)
        df['is_whiff'] = desc.isin([
            'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'
        ])
        
        # Quality of contact metrics
        df['is_hard_hit'] = False
        if 'launch_speed' in df.columns:
            df['is_hard_hit'] = df['launch_speed'] >= 95
        
        df['is_barrel'] = False
        if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
            # Barrel definition: combination of launch speed and angle
            # Simplified: exit velo >= 98 and launch angle between 8-50 degrees
            # or exit velo >= 95 and launch angle 26-30 degrees
            launch_speed = df['launch_speed']
            launch_angle = df['launch_angle']
            df['is_barrel'] = (
                ((launch_speed >= 98) & (launch_angle >= 8) & (launch_angle <= 50)) |
                ((launch_speed >= 95) & (launch_angle >= 26) & (launch_angle <= 30))
            )
        
        # Create grid for strike zone (12x12 grid for better resolution)
        grid_size = 12
        x_min, x_max = -2.0, 2.0
        z_min, z_max = 0.5, 4.5
        
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        z_bins = np.linspace(z_min, z_max, grid_size + 1)
        
        # Bin the data
        df['x_bin'] = pd.cut(df['plate_x'], bins=x_bins, labels=False)
        df['z_bin'] = pd.cut(df['plate_z'], bins=z_bins, labels=False)
        
        # Calculate metrics for each grid cell
        grid_data = []
        for x_idx in range(grid_size):
            for z_idx in range(grid_size):
                cell_data = df[
                    (df['x_bin'] == x_idx) & 
                    (df['z_bin'] == z_idx)
                ]
                
                total_pitches = len(cell_data)
                
                if total_pitches == 0:
                    continue
                
                # Calculate rates
                swings = cell_data['is_swing'].sum()
                contacts = cell_data['is_contact'].sum()
                whiffs = cell_data['is_whiff'].sum()
                
                # Contact rate (contacts / swings)
                contact_rate = (contacts / swings * 100) if swings > 0 else None
                
                # Swing rate (swings / total pitches)
                swing_rate = (swings / total_pitches * 100) if total_pitches > 0 else 0.0
                
                # Whiff rate (whiffs / swings)
                whiff_rate = (whiffs / swings * 100) if swings > 0 else None
                
                # Quality of contact (only for batted balls)
                batted_balls = cell_data[cell_data['is_contact']]
                hard_hit_rate = None
                barrel_rate = None
                avg_exit_velo = None
                avg_launch_angle = None
                
                if len(batted_balls) > 0:
                    hard_hits = batted_balls['is_hard_hit'].sum()
                    hard_hit_rate = (hard_hits / len(batted_balls) * 100) if len(batted_balls) > 0 else None
                    
                    barrels = batted_balls['is_barrel'].sum()
                    barrel_rate = (barrels / len(batted_balls) * 100) if len(batted_balls) > 0 else None
                    
                    if 'launch_speed' in batted_balls.columns:
                        ev_values = batted_balls['launch_speed'].dropna()
                        if len(ev_values) > 0:
                            avg_exit_velo = float(ev_values.mean())
                    
                    if 'launch_angle' in batted_balls.columns:
                        la_values = batted_balls['launch_angle'].dropna()
                        if len(la_values) > 0:
                            avg_launch_angle = float(la_values.mean())
                
                # Calculate center coordinates of the cell
                x_center = (x_bins[x_idx] + x_bins[x_idx + 1]) / 2
                z_center = (z_bins[z_idx] + z_bins[z_idx + 1]) / 2
                
                grid_data.append({
                    'x': x_idx,
                    'y': grid_size - 1 - z_idx,  # Flip y-axis for display
                    'x_center': float(x_center),
                    'z_center': float(z_center),
                    'total_pitches': total_pitches,
                    'swings': int(swings),
                    'contacts': int(contacts),
                    'whiffs': int(whiffs),
                    'contact_rate': contact_rate,
                    'swing_rate': swing_rate,
                    'whiff_rate': whiff_rate,
                    'hard_hit_rate': hard_hit_rate,
                    'barrel_rate': barrel_rate,
                    'avg_exit_velo': avg_exit_velo,
                    'avg_launch_angle': avg_launch_angle
                })
        
        # Filter out cells with no data
        grid_data = [cell for cell in grid_data if cell['total_pitches'] > 0]
        
        if not grid_data:
            return jsonify({
                "error": f"No contact rate data available for {batter_name}",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Get batter handedness
        batter_hand = 'R'  # Default
        if 'stand' in df.columns:
            stand_values = df['stand'].dropna()
            if len(stand_values) > 0:
                batter_hand = str(stand_values.mode().iloc[0]) if len(stand_values.mode()) > 0 else 'R'
        
        return jsonify({
            'batter': batter_name,
            'batter_hand': batter_hand,
            'season': season,
            'total_pitches': len(df),
            'grid': grid_data,
            'grid_size': grid_size,
            'x_range': [float(x_min), float(x_max)],
            'z_range': [float(z_min), float(z_max)]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/plate-discipline-matrix', methods=['GET'])
def api_visuals_plate_discipline_matrix():
    """Get plate discipline matrix data: swing rates, chase rates, and contact rates by pitch type and location"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        batter_name = request.args.get('batter', '').strip()
        if not batter_name:
            return jsonify({"error": "Batter name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get batter ID
        try:
            batter_id = lookup_batter_id(batter_name)
        except Exception as e:
            return jsonify({"error": f"Could not find batter ID for '{batter_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
            filter_by_season = True
        else:
            current_year = datetime.now().year
            start_date = "2008-03-01"
            end_date = f"{current_year}-11-30"
            filter_by_season = False
        
        # Fetch statcast data
        try:
            df = fetch_batter_statcast(batter_id, start_date, end_date)
        except Exception as e:
            return jsonify({"error": f"Error fetching statcast data: {str(e)}"}), 500
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data found for {batter_name}",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Filter by season if specified
        if filter_by_season:
            year_column = None
            for col_name in ['game_year', 'year', 'Year', 'season']:
                if col_name in df.columns:
                    year_column = col_name
                    break
            
            if year_column:
                if not pd.api.types.is_integer_dtype(df[year_column]):
                    df[year_column] = pd.to_numeric(df[year_column], errors='coerce')
                df = df[
                    (df[year_column].notna()) & 
                    (df[year_column] == int(season))
                ]
            else:
                date_column = None
                for col_name in ['game_date', 'gameDate', 'date', 'Date', 'game_day']:
                    if col_name in df.columns:
                        date_column = col_name
                        break
                
                if date_column:
                    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                    
                    season_start = pd.to_datetime(f"{season}-03-01")
                    season_end = pd.to_datetime(f"{season}-11-30")
                    
                    df = df[df[date_column].notna()]
                    df = df[
                        (df[date_column] >= season_start) & 
                        (df[date_column] <= season_end)
                    ]
            
            if df.empty:
                return jsonify({
                    "error": f"No statcast data found for {batter_name} for season {season}",
                    "batter": batter_name,
                    "data": {}
                }), 200
        
        # Filter out rows without necessary data
        df = df.dropna(subset=['pitch_type'])
        if df.empty:
            return jsonify({
                "error": "No valid pitch data available",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Calculate swing, contact, and zone indicators
        desc = df['description'].astype(str).str.lower()
        df['is_swing'] = desc.isin([
            'foul', 'foul_tip', 'swinging_strike', 'swinging_strike_blocked', 
            'missed_bunt', 'foul_bunt', 'hit_into_play'
        ])
        
        # Contact made (swing that resulted in contact, not a whiff)
        df['is_contact'] = desc.isin([
            'foul', 'foul_tip', 'foul_bunt', 'hit_into_play'
        ])
        
        # Determine if pitch is in zone (zone 1-9 are in zone, 11-14 are outside, NaN needs plate_x/plate_z check)
        df['is_in_zone'] = False
        if 'zone' in df.columns:
            # Zone 1-9 are in the strike zone
            df['is_in_zone'] = df['zone'].between(1, 9, inclusive='both')
            # For pitches without zone data, try to infer from plate_x and plate_z
            # Standard strike zone: plate_x: -0.855 to 0.855, plate_z: ~1.5 to ~3.5 (varies by batter)
            if 'plate_x' in df.columns and 'plate_z' in df.columns:
                zone_na_mask = df['zone'].isna()
                if zone_na_mask.any():
                    # Approximate zone boundaries (can be refined)
                    df.loc[zone_na_mask, 'is_in_zone'] = (
                        (df.loc[zone_na_mask, 'plate_x'].abs() <= 0.855) &
                        (df.loc[zone_na_mask, 'plate_z'] >= 1.5) &
                        (df.loc[zone_na_mask, 'plate_z'] <= 3.5)
                    )
        elif 'plate_x' in df.columns and 'plate_z' in df.columns:
            # Fallback: estimate zone from plate_x and plate_z
            df['is_in_zone'] = (
                (df['plate_x'].abs() <= 0.855) &
                (df['plate_z'] >= 1.5) &
                (df['plate_z'] <= 3.5)
            )
        
        # Chase = swing on pitch outside the zone
        df['is_chase'] = df['is_swing'] & ~df['is_in_zone']
        
        # Get pitch types
        pitch_types = df['pitch_type'].dropna().unique()
        if len(pitch_types) == 0:
            return jsonify({
                "error": "No pitch type data available",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Define location zones (simplified: In Zone vs Out of Zone)
        # Could expand to 9-zone grid later
        location_zones = ['In Zone', 'Out of Zone']
        
        # Calculate metrics by pitch type and location
        matrix_data = []
        
        for pitch_type in pitch_types:
            df_pitch = df[df['pitch_type'] == pitch_type]
            
            for zone_type in location_zones:
                if zone_type == 'In Zone':
                    df_zone = df_pitch[df_pitch['is_in_zone']]
                else:
                    df_zone = df_pitch[~df_pitch['is_in_zone']]
                
                if len(df_zone) == 0:
                    continue
                
                total_pitches = len(df_zone)
                swings = df_zone['is_swing'].sum()
                chases = df_zone['is_chase'].sum()
                contacts = df_zone['is_contact'].sum()
                
                # Calculate rates
                swing_rate = (swings / total_pitches * 100) if total_pitches > 0 else 0.0
                chase_rate = (chases / total_pitches * 100) if total_pitches > 0 else 0.0
                contact_rate = (contacts / swings * 100) if swings > 0 else None
                
                matrix_data.append({
                    'pitch_type': str(pitch_type),
                    'location': zone_type,
                    'total_pitches': int(total_pitches),
                    'swings': int(swings),
                    'chases': int(chases),
                    'contacts': int(contacts),
                    'swing_rate': round(swing_rate, 1),
                    'chase_rate': round(chase_rate, 1),
                    'contact_rate': round(contact_rate, 1) if contact_rate is not None else None
                })
        
        if not matrix_data:
            return jsonify({
                "error": f"No plate discipline data available for {batter_name}",
                "batter": batter_name,
                "data": {}
            }), 200
        
        # Get batter handedness
        batter_hand = 'R'  # Default
        if 'stand' in df.columns:
            stand_values = df['stand'].dropna()
            if len(stand_values) > 0:
                batter_hand = str(stand_values.mode().iloc[0]) if len(stand_values.mode()) > 0 else 'R'
        
        return jsonify({
            'batter': batter_name,
            'batter_hand': batter_hand,
            'season': season,
            'total_pitches': len(df),
            'matrix': matrix_data,
            'pitch_types': [str(pt) for pt in pitch_types],
            'location_zones': location_zones
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/visuals/expected-stats-comparison', methods=['GET'])
def api_visuals_expected_stats_comparison():
    """Get expected stats comparison data: xwOBA, xBA, xSLG, xISO vs actual with confidence intervals"""
    if not csv_loader:
        return jsonify({"error": "CSV data loader not available"}), 500
    
    try:
        batter_name = request.args.get('player', '').strip()
        if not batter_name:
            return jsonify({"error": "Player name is required"}), 400
        
        season = request.args.get('season', '').strip() or None
        if season:
            try:
                season = int(season)
            except ValueError:
                season = None
        
        pitch_type = request.args.get('pitch_type', '').strip() or None
        pitcher_hand = request.args.get('pitcher_hand', '').strip() or None
        count = request.args.get('count', '').strip() or None
        
        # Import statcast functions
        sys.path.insert(0, str(ROOT_DIR / "src"))
        from scrape_savant import fetch_batter_statcast, lookup_batter_id
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        # Get batter ID
        try:
            batter_id = lookup_batter_id(batter_name)
        except Exception as e:
            return jsonify({"error": f"Could not find player ID for '{batter_name}': {str(e)}"}), 404
        
        # Calculate date range
        if season:
            start_date = f"{season}-03-01"
            end_date = f"{season}-11-30"
        else:
            # Get last 3 years of data
            current_year = datetime.now().year
            start_date = f"{current_year - 3}-03-01"
            end_date = f"{current_year}-11-30"
        
        # Fetch statcast data
        df = fetch_batter_statcast(batter_id, start_date, end_date)
        
        if df is None or df.empty:
            return jsonify({
                "error": f"No statcast data available for {batter_name}",
                "player": batter_name,
                "stats": {}
            }), 200
        
        # Filter by season if specified
        if season and 'game_year' in df.columns:
            df = df[df['game_year'] == season]
        
        # Apply filters
        if pitch_type:
            df = df[df['pitch_type'] == pitch_type]
        if pitcher_hand and 'p_throws' in df.columns:
            df = df[df['p_throws'] == pitcher_hand]
        if count and 'balls' in df.columns and 'strikes' in df.columns:
            balls, strikes = map(int, count.split('-'))
            df = df[(df['balls'] == balls) & (df['strikes'] == strikes)]
        
        if df.empty:
            return jsonify({
                "error": f"No data available for {batter_name} with selected filters",
                "player": batter_name,
                "stats": {}
            }), 200
        
        # Calculate expected and actual stats
        stats_result = calculate_expected_stats_comparison(df, batter_name)
        
        return jsonify(stats_result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def calculate_expected_stats_comparison(df, batter_name):
    """Calculate expected vs actual stats with confidence intervals"""
    import pandas as pd
    import numpy as np
    
    # Filter to only batted ball events
    batted_balls = df[df['type'] == 'X'].copy()
    
    if batted_balls.empty:
        return {
            "player": batter_name,
            "summary": {
                "total_pa": len(df),
                "total_batted_balls": 0
            },
            "stats": {}
        }
    
    # Calculate plate appearances
    total_pa = len(df[df['type'].isin(['X', 'S', 'B', 'HBP', 'E', 'D'])])
    
    # Expected stats fields from statcast
    expected_fields = {
        'xwoba': 'estimated_woba_using_speedangle',
        'xba': 'estimated_ba_using_speedangle',
        'xslg': 'estimated_slg_using_speedangle',
    }
    
    # Calculate xISO (expected ISO = xSLG - xBA)
    batted_balls['estimated_iso'] = (
        batted_balls['estimated_slg_using_speedangle'].fillna(0) - 
        batted_balls['estimated_ba_using_speedangle'].fillna(0)
    )
    
    # Calculate actual stats
    # Actual BA = hits / at bats (batted balls that are hits)
    hits = batted_balls[batted_balls['events'].isin(['single', 'double', 'triple', 'home_run'])]
    actual_ba = len(hits) / len(batted_balls) if len(batted_balls) > 0 else 0
    
    # Actual SLG = total bases / at bats
    total_bases = 0
    for event in batted_balls['events']:
        if event == 'single':
            total_bases += 1
        elif event == 'double':
            total_bases += 2
        elif event == 'triple':
            total_bases += 3
        elif event == 'home_run':
            total_bases += 4
    actual_slg = total_bases / len(batted_balls) if len(batted_balls) > 0 else 0
    
    # Actual ISO = SLG - BA
    actual_iso = actual_slg - actual_ba
    
    # Actual wOBA (simplified - using statcast woba_value if available)
    # Use woba_value from all plate appearances, not just batted balls
    if 'woba_value' in df.columns:
        woba_pa = df['woba_value'].dropna()
        actual_woba = woba_pa.mean() if len(woba_pa) > 0 else 0
    else:
        # Approximate wOBA from events (using full PA dataset)
        woba_values = {
            'single': 0.9,
            'double': 1.25,
            'triple': 1.6,
            'home_run': 2.0,
            'walk': 0.7,
            'strikeout': 0.0,
            'out': 0.0
        }
        df['woba_approx'] = df['events'].map(woba_values).fillna(0)
        woba_pa = df['woba_approx'].dropna()
        actual_woba = woba_pa.mean() if len(woba_pa) > 0 else 0
    
    # Calculate expected stats (mean of expected values)
    expected_woba = batted_balls['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in batted_balls.columns else 0
    expected_ba = batted_balls['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in batted_balls.columns else 0
    expected_slg = batted_balls['estimated_slg_using_speedangle'].mean() if 'estimated_slg_using_speedangle' in batted_balls.columns else 0
    expected_iso = batted_balls['estimated_iso'].mean()
    
    # Calculate confidence intervals (95% CI using standard error)
    # For large samples, CI = mean ± 1.96 * SE
    def calculate_ci(values, mean_val):
        if len(values) == 0:
            return mean_val, mean_val
        std_err = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ci_margin = 1.96 * std_err
        return max(0, mean_val - ci_margin), mean_val + ci_margin
    
    woba_values = batted_balls['estimated_woba_using_speedangle'].dropna()
    ba_values = batted_balls['estimated_ba_using_speedangle'].dropna()
    slg_values = batted_balls['estimated_slg_using_speedangle'].dropna()
    iso_values = batted_balls['estimated_iso'].dropna()
    
    woba_ci_lower, woba_ci_upper = calculate_ci(woba_values, expected_woba)
    ba_ci_lower, ba_ci_upper = calculate_ci(ba_values, expected_ba)
    slg_ci_lower, slg_ci_upper = calculate_ci(slg_values, expected_slg)
    iso_ci_lower, iso_ci_upper = calculate_ci(iso_values, expected_iso)
    
    # Create time series data (group by month or game date)
    # For xwOBA, we need full dataset (all PA), for others use batted balls only
    time_series_data = {}
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df = df.dropna(subset=['game_date'])
        df['month'] = df['game_date'].dt.to_period('M')
        
        batted_balls_ts = batted_balls.copy()
        if 'game_date' in batted_balls_ts.columns:
            batted_balls_ts['game_date'] = pd.to_datetime(batted_balls_ts['game_date'], errors='coerce')
            batted_balls_ts = batted_balls_ts.dropna(subset=['game_date'])
            batted_balls_ts['month'] = batted_balls_ts['game_date'].dt.to_period('M')
        
        for stat_key in ['xwoba', 'xba', 'xslg', 'xiso']:
            if stat_key == 'xwoba':
                # For xwOBA, use full PA dataset
                expected_col = 'estimated_woba_using_speedangle'
                actual_col = 'woba_value'
                group_df = batted_balls_ts  # Expected comes from batted balls
                full_df = df  # Actual comes from all PA
            elif stat_key == 'xba':
                expected_col = 'estimated_ba_using_speedangle'
                actual_col = None  # Calculate from events
                group_df = batted_balls_ts
                full_df = batted_balls_ts
            elif stat_key == 'xslg':
                expected_col = 'estimated_slg_using_speedangle'
                actual_col = None  # Calculate from events
                group_df = batted_balls_ts
                full_df = batted_balls_ts
            else:  # xiso
                expected_col = 'estimated_iso'
                actual_col = None
                group_df = batted_balls_ts
                full_df = batted_balls_ts
            
            monthly_data = []
            for month, group in group_df.groupby('month'):
                if expected_col in group.columns:
                    exp_mean = group[expected_col].mean()
                else:
                    exp_mean = 0
                
                # Calculate actual for this month
                if stat_key == 'xwoba':
                    # For xwOBA, use full PA dataset for actual
                    month_full = full_df[full_df['month'] == month]
                    if actual_col and actual_col in month_full.columns:
                        woba_values_month = month_full[actual_col].dropna()
                        act_mean = woba_values_month.mean() if len(woba_values_month) > 0 else 0
                    else:
                        # Approximate from events
                        woba_values_dict = {
                            'single': 0.9, 'double': 1.25, 'triple': 1.6,
                            'home_run': 2.0, 'walk': 0.7, 'strikeout': 0.0, 'out': 0.0
                        }
                        month_full['woba_approx'] = month_full['events'].map(woba_values_dict).fillna(0)
                        act_mean = month_full['woba_approx'].mean() if len(month_full) > 0 else 0
                elif stat_key == 'xba':
                    hits_month = group[group['events'].isin(['single', 'double', 'triple', 'home_run'])]
                    act_mean = len(hits_month) / len(group) if len(group) > 0 else 0
                elif stat_key == 'xslg':
                    total_bases_month = 0
                    for event in group['events']:
                        if event == 'single':
                            total_bases_month += 1
                        elif event == 'double':
                            total_bases_month += 2
                        elif event == 'triple':
                            total_bases_month += 3
                        elif event == 'home_run':
                            total_bases_month += 4
                    act_mean = total_bases_month / len(group) if len(group) > 0 else 0
                elif stat_key == 'xiso':
                    hits_month = group[group['events'].isin(['single', 'double', 'triple', 'home_run'])]
                    ba_month = len(hits_month) / len(group) if len(group) > 0 else 0
                    total_bases_month = 0
                    for event in group['events']:
                        if event == 'single':
                            total_bases_month += 1
                        elif event == 'double':
                            total_bases_month += 2
                        elif event == 'triple':
                            total_bases_month += 3
                        elif event == 'home_run':
                            total_bases_month += 4
                    slg_month = total_bases_month / len(group) if len(group) > 0 else 0
                    act_mean = slg_month - ba_month
                else:
                    act_mean = 0
                
                exp_values = group[expected_col].dropna() if expected_col in group.columns else pd.Series([])
                if len(exp_values) > 1:
                    std_err = np.std(exp_values, ddof=1) / np.sqrt(len(exp_values))
                    ci_margin = 1.96 * std_err
                    ci_lower = max(0, exp_mean - ci_margin)
                    ci_upper = exp_mean + ci_margin
                else:
                    ci_lower = exp_mean
                    ci_upper = exp_mean
                
                monthly_data.append({
                    'label': str(month),
                    'expected': float(exp_mean) if not pd.isna(exp_mean) else 0,
                    'actual': float(act_mean) if not pd.isna(act_mean) else 0,
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper)
                })
            
            time_series_data[stat_key] = monthly_data
    else:
        # No time series data available
        for stat_key in ['xwoba', 'xba', 'xslg', 'xiso']:
            time_series_data[stat_key] = []
    
    return {
        "player": batter_name,
        "summary": {
            "total_pa": int(total_pa),
            "total_batted_balls": int(len(batted_balls))
        },
        "stats": {
            "xwoba": {
                "expected": float(expected_woba) if not pd.isna(expected_woba) else 0,
                "actual": float(actual_woba) if not pd.isna(actual_woba) else 0,
                "ci_lower": float(woba_ci_lower),
                "ci_upper": float(woba_ci_upper),
                "time_series": time_series_data.get('xwoba', [])
            },
            "xba": {
                "expected": float(expected_ba) if not pd.isna(expected_ba) else 0,
                "actual": float(actual_ba) if not pd.isna(actual_ba) else 0,
                "ci_lower": float(ba_ci_lower),
                "ci_upper": float(ba_ci_upper),
                "time_series": time_series_data.get('xba', [])
            },
            "xslg": {
                "expected": float(expected_slg) if not pd.isna(expected_slg) else 0,
                "actual": float(actual_slg) if not pd.isna(actual_slg) else 0,
                "ci_lower": float(slg_ci_lower),
                "ci_upper": float(slg_ci_upper),
                "time_series": time_series_data.get('xslg', [])
            },
            "xiso": {
                "expected": float(expected_iso) if not pd.isna(expected_iso) else 0,
                "actual": float(actual_iso) if not pd.isna(actual_iso) else 0,
                "ci_lower": float(iso_ci_lower),
                "ci_upper": float(iso_ci_upper),
                "time_series": time_series_data.get('xiso', [])
            }
        }
    }

@app.route('/api/pitcher/<pitcher_name>/seasons', methods=['GET'])
def api_pitcher_seasons(pitcher_name):
    """Get available seasons for a specific pitcher"""
    try:
        from urllib.parse import unquote
        pitcher_name = unquote(pitcher_name)
        
        # Use CSV seasons endpoint logic
        if csv_loader:
            try:
                player_data = csv_loader.get_player_data(pitcher_name)
                if player_data and player_data.get('fangraphs'):
                    seasons = set()
                    for row in player_data['fangraphs']:
                        if 'Season' in row and row['Season'] is not None:
                            try:
                                seasons.add(int(row['Season']))
                            except (ValueError, TypeError):
                                pass
                    
                    if seasons:
                        seasons_str = sorted([str(s) for s in seasons], reverse=True)
                        return jsonify({
                            "pitcher": pitcher_name,
                            "seasons": seasons_str
                        })
            except Exception:
                pass
        
        # Fallback: return common seasons
        from datetime import datetime
        current_year = datetime.now().year
        seasons = [str(year) for year in range(2015, current_year + 1)]
        
        return jsonify({
            "pitcher": pitcher_name,
            "seasons": seasons
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/game-analysis')
def game_analysis():
    """Game analysis page"""
    return render_template('game_analysis.html')

@app.route('/reports-library')
def reports_library():
    """Reports library page"""
    return render_template('reports_library.html')


@app.route('/workouts')
@login_required
def workouts():
    """Workouts page with admin/player modes."""
    viewer_user = getattr(g, "user", None)
    current_user_id = viewer_user.get("id") if viewer_user else None
    workout_document = None
    if PlayerDB and current_user_id:
        db = None
        try:
            db = PlayerDB()
            latest = db.get_latest_player_document_by_category(current_user_id, WORKOUT_CATEGORY)
            if latest:
                workout_document = _format_player_document(latest)
        except Exception as exc:
            print(f"Warning: unable to load workout document: {exc}")
        finally:
            if db:
                db.close()

    first_name = (viewer_user or {}).get("first_name") or ""
    last_name = (viewer_user or {}).get("last_name") or ""
    name_parts = [part for part in (first_name.strip(), last_name.strip()) if part]
    initial_player_label = " ".join(name_parts) if name_parts else ((viewer_user or {}).get("email") or "Player")

    is_admin = bool(session.get("is_admin"))
    initial_player_id = None if is_admin else current_user_id
    if is_admin:
        workout_document = None

    return render_template(
        'workouts.html',
        workout_document=workout_document,
        csrf_token=generate_csrf_token(),
        workout_category=WORKOUT_CATEGORY,
        initial_player_id=initial_player_id,
        initial_player_label=initial_player_label if initial_player_id else "",
        current_user_id=current_user_id,
        current_user_label=initial_player_label,
    )


@app.route('/nutrition')
@login_required
def nutrition():
    """Nutrition placeholder page"""
    return render_template('nutrition.html')


@app.route('/journaling', methods=['GET', 'POST'])
@login_required
def journaling():
    """Personal journaling workspace for players."""
    viewer_user = getattr(g, "user", None)
    if not viewer_user:
        abort(403)

    today_iso = datetime.now().strftime("%Y-%m-%d")
    selected_visibility = _normalize_journal_visibility(
        request.values.get("visibility") if request.method == "GET" else request.form.get("visibility"),
        default="private",
    )
    selected_date_raw = request.values.get("date") if request.method == "GET" else request.form.get("entry_date")
    if not selected_date_raw:
        selected_date_raw = today_iso

    entry_errors: List[str] = []
    just_saved = False

    if request.method == "POST":
        if not validate_csrf(request.form.get("csrf_token")):
            abort(400, description="Invalid CSRF token")

        entry_date = (request.form.get("entry_date") or "").strip()
        visibility_choice = _normalize_journal_visibility(request.form.get("visibility"), default="private")
        title = (request.form.get("title") or "").strip()
        body = request.form.get("body") or ""

        if not entry_date:
            entry_errors.append("Entry date is required.")

        try:
            datetime.strptime(entry_date, "%Y-%m-%d")
        except ValueError:
            entry_errors.append("Entry date must be in YYYY-MM-DD format.")

        if PlayerDB is None:
            entry_errors.append("Database is unavailable. Please try again later.")

        if not entry_errors and PlayerDB:
            db = None
            try:
                db = PlayerDB()
                db.upsert_journal_entry(
                    user_id=viewer_user["id"],
                    entry_date=entry_date,
                    visibility=visibility_choice,
                    title=title,
                    body=body,
                )
                flash("Journal entry saved.", "success")
                just_saved = True
                return redirect(url_for(
                    "journaling",
                    date=entry_date,
                    visibility=visibility_choice,
                ))
            except ValueError as exc:
                entry_errors.append(str(exc))
            except Exception as exc:
                print(f"Warning: failed to save journal entry: {exc}")
                entry_errors.append("An unexpected error occurred while saving.")
            finally:
                if db:
                    db.close()

        selected_date_raw = entry_date or selected_date_raw
        selected_visibility = visibility_choice

    timeline_entries: List[Dict[str, Any]] = []
    current_entry: Optional[Dict[str, Any]] = None

    if PlayerDB:
        db = None
        try:
            db = PlayerDB()
            entries = db.list_journal_entries(
                user_id=viewer_user["id"],
                limit=MAX_JOURNAL_TIMELINE_ENTRIES,
            )
            timeline_entries = _prepare_journal_timeline(entries)

            # Ensure the selected date corresponds to an existing entry if possible
            known_dates = {item["date"] for item in timeline_entries}
            if selected_date_raw not in known_dates and timeline_entries:
                selected_date_raw = timeline_entries[0]["date"]

            current_entry = db.get_journal_entry(
                user_id=viewer_user["id"],
                entry_date=selected_date_raw,
                visibility=selected_visibility,
            )
            current_entry = _augment_journal_entry(current_entry)
        except Exception as exc:
            print(f"Warning: unable to load journal entries: {exc}")
            timeline_entries = []
            current_entry = None
        finally:
            if db:
                db.close()
    else:
        flash("Journal features are temporarily unavailable.", "warning")

    return render_template(
        'journaling.html',
        selected_date=selected_date_raw,
        selected_visibility=selected_visibility,
        entry=current_entry,
        timeline_entries=timeline_entries,
        entry_errors=entry_errors,
        just_saved=just_saved,
        journal_visibility_options=JOURNAL_VISIBILITY_OPTIONS,
        today=today_iso,
        today_display=_format_journal_date(today_iso),
        is_admin_view=False,
        target_user=viewer_user,
        selected_date_display=_format_journal_date(selected_date_raw),
    )


@app.route('/journaling/admin')
@login_required
def journaling_admin():
    """Admin view of public journal entries."""
    if not session.get("is_admin"):
        abort(403)

    viewer_user = getattr(g, "user", None)
    today_iso = datetime.now().strftime("%Y-%m-%d")
    selected_user_id = request.args.get("user_id", type=int)
    selected_date_raw = request.args.get("date")

    user_options: List[Dict[str, Any]] = []
    target_user: Optional[Dict[str, Any]] = None
    timeline_entries: List[Dict[str, Any]] = []
    current_entry: Optional[Dict[str, Any]] = None

    def _format_user_label(record: Dict[str, Any]) -> str:
        first = (record.get("first_name") or "").strip()
        last = (record.get("last_name") or "").strip()
        if first or last:
            return f"{first} {last}".strip()
        return (record.get("email") or f"User #{record.get('id')}").strip()

    if PlayerDB:
        db = None
        try:
            db = PlayerDB()
            user_records = db.list_users()
            user_options = [
                {"id": record["id"], "label": _format_user_label(record)}
                for record in user_records
            ]
            if selected_user_id:
                target_user = db.get_user_by_id(selected_user_id)
            if not target_user and user_records:
                target_user = user_records[0]

            if target_user:
                entries = db.list_journal_entries(
                    user_id=target_user["id"],
                    visibility="public",
                    limit=MAX_JOURNAL_TIMELINE_ENTRIES,
                )
                timeline_entries = _prepare_journal_timeline(entries)
                known_dates = {item["date"] for item in timeline_entries}
                if not selected_date_raw and timeline_entries:
                    selected_date_raw = timeline_entries[0]["date"]
                elif selected_date_raw not in known_dates:
                    selected_date_raw = timeline_entries[0]["date"] if timeline_entries else selected_date_raw

                if selected_date_raw:
                    current_entry = db.get_journal_entry(
                        user_id=target_user["id"],
                        entry_date=selected_date_raw,
                        visibility="public",
                    )
                    current_entry = _augment_journal_entry(current_entry)
        except Exception as exc:
            print(f"Warning: admin journal view error: {exc}")
            timeline_entries = []
            current_entry = None
        finally:
            if db:
                db.close()
    else:
        flash("Journal features are temporarily unavailable.", "warning")

    return render_template(
        'journaling.html',
        selected_date=selected_date_raw,
        selected_visibility="public",
        entry=current_entry,
        timeline_entries=timeline_entries,
        entry_errors=[],
        just_saved=False,
        journal_visibility_options=JOURNAL_VISIBILITY_OPTIONS,
        today=today_iso,
        today_display=_format_journal_date(today_iso),
        is_admin_view=True,
        target_user=target_user,
        admin_user_options=user_options,
        viewer_user=viewer_user,
        selected_date_display=_format_journal_date(selected_date_raw),
    )


@app.route('/journaling/delete', methods=['POST'])
@login_required
def delete_journal_entry_route():
    """Delete a journal entry belonging to the current user."""
    viewer_user = getattr(g, "user", None)
    if not viewer_user:
        abort(403)

    if not validate_csrf(request.form.get("csrf_token")):
        abort(400, description="Invalid CSRF token")

    entry_id = request.form.get("entry_id", type=int)
    entry_date = (request.form.get("entry_date") or "").strip()
    visibility = _normalize_journal_visibility(request.form.get("visibility"), default="private")

    if not entry_date:
        entry_date = datetime.now().strftime("%Y-%m-%d")

    if entry_id is None or PlayerDB is None:
        flash("Unable to delete journal entry.", "error")
        return redirect(url_for("journaling", date=entry_date, visibility=visibility))

    success = False
    db = None
    try:
        db = PlayerDB()
        success = db.delete_journal_entry(entry_id, viewer_user["id"])
    except Exception as exc:
        print(f"Warning: failed to delete journal entry {entry_id}: {exc}")
        success = False
    finally:
        if db:
            db.close()

    if success:
        flash("Journal entry deleted.", "success")
    else:
        flash("Unable to delete journal entry.", "error")

    return redirect(url_for("journaling", date=entry_date, visibility=visibility))


@app.route('/profile-settings', methods=['GET', 'POST'])
@login_required
def profile_settings():
    """Allow authenticated users to manage their personal preferences."""
    if not PlayerDB:
        flash("Profile settings are temporarily unavailable.", "error")
        return redirect(url_for("home"))

    viewer = g.user or {}
    notification_prefs_raw = viewer.get("notification_preferences") or "{}"
    if isinstance(notification_prefs_raw, str):
        try:
            notification_prefs = json.loads(notification_prefs_raw or "{}")
            if not isinstance(notification_prefs, dict):
                notification_prefs = {}
        except Exception:
            notification_prefs = {}
    elif isinstance(notification_prefs_raw, dict):
        notification_prefs = notification_prefs_raw
    else:
        notification_prefs = {}

    if request.method == "POST":
        if not validate_csrf(request.form.get("csrf_token")):
            flash("Invalid form submission. Please try again.", "error")
            return redirect(url_for("profile_settings"))

        form_name = (request.form.get("form_name") or "basic-info").strip().lower()
        db = None
        try:
            db = PlayerDB()
            if form_name == "basic-info":
                updates = {
                    "first_name": clean_str(request.form.get("first_name")),
                    "last_name": clean_str(request.form.get("last_name")),
                    "pronouns": clean_str(request.form.get("pronouns")),
                    "job_title": clean_str(request.form.get("job_title")),
                    "phone": clean_str(request.form.get("phone")),
                    "timezone": clean_str(request.form.get("timezone")),
                    "bio": (request.form.get("bio") or "").strip(),
                }
                # Normalize optional fields
                for key, value in list(updates.items()):
                    if value is not None:
                        value = value.strip()
                        updates[key] = value or None
                success = db.update_user_profile(viewer["id"], **updates)
                if success:
                    flash("Profile details updated.", "success")
                else:
                    flash("No profile changes detected.", "info")

            elif form_name == "appearance":
                theme = (request.form.get("theme_preference") or "").strip().lower()
                if theme not in {"light", "dark"}:
                    flash("Unknown theme selection.", "error")
                else:
                    db.update_user_profile(viewer["id"], theme_preference=theme)
                    session["theme_preference"] = theme
                    flash("Appearance preference saved.", "success")

            elif form_name == "notifications":
                prefs_payload = {
                    "weekly_digest": bool(request.form.get("weekly_digest")),
                    "reports_ready": bool(request.form.get("reports_ready")),
                    "system_updates": bool(request.form.get("system_updates")),
                }
                db.update_user_profile(
                    viewer["id"],
                    notification_preferences=json.dumps(prefs_payload)
                )
                flash("Notification preferences updated.", "success")

            elif form_name == "change-password":
                current_pw = request.form.get("current_password") or ""
                new_pw = request.form.get("new_password") or ""
                confirm_pw = request.form.get("confirm_password") or ""
                stored_hash = viewer.get("password_hash") or ""

                if not check_password_hash(stored_hash, current_pw):
                    flash("Current password is incorrect.", "error")
                elif new_pw != confirm_pw:
                    flash("New passwords do not match.", "error")
                elif len(new_pw) < 12:
                    flash("Password must be at least 12 characters.", "error")
                else:
                    db.update_user_password(viewer["id"], generate_password_hash(new_pw))
                    flash("Password updated.", "success")

            elif form_name == "avatar":
                file = request.files.get("profile_image")
                if not file or not file.filename:
                    flash("Please select an image to upload.", "error")
                else:
                    filename = secure_filename(file.filename)
                    extension = Path(filename).suffix.lower()
                    if extension not in ALLOWED_PROFILE_IMAGE_EXT:
                        flash("Unsupported image type.", "error")
                    else:
                        data = file.read()
                        if len(data) > MAX_PROFILE_IMAGE_BYTES:
                            flash("Image exceeds 5 MB limit.", "error")
                        else:
                            detected_type = detect_image_type(data)
                            if detected_type not in ALLOWED_PROFILE_IMAGE_TYPES:
                                flash("Uploaded file is not a valid image.", "error")
                            else:
                                unique_name = f"user-{viewer['id']}-{uuid.uuid4().hex}{extension}"
                                destination = PROFILE_UPLOAD_DIR / unique_name
                                with destination.open("wb") as fh:
                                    fh.write(data)

                                # Remove previous avatar if one exists
                                previous = viewer.get("profile_image_path")
                                if previous:
                                    old_path = ROOT_DIR / "static" / previous
                                    try:
                                        old_path.unlink()
                                    except FileNotFoundError:
                                        pass
                                rel_path = f"uploads/profile_photos/{unique_name}"
                                db.update_user_profile(viewer["id"], profile_image_path=rel_path)
                                flash("Profile photo updated.", "success")
            else:
                flash("Unknown form submission.", "error")
        except Exception as exc:
            flash(f"Unable to update profile: {exc}", "error")
        finally:
            if db:
                db.close()

        return redirect(url_for("profile_settings"))

    common_timezones = [
        "US/Pacific", "US/Mountain", "US/Central", "US/Eastern",
        "US/Arizona", "US/Hawaii", "Canada/Eastern", "Europe/London",
        "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
    ]

    return render_template(
        "profile_settings.html",
        notification_prefs=notification_prefs,
        timezones=common_timezones
    )


@app.route('/gameday')
@login_required
def gameday():
    """Daily hub for schedule, reports, notes, and standings."""
    viewer_user = getattr(g, "user", None)
    target_user = viewer_user
    admin_user_options: List[Dict[str, Any]] = []
    requested_user_id = request.args.get("user_id", type=int)

    def _format_user_label(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "Unknown User"
        first = (record.get("first_name") or "").strip()
        last = (record.get("last_name") or "").strip()
        full_name = f"{first} {last}".strip()
        if full_name:
            return full_name
        email = (record.get("email") or "").strip()
        if email:
            return email
        return f"User #{record.get('id')}"

    if session.get("is_admin") and PlayerDB:
        user_rows: List[Dict[str, Any]] = []
        db = None
        try:
            db = PlayerDB()
            user_rows = db.list_users()
        except Exception as exc:
            print(f"Warning fetching users for admin gameday selector: {exc}")
        finally:
            try:
                if db:
                    db.close()
            except Exception:
                pass

        admin_user_options = [
            {"id": row["id"], "label": _format_user_label(row)}
            for row in user_rows
        ]

        if requested_user_id:
            for row in user_rows:
                if row["id"] == requested_user_id:
                    target_user = row
                    break

        if target_user is None and viewer_user is not None:
            target_user = viewer_user
        elif target_user is None and user_rows:
            target_user = user_rows[0]
    else:
        requested_user_id = None

    team_abbr = _determine_user_team(target_user)
    team_metadata = _get_team_metadata(team_abbr)

    _purge_concluded_series_documents()

    # Get date parameter if provided (from calendar click)
    requested_date = request.args.get("date")
    
    # Get raw games first if we need to filter by date
    raw_games = None
    if requested_date:
        try:
            filter_date = datetime.fromisoformat(requested_date).date()
            # Get raw games to filter by date
            use_mock = bool(app.config.get("USE_MOCK_SCHEDULE"))
            if use_mock:
                raw_games = _build_mock_upcoming_games(team_abbr, limit=20)
            elif next_games:
                try:
                    raw_games = next_games(team_abbr, days_ahead=14)
                except Exception:
                    raw_games = []
            
            # Filter raw games by date
            if raw_games:
                filtered_raw = []
                for game in raw_games:
                    game_date_str = game.get("game_date") or game.get("game_date_iso") or game.get("date")
                    if game_date_str:
                        try:
                            if isinstance(game_date_str, str):
                                if 'T' in game_date_str:
                                    game_date = datetime.fromisoformat(game_date_str.split('T')[0]).date()
                                else:
                                    game_date = datetime.fromisoformat(game_date_str).date()
                                if game_date == filter_date:
                                    filtered_raw.append(game)
                        except Exception:
                            continue
                if filtered_raw:
                    raw_games = filtered_raw
        except Exception as e:
            print(f"Warning filtering games by date {requested_date}: {e}")
            raw_games = None
    
    # Get formatted games (will use filtered raw games if available)
    if raw_games:
        # Format the filtered games
        formatted = []
        for game in raw_games[:5]:
            date_str = game.get("game_date") or game.get("game_date_iso") or game.get("date")
            display_date = date_str
            display_time = "TBD"
            if date_str:
                try:
                    if isinstance(date_str, str) and 'T' not in date_str and len(date_str) > 10:
                        display_date = datetime.fromisoformat(date_str).strftime("%a, %b %d")
                    elif isinstance(date_str, str):
                        display_date = datetime.fromisoformat(date_str.split('T')[0] if 'T' in date_str else date_str).strftime("%a, %b %d")
                except Exception:
                    pass

            game_time = game.get("game_datetime")
            if game_time:
                try:
                    display_time = datetime.fromisoformat(game_time.replace("Z", "+00:00")).astimezone().strftime("%I:%M %p %Z")
                except Exception:
                    display_time = "TBD"

            team_abbr_code = _team_abbr_from_id(game.get("opponent_id"))
            # Handle probable_pitchers as either list of dicts or list of strings
            probables_raw = game.get("probable_pitchers") or []
            probables = []
            for p in probables_raw:
                if isinstance(p, dict):
                    name = p.get("name")
                    if name:
                        probables.append(name)
                elif isinstance(p, str):
                    probables.append(p)
            formatted.append({
                "date": display_date,
                "time": display_time,
                "opponent": game.get("opponent_name") or game.get("opponent"),
                "opponent_abbr": team_abbr_code,
                "opponent_id": game.get("opponent_id"),
                "home": game.get("is_home"),
                "venue": game.get("venue"),
                "series": game.get("series_description") or game.get("series"),
                "status": game.get("status"),
                "game_pk": game.get("game_pk"),
                "probable_pitchers": probables,
                "reports": [],
            })
        upcoming_games = formatted
    else:
        upcoming_games = _collect_upcoming_games(team_abbr)
    
    _schedule_auto_reports(upcoming_games, team_abbr)

    player_slug = None
    if target_user and target_user.get("first_name") and target_user.get("last_name"):
        player_slug = _sanitize_filename_component(
            f"{target_user['first_name']} {target_user['last_name']}"
        ).lower().replace(" ", "_")

    recent_reports = []
    for report in _collect_recent_reports(limit=25):
        filename = report.get("filename") or ""
        if player_slug and player_slug not in filename.lower():
            continue
        recent_reports.append({
            **report,
            "url": url_for("download_report_file", filename=filename)
        })

    upcoming_games = _attach_reports_to_games(upcoming_games, recent_reports)
    league_leader_groups = _collect_league_leaders()

    player_documents = []
    document_log = []
    if PlayerDB and target_user and target_user.get("id"):
        db = None
        try:
            db = PlayerDB()
            user_docs = db.list_player_documents(target_user.get("id"))
            player_documents = [_format_player_document(doc) for doc in user_docs]
            events = db.list_player_document_events(player_id=target_user.get("id"), limit=20)
            document_log = [
                {
                    "filename": evt.get("filename"),
                    "action": evt.get("action"),
                    "performed_by": evt.get("performed_by"),
                    "timestamp": evt.get("timestamp"),
                    "timestamp_human": datetime.fromtimestamp(evt["timestamp"]).strftime("%b %d, %Y %I:%M %p")
                    if evt.get("timestamp") else None,
                }
                for evt in events
            ]
        except Exception as exc:
            print(f"Warning fetching player documents: {exc}")
        finally:
            try:
                if db:
                    db.close()
            except Exception:
                pass

    standings_view = request.args.get('standings_view', 'division').lower()
    if standings_view not in {"division", "wildcard"}:
        standings_view = "division"

    requested_division_id = request.args.get('division_id', type=int)
    requested_league_id = request.args.get('league_id', type=int)

    standings_data = _collect_standings_data(
        standings_view,
        team_metadata,
        division_id=requested_division_id,
        league_id=requested_league_id
    )

    selected_division_id = None
    selected_league_id = None
    if standings_data:
        selected_division_id = standings_data.get("division_id")
        selected_league_id = standings_data.get("league_id")
    else:
        selected_division_id = requested_division_id or (team_metadata or {}).get("division_id")
        selected_league_id = requested_league_id or (team_metadata or {}).get("league_id")

    if not selected_division_id and standings_view == "division":
        selected_division_id = (team_metadata or {}).get("division_id") or DIVISION_OPTIONS[0]["id"]

    if not selected_league_id:
        selected_league_id = (team_metadata or {}).get("league_id") or LEAGUE_OPTIONS[0]["id"]

    return render_template(
        'gameday.html',
        team_abbr=team_abbr,
        upcoming_games=upcoming_games,
        league_leader_groups=league_leader_groups,
        player_documents=player_documents,
        document_log=document_log,
        standings_data=standings_data,
        standings_view=standings_view,
        division_options=DIVISION_OPTIONS,
        league_options=LEAGUE_OPTIONS,
        selected_division_id=selected_division_id,
        selected_league_id=selected_league_id,
        admin_user_options=admin_user_options,
        selected_user_id=(target_user.get("id") if target_user else None),
        target_user_label=_format_user_label(target_user),
        viewer_user=viewer_user,
    )


@app.route('/admin')
@admin_required
def admin_dashboard():
    """Render the admin control center."""
    return render_template('admin.html')


def _format_user_record(user: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize user rows for admin API responses."""
    created_ts = user.get("created_at")
    updated_ts = user.get("updated_at")

    def _iso(ts):
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None

    return {
        "id": user.get("id"),
        "email": user.get("email"),
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "created_at": _iso(created_ts),
        "updated_at": _iso(updated_ts),
        "is_admin": bool(user.get("is_admin")),
    }


@app.route('/api/admin/users', methods=['GET'])
@admin_required
def api_admin_users():
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    try:
        db = PlayerDB()
        users = [
            _format_user_record(row)
            for row in db.list_users()
            if not row.get("is_admin")
        ]
        db.close()
        return jsonify({"users": users})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/api/admin/users/<int:user_id>/role', methods=['POST'])
@admin_required
def api_admin_set_role(user_id: int):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    payload = request.get_json(silent=True) or {}
    is_admin = bool(payload.get("is_admin"))

    if g.user and g.user.get("id") == user_id and not is_admin:
        return jsonify({"error": "You cannot revoke your own admin access."}), 400

    try:
        db = PlayerDB()
        db.set_user_admin(user_id, is_admin)
        updated = db.get_user_by_id(user_id)
        db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not updated:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": _format_user_record(updated)})


@app.route('/api/admin/staff-notes', methods=['GET', 'POST'])
@admin_required
def api_admin_staff_notes():
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    if request.method == 'GET':
        team_filter = request.args.get("team")
        try:
            db = PlayerDB()
            notes = db.list_staff_notes(team_abbr=team_filter, limit=100)
            db.close()
            return jsonify({"notes": [_format_staff_note(note) for note in notes]})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # POST - create note
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    body = (payload.get("body") or "").strip()
    team_abbr = payload.get("team_abbr")
    tags = payload.get("tags") or []
    pinned = bool(payload.get("pinned"))

    if not title or not body:
        return jsonify({"error": "Title and body are required."}), 400

    try:
        db = PlayerDB()
        note_id = db.create_staff_note(
            title=title,
            body=body,
            team_abbr=team_abbr,
            tags=tags,
            pinned=pinned,
            author_id=g.user.get("id") if g.user else None,
            author_name=f"{g.user.get('first_name', '')} {g.user.get('last_name', '')}".strip() if g.user else "Admin"
        )
        created = db.get_staff_note(note_id)
        db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"note": _format_staff_note(created)}), 201


@app.route('/api/admin/staff-notes/<int:note_id>', methods=['PUT', 'DELETE'])
@admin_required
def api_admin_staff_note_detail(note_id: int):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    if request.method == 'DELETE':
        try:
            db = PlayerDB()
            deleted = db.delete_staff_note(note_id)
            db.close()
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        if not deleted:
            return jsonify({"error": "Note not found"}), 404

        return jsonify({"status": "deleted"})

    # PUT - update
    payload = request.get_json(silent=True) or {}
    fields = {}
    if "title" in payload:
        fields["title"] = payload["title"]
    if "body" in payload:
        fields["body"] = payload["body"]
    if "team_abbr" in payload:
        fields["team_abbr"] = payload["team_abbr"]
    if "tags" in payload:
        fields["tags"] = payload["tags"]
    if "pinned" in payload:
        fields["pinned"] = payload["pinned"]

    if not fields:
        return jsonify({"error": "No updates supplied."}), 400

    try:
        db = PlayerDB()
        updated = db.update_staff_note(note_id, **fields)
        note = db.get_staff_note(note_id) if updated else None
        db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not updated or not note:
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"note": _format_staff_note(note)})


@app.route('/api/admin/players', methods=['GET'])
@admin_required
def api_admin_players():
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    search = request.args.get("search", "").strip() or None
    limit = request.args.get("limit", type=int) or 200

    try:
        db = PlayerDB()
        players = db.search_players(search=search, limit=limit)
        db.close()
        return jsonify({"players": players})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _format_player_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize player document metadata."""
    uploaded_ts = doc.get("uploaded_at")
    series_start_ts = doc.get("series_start")
    series_end_ts = doc.get("series_end")
    category = (doc.get("category") or "").strip().lower() if doc else None

    def _fmt(ts):
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).strftime("%b %d, %Y %I:%M %p")
        except Exception:
            return None

    def _iso(ts):
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None

    def _fmt_date(ts):
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).strftime("%b %d, %Y")
        except Exception:
            return None

    now_ts = datetime.now().timestamp()
    series_status = None
    start_date = datetime.fromtimestamp(series_start_ts).date() if series_start_ts else None
    end_date = datetime.fromtimestamp(series_end_ts).date() if series_end_ts else None
    today_date = datetime.fromtimestamp(now_ts).date()
    if series_start_ts and series_end_ts:
        if series_end_ts < now_ts:
            series_status = "expired"
        elif start_date and end_date and start_date <= today_date <= end_date:
            series_status = "current"
        elif series_start_ts <= now_ts <= series_end_ts:
            series_status = "current"
        else:
            series_status = "upcoming"

    series_start_display = _fmt_date(series_start_ts)
    series_end_display = _fmt_date(series_end_ts)
    if series_start_display and series_end_display:
        if series_start_display == series_end_display:
            series_range_display = series_start_display
        else:
            series_range_display = f"{series_start_display} – {series_end_display}"
    else:
        series_range_display = series_start_display or series_end_display

    viewer_url = None
    if category == WORKOUT_CATEGORY:
        viewer_url = url_for("view_workout_document", doc_id=doc.get("id"))

    return {
        "id": doc.get("id"),
        "player_id": doc.get("player_id"),
        "filename": doc.get("filename"),
        "uploaded_at": _fmt(uploaded_ts),
        "uploaded_at_iso": _iso(uploaded_ts),
        "download_url": url_for("download_player_document", doc_id=doc.get("id")),
        "uploaded_by": doc.get("uploaded_by"),
        "category": category,
        "viewer_url": viewer_url,
        "series_opponent": doc.get("series_opponent"),
        "series_label": doc.get("series_label"),
        "series_start": _iso(series_start_ts),
        "series_start_display": series_start_display,
        "series_end": _iso(series_end_ts),
        "series_end_display": series_end_display,
        "series_range_display": series_range_display,
        "series_status": series_status,
    }


def _validate_workout_upload(file_storage) -> Optional[str]:
    if not file_storage:
        return "No document provided."
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        return "Please choose a file to upload."
    ext = Path(filename).suffix.lower()
    if ext not in WORKOUT_ALLOWED_EXTENSIONS:
        return "Unsupported file type. Upload a PDF workout sheet."
    return None


@app.route('/api/admin/workouts', methods=['POST'])
@admin_required
def api_admin_workouts_upload():
    """Upload or replace the current workout document."""
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    csrf_token = request.form.get("csrf_token")
    if not validate_csrf(csrf_token):
        return jsonify({"error": "Invalid CSRF token. Refresh the page and try again."}), 400

    player_id_raw = (request.form.get("player_id") or "").strip()
    if not player_id_raw:
        return jsonify({"error": "Select a player before uploading."}), 400
    try:
        player_id = int(player_id_raw)
    except ValueError:
        return jsonify({"error": "Invalid player selection."}), 400

    file = request.files.get("document")
    error = _validate_workout_upload(file)
    if error:
        return jsonify({"error": error}), 400

    original_filename = secure_filename(file.filename)
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    storage_name = f"{timestamp_label}_{original_filename}"
    dest_path: Optional[Path] = None
    db = None
    try:
        db = PlayerDB()
        player_record = db.get_user_by_id(player_id)
        if not player_record:
            raise LookupError("Player account not found.")
        if player_record.get("is_admin"):
            raise PermissionError("Cannot attach workouts to admin accounts.")

        player_dir = WORKOUT_DOCS_DIR / str(player_id)
        player_dir.mkdir(parents=True, exist_ok=True)
        dest_path = player_dir / storage_name

        try:
            file.save(dest_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to save workout document: {exc}") from exc

        uploader_id = g.user.get("id") if g.user else None
        doc_id = db.create_player_document(
            player_id=player_id,
            filename=original_filename,
            path=str(dest_path),
            uploaded_by=uploader_id,
            category=WORKOUT_CATEGORY,
        )
        doc = db.get_player_document(doc_id)
        db.record_player_document_event(
            player_id=player_id,
            filename=original_filename,
            action="upload_workout",
            performed_by=uploader_id,
        )
    except LookupError as exc:
        if dest_path and dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        return jsonify({"error": str(exc)}), 404
    except PermissionError as exc:
        if dest_path and dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        return jsonify({"error": str(exc)}), 403
    except Exception as exc:
        if dest_path and dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        return jsonify({"error": str(exc)}), 500
    finally:
        if db:
            db.close()

    return jsonify({"workout": _format_player_document(doc)}), 201


@app.route('/api/workouts/latest', methods=['GET'])
@login_required
def api_workouts_latest():
    """Return the latest workout document metadata."""
    if not PlayerDB:
        return jsonify({"workout": None})

    viewer_user = getattr(g, "user", None)
    if not viewer_user:
        return jsonify({"error": "User session unavailable."}), 403

    requested_player_id = request.args.get("player_id", type=int)
    player_id = requested_player_id or viewer_user.get("id")

    if not player_id:
        return jsonify({"workout": None})

    if (
        requested_player_id
        and not session.get("is_admin")
        and player_id != viewer_user.get("id")
    ):
        return jsonify({"error": "Not authorized to view this workout."}), 403

    db = None
    doc = None
    try:
        db = PlayerDB()
        doc = db.get_latest_player_document_by_category(player_id, WORKOUT_CATEGORY)
    except Exception as exc:
        if db:
            db.close()
            db = None
        return jsonify({"error": str(exc)}), 500
    finally:
        if db:
            db.close()

    if not doc:
        return jsonify({"workout": None})

    return jsonify({"workout": _format_player_document(doc)})


@app.route('/api/admin/workouts/player/<int:player_id>', methods=['GET'])
@admin_required
def api_admin_workouts_for_player(player_id: int):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    db = None
    try:
        db = PlayerDB()
        player = db.get_user_by_id(player_id)
        if not player or player.get("is_admin"):
            return jsonify({"error": "Player account not found."}), 404
        docs = db.list_player_documents(player_id, category=WORKOUT_CATEGORY)
        formatted = [_format_player_document(doc) for doc in docs]
    except Exception as exc:
        if db:
            db.close()
        return jsonify({"error": str(exc)}), 500
    finally:
        if db:
            db.close()

    return jsonify({"workouts": formatted})


@app.route('/api/admin/workouts/<int:doc_id>', methods=['DELETE'])
@admin_required
def api_admin_workouts_delete(doc_id: int):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    db = None
    doc = None
    try:
        db = PlayerDB()
        doc = db.get_player_document(doc_id)
        if not doc or (doc.get("category") or "").strip().lower() != WORKOUT_CATEGORY:
            db.close()
            return jsonify({"error": "Workout not found."}), 404
        deleted = db.delete_player_document(doc_id)
        if not deleted:
            db.close()
            return jsonify({"error": "Workout not found."}), 404
        doc = deleted
        db.record_player_document_event(
            player_id=doc.get("player_id"),
            filename=doc.get("filename"),
            action="delete_workout",
            performed_by=g.user.get("id") if g.user else None,
        )
    except Exception as exc:
        if db:
            db.close()
        return jsonify({"error": str(exc)}), 500
    finally:
        if db:
            db.close()

    path = Path((doc or {}).get("path") or "")
    if path.exists() and path.is_file():
        try:
            path.unlink()
        except OSError as exc:
            print(f"Warning removing workout document file {path}: {exc}")

    return jsonify({
        "status": "deleted",
        "workout": {
            "id": doc.get("id"),
            "player_id": doc.get("player_id"),
            "filename": doc.get("filename"),
        }
    })


@app.route('/api/admin/player-docs', methods=['POST'])
@admin_required
def api_admin_player_docs_upload():
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    player_id_raw = request.form.get("player_id", "").strip()
    file = request.files.get("document")

    if not player_id_raw:
        return jsonify({"error": "Player selection is required."}), 400
    if not file or not file.filename:
        return jsonify({"error": "A document must be provided."}), 400

    series_choice = (request.form.get("series_id") or "").strip()
    raw_series_opponent = clean_str(request.form.get("series_opponent")).upper()
    raw_series_label = clean_str(request.form.get("series_label"))
    series_start_raw = request.form.get("series_start", "").strip()
    series_end_raw = request.form.get("series_end", "").strip()

    def _parse_series_ts(raw: str) -> Optional[float]:
        if not raw:
            return None
        value = raw
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            try:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                return None

    series_opponent = None
    series_label = None
    series_start_ts = None
    series_end_ts = None

    if series_choice and series_choice != "__none__":
        series_start_ts = _parse_series_ts(series_start_raw)
        series_end_ts = _parse_series_ts(series_end_raw)
        series_opponent = raw_series_opponent or None
        series_label = raw_series_label or None

        if not series_opponent or series_start_ts is None or series_end_ts is None:
            return jsonify({"error": "Series selection is required before uploading a document."}), 400

        if series_end_ts < series_start_ts:
            return jsonify({"error": "Series end date must be after its start date."}), 400

    try:
        player_id = int(player_id_raw)
    except ValueError:
        return jsonify({"error": "Invalid player identifier."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    player_dir = PLAYER_DOCS_DIR / str(player_id)
    player_dir.mkdir(parents=True, exist_ok=True)
    dest_path = player_dir / filename

    try:
        file.save(dest_path)
        db = PlayerDB()
        player_record = db.get_user_by_id(player_id)
        if not player_record:
            db.close()
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            return jsonify({"error": "Player account not found."}), 404
        doc_id = db.create_player_document(
            player_id=player_id,
            filename=filename,
            path=str(dest_path),
            uploaded_by=g.user.get("id") if g.user else None,
            series_opponent=series_opponent,
            series_label=series_label,
            series_start=series_start_ts,
            series_end=series_end_ts
        )
        doc = db.get_player_document(doc_id)
        db.record_player_document_event(
            player_id=player_id,
            filename=filename,
            action="upload",
            performed_by=g.user.get("id") if g.user else None
        )
        db.close()
    except Exception as exc:
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        return jsonify({"error": str(exc)}), 500

    return jsonify({"document": _format_player_document(doc)}), 201


@app.route('/api/admin/player-series/<int:user_id>', methods=['GET'])
@admin_required
def api_admin_player_series(user_id: int):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500
    try:
        db = PlayerDB()
        user = db.get_user_by_id(user_id)
        db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not user or user.get("is_admin"):
        return jsonify({"series": []})

    team_abbr = _determine_user_team(user)
    series = _collect_series_for_team(team_abbr)
    return jsonify({"series": series, "team": team_abbr})


@app.route('/api/admin/player-docs/<player_id>', methods=['GET'])
@admin_required
def api_admin_player_docs_list(player_id: str):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500
    _purge_concluded_series_documents()
    try:
        db = PlayerDB()
        docs = db.list_player_documents(int(player_id))
        db.close()
        return jsonify({"documents": [_format_player_document(doc) for doc in docs]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/player-docs/<int:doc_id>')
@login_required
def download_player_document(doc_id: int):
    if not PlayerDB:
        abort(404)
    _purge_concluded_series_documents()
    try:
        db = PlayerDB()
        doc = db.get_player_document(doc_id)
        db.close()
    except Exception:
        doc = None
    if not doc:
        abort(404)
    path = Path(doc.get("path") or "")
    if not path.exists() or not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name=doc.get("filename") or path.name)


@app.route('/workout-docs/<int:doc_id>')
@login_required
def view_workout_document(doc_id: int):
    if not PlayerDB:
        abort(404)
    try:
        db = PlayerDB()
        doc = db.get_player_document(doc_id)
        db.close()
    except Exception:
        doc = None
    if not doc or (doc.get("category") or "").strip().lower() != WORKOUT_CATEGORY:
        abort(404)
    viewer_user = getattr(g, "user", None)
    viewer_id = viewer_user.get("id") if viewer_user else None
    if viewer_id is None:
        abort(403)
    if doc.get("player_id") != viewer_id and not session.get("is_admin"):
        abort(403)
    path = Path(doc.get("path") or "")
    if not path.exists() or not path.is_file():
        abort(404)
    mime_type, _ = mimetypes.guess_type(path.name)
    return send_file(
        path,
        as_attachment=False,
        download_name=doc.get("filename") or path.name,
        mimetype=mime_type or "application/pdf",
    )


@app.route('/api/admin/player-docs/<int:doc_id>', methods=['DELETE'])
@admin_required
def api_admin_player_docs_delete(doc_id: int):
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500
    try:
        db = PlayerDB()
        doc = db.delete_player_document(doc_id)
        if not doc:
            db.close()
            return jsonify({"error": "Document not found"}), 404
        db.record_player_document_event(
            player_id=doc["player_id"],
            filename=doc["filename"],
            action="delete",
            performed_by=g.user.get("id") if g.user else None
        )
        db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    try:
        path = Path(doc.get("path") or "")
        if path.exists() and path.is_file():
            path.unlink()
    except OSError as exc:
        print(f"Warning removing document file: {exc}")

    return jsonify({"status": "deleted", "document": _format_player_document(doc)})


@app.route('/api/admin/player-docs/logs', methods=['GET'])
@admin_required
def api_admin_player_docs_logs():
    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500
    _purge_concluded_series_documents()
    player_id = request.args.get("player_id", type=int)
    limit = request.args.get("limit", type=int) or 200
    try:
        db = PlayerDB()
        events = db.list_player_document_events(player_id=player_id, limit=limit)
        db.close()

        def _format_event(evt: Dict[str, Any]) -> Dict[str, Any]:
            ts = evt.get("timestamp")
            try:
                human = datetime.fromtimestamp(ts).strftime("%b %d, %Y %I:%M %p") if ts else None
            except Exception:
                human = None
            return {
                "id": evt.get("id"),
                "player_id": evt.get("player_id"),
                "filename": evt.get("filename"),
                "action": evt.get("action"),
                "performed_by": evt.get("performed_by"),
                "timestamp": ts,
                "timestamp_human": human,
            }

        return jsonify({"events": [_format_event(evt) for evt in events]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route('/api/admin/staff-notes/<int:note_id>/pin', methods=['POST'])
@admin_required
def api_admin_staff_note_pin(note_id: int):
    payload = request.get_json(silent=True) or {}
    pinned = bool(payload.get("pinned"))

    if not PlayerDB:
        return jsonify({"error": "Database unavailable"}), 500

    try:
        db = PlayerDB()
        updated = db.update_staff_note(note_id, pinned=pinned)
        note = db.get_staff_note(note_id) if updated else None
        db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not updated or not note:
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"note": _format_staff_note(note)})


@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')


@app.route('/api/settings', methods=['GET', 'PUT', 'PATCH', 'POST', 'DELETE'])
def api_settings():
    """Retrieve or update application settings."""
    try:
        if request.method == 'GET':
            return jsonify(get_cached_settings())

        if request.method == 'DELETE':
            defaults = settings_manager.reset_settings()
            refresh_settings_cache()
            return jsonify(defaults)

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        if not isinstance(payload, dict):
            return jsonify({"error": "Settings payload must be a JSON object"}), 400

        updated = settings_manager.update_settings(payload)
        refresh_settings_cache()
        return jsonify(updated)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to process settings: {exc}"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Start report generation (supports single or batch)"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON in request"}), 400
        
        # Support both old 'hitter_name' and new 'hitter_names' for backward compatibility
        hitter_names_raw = data.get('hitter_names') or data.get('hitter_name')
        hitter_names_str = (hitter_names_raw or '').strip()
        
        if not hitter_names_str:
            return jsonify({"error": "Player name(s) are required"}), 400
        
        # Parse entries - split by newlines or commas
        # Each entry can be just a name or "Name | Team | Opponent"
        entries = [entry.strip() for entry in hitter_names_str.replace(',', '\n').split('\n') if entry.strip()]
        
        if not entries:
            return jsonify({"error": "Please enter at least one player name"}), 400
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Get default parameters (used if not specified per-player)
        settings = get_cached_settings()
        report_defaults = settings.get("reports", {})

        default_team = clean_str(data.get('team')) or clean_str(report_defaults.get('default_team')) or 'AUTO'
        season_start = clean_str(data.get('season_start')) or clean_str(report_defaults.get('default_season_start')) or '2025-03-20'
        use_next_series = parse_bool(data.get('use_next_series'), report_defaults.get('use_next_series', False))
        opponent_team_raw = data.get('opponent_team')
        default_opponent = clean_str(opponent_team_raw) or clean_str(report_defaults.get('default_opponent')) or None
        
        # Determine if this is a batch or single report
        is_batch = len(entries) > 1
        
        if is_batch:
            # Initialize batch job status
            job_status[job_id] = {
                "status": "queued",
                "message": f"Queued batch of {len(entries)} players...",
                "total": len(entries),
                "completed": 0,
                "failed": 0,
                "pdfs": [],
                "errors": []
            }
            
            # Start background thread for batch processing
            thread = threading.Thread(
                target=generate_batch_reports,
                args=(entries, default_team, season_start, use_next_series, default_opponent, job_id)
            )
        else:
            # Single report (backward compatible)
            # Parse the entry in case it has team/opponent specified
            hitter_name, team, opponent = parse_player_entry(entries[0])
            if not hitter_name:
                return jsonify({"error": "Invalid player name format"}), 400
            
            player_team = team if team else default_team
            player_opponent = opponent if opponent else default_opponent
            
            job_status[job_id] = {"status": "queued", "message": "Queued for processing..."}
            
            # Start background thread
            thread = threading.Thread(
                target=generate_report,
                args=(hitter_name, player_team, season_start, use_next_series, player_opponent, job_id)
            )
        
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "is_batch": is_batch,
            "total": len(entries) if is_batch else 1
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": f"Server error: {error_msg}"}), 500

@app.route('/generate-pitcher', methods=['POST'])
def generate_pitcher():
    """Start pitcher report generation (supports single or batch)"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON in request"}), 400
        
        # Get pitcher names
        pitcher_names_raw = data.get('pitcher_names') or data.get('pitcher_name')
        pitcher_names_str = (pitcher_names_raw or '').strip()
        
        if not pitcher_names_str:
            return jsonify({"error": "Pitcher name(s) are required"}), 400
        
        # Parse entries - split by newlines or commas
        # Each entry can be just a name or "Name | Team | Opponent"
        entries = [entry.strip() for entry in pitcher_names_str.replace(',', '\n').split('\n') if entry.strip()]
        
        if not entries:
            return jsonify({"error": "Please enter at least one pitcher name"}), 400
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Get default parameters (used if not specified per-player)
        settings = get_cached_settings()
        report_defaults = settings.get("reports", {})

        default_team = clean_str(data.get('team')) or clean_str(report_defaults.get('default_pitcher_team')) or clean_str(report_defaults.get('default_team')) or 'AUTO'
        season_start = clean_str(data.get('season_start')) or clean_str(report_defaults.get('default_season_start')) or '2025-03-20'
        use_next_series = parse_bool(data.get('use_next_series'), report_defaults.get('use_next_series', False))
        opponent_team_raw = data.get('opponent_team')
        default_opponent = clean_str(opponent_team_raw) or clean_str(report_defaults.get('default_opponent')) or None
        
        # Determine if this is a batch or single report
        is_batch = len(entries) > 1
        
        if is_batch:
            # Initialize batch job status
            job_status[job_id] = {
                "status": "queued",
                "message": f"Queued batch of {len(entries)} pitchers...",
                "total": len(entries),
                "completed": 0,
                "failed": 0,
                "pdfs": [],
                "errors": []
            }
            
            # Start background thread for batch processing
            thread = threading.Thread(
                target=generate_batch_pitcher_reports,
                args=(entries, default_team, season_start, use_next_series, default_opponent, job_id)
            )
        else:
            # Single report
            # Parse the entry in case it has team/opponent specified
            pitcher_name, team, opponent = parse_pitcher_entry(entries[0])
            if not pitcher_name:
                return jsonify({"error": "Invalid pitcher name format"}), 400
            
            player_team = team if team else default_team
            player_opponent = opponent if opponent else default_opponent
            
            job_status[job_id] = {"status": "queued", "message": "Queued for processing..."}
            
            # Start background thread
            thread = threading.Thread(
                target=generate_pitcher_report,
                args=(pitcher_name, player_team, season_start, use_next_series, player_opponent, job_id)
            )
        
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "is_batch": is_batch,
            "total": len(entries) if is_batch else 1
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": f"Error generating pitcher report: {error_msg}"}), 500

@app.route('/status/<job_id>')
def status(job_id):
    """Check the status of a job"""
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    
    status_info = job_status[job_id].copy()
    
    # Clean up old completed/error jobs (keep last 10)
    if status_info["status"] in ["completed", "error"]:
        completed_jobs = [jid for jid, info in job_status.items() 
                         if info["status"] in ["completed", "error"]]
        if len(completed_jobs) > 10:
            oldest = completed_jobs[0]
            if oldest != job_id:
                del job_status[oldest]
    
    return jsonify(status_info)

@app.route('/download/<job_id>')
def download(job_id):
    """Download the generated PDF (for single reports)"""
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    
    status_info = job_status[job_id]
    
    if status_info["status"] != "completed":
        return jsonify({"error": "Report not ready yet"}), 400
    
    pdf_path = status_info.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return jsonify({"error": "PDF file not found"}), 404
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=status_info.get("pdf_filename", "report.pdf")
    )

@app.route('/download/<job_id>/<int:pdf_index>')
def download_batch_pdf(job_id, pdf_index):
    """Download a specific PDF from a batch"""
    if job_id not in job_status:
        return jsonify({"error": "Job not found"}), 404
    
    status_info = job_status[job_id]
    
    if status_info["status"] != "completed":
        return jsonify({"error": "Batch not ready yet"}), 400
    
    pdfs = status_info.get("pdfs", [])
    if pdf_index < 0 or pdf_index >= len(pdfs):
        return jsonify({"error": "Invalid PDF index"}), 404
    
    pdf_info = pdfs[pdf_index]
    pdf_path = pdf_info.get("path")
    
    if not pdf_path or not Path(pdf_path).exists():
        return jsonify({"error": "PDF file not found"}), 404
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=pdf_info.get("filename", "report.pdf")
    )

@app.route('/reports/files/<path:filename>')
@login_required
def download_report_file(filename):
    """Provide direct download access for generated reports."""
    safe_name = os.path.basename(filename)
    pdf_path = OUT_DIR / safe_name
    if not pdf_path.exists() or not pdf_path.is_file():
        flash("Report not found.", "error")
        return redirect(url_for('gameday'))
    return send_file(pdf_path, as_attachment=True, download_name=safe_name)

@app.route('/reports')
def list_reports():
    """List all available reports"""
    pdf_files = sorted(OUT_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports = [
        {
            "name": pdf.name,
            "size": pdf.stat().st_size,
            "created": datetime.fromtimestamp(pdf.stat().st_mtime).isoformat()
        }
        for pdf in pdf_files[:20]  # Last 20 reports
    ]
    return jsonify({"reports": reports})

if __name__ == '__main__':
    import socket
    
    # Find an available port starting from 5000
    def find_free_port(start_port=5000):
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return 5001  # Fallback
    
    port = find_free_port(5001)  # Start from 5001 to match browser config
    
    print(f"Starting Scouting Report Web UI...")
    print(f"Reports will be saved to: {OUT_DIR}")
    print(f"Open http://127.0.0.1:{port} in your browser")
    app.run(debug=True, host='127.0.0.1', port=port)

