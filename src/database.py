# src/database.py
"""
Player Database - SQLite schema and operations
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

class PlayerDB:
    """Database operations for player data"""
    
    def __init__(self, db_path: str = "build/database/players.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like row access
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id TEXT PRIMARY KEY,
                abbreviation TEXT,
                name TEXT,
                city TEXT,
                league TEXT,
                division TEXT,
                updated_at REAL
            )
        """)
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                sportradar_id TEXT PRIMARY KEY,
                mlbam_id TEXT,
                name TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                position TEXT,
                primary_position TEXT,
                team_id TEXT,
                team_abbr TEXT,
                jersey_number TEXT,
                handedness TEXT,
                height TEXT,
                weight INTEGER,
                birth_date TEXT,
                birth_place TEXT,
                debut_date TEXT,
                updated_at REAL,
                FOREIGN KEY (team_id) REFERENCES teams(team_id)
            )
        """)
        
        # Player stats (time-series)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sportradar_id TEXT NOT NULL,
                season TEXT,
                stat_type TEXT,
                category TEXT,
                value REAL,
                updated_at REAL,
                FOREIGN KEY (sportradar_id) REFERENCES players(sportradar_id),
                UNIQUE(sportradar_id, season, stat_type, category)
            )
        """)
        
        # Player seasons (season-level aggregated stats)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_seasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sportradar_id TEXT NOT NULL,
                season TEXT NOT NULL,
                games INTEGER,
                at_bats INTEGER,
                hits INTEGER,
                doubles INTEGER,
                triples INTEGER,
                home_runs INTEGER,
                rbi INTEGER,
                runs INTEGER,
                stolen_bases INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                avg REAL,
                obp REAL,
                slg REAL,
                ops REAL,
                updated_at REAL,
                FOREIGN KEY (sportradar_id) REFERENCES players(sportradar_id),
                UNIQUE(sportradar_id, season)
            )
        """)
        
        # Player history (transactions, team changes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sportradar_id TEXT NOT NULL,
                date TEXT NOT NULL,
                event_type TEXT,
                from_team TEXT,
                to_team TEXT,
                details TEXT,
                updated_at REAL,
                FOREIGN KEY (sportradar_id) REFERENCES players(sportradar_id)
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_abbr)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_seasons_player_season ON player_seasons(sportradar_id, season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_stats_player_season ON player_stats(sportradar_id, season)")
        
        # Users table for authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

        # Ensure legacy databases gain the is_admin column
        cursor.execute("PRAGMA table_info(users)")
        user_columns = {row[1] for row in cursor.fetchall()}
        if "updated_at" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN updated_at REAL NOT NULL DEFAULT 0")
        if "is_admin" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        if "theme_preference" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN theme_preference TEXT DEFAULT 'dark'")
        if "profile_image_path" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_image_path TEXT")
        if "bio" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN bio TEXT")
        if "job_title" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN job_title TEXT")
        if "pronouns" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN pronouns TEXT")
        if "phone" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        if "timezone" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN timezone TEXT")
        if "notification_preferences" not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN notification_preferences TEXT")

        # Player documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                uploaded_by INTEGER,
                uploaded_at REAL NOT NULL,
                category TEXT,
                series_opponent TEXT,
                series_label TEXT,
                series_start REAL,
                series_end REAL,
                FOREIGN KEY (player_id) REFERENCES users(id),
                FOREIGN KEY (uploaded_by) REFERENCES users(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_documents_player ON player_documents(player_id)")

        # Ensure legacy databases pick up the series metadata columns
        cursor.execute("PRAGMA table_info(player_documents)")
        pd_columns = {row[1] for row in cursor.fetchall()}
        if "category" not in pd_columns:
            cursor.execute("ALTER TABLE player_documents ADD COLUMN category TEXT")
        if "series_opponent" not in pd_columns:
            cursor.execute("ALTER TABLE player_documents ADD COLUMN series_opponent TEXT")
        if "series_label" not in pd_columns:
            cursor.execute("ALTER TABLE player_documents ADD COLUMN series_label TEXT")
        if "series_start" not in pd_columns:
            cursor.execute("ALTER TABLE player_documents ADD COLUMN series_start REAL")
        if "series_end" not in pd_columns:
            cursor.execute("ALTER TABLE player_documents ADD COLUMN series_end REAL")

        # Player document activity log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_document_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                action TEXT NOT NULL,
                performed_by INTEGER,
                timestamp REAL NOT NULL,
                FOREIGN KEY (player_id) REFERENCES users(id),
                FOREIGN KEY (performed_by) REFERENCES users(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_document_log_player ON player_document_log(player_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_document_log_time ON player_document_log(timestamp)")

        # Journal entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date TEXT NOT NULL,
                visibility TEXT NOT NULL,
                title TEXT,
                body TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, entry_date, visibility)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_journal_entries_user_date
            ON journal_entries(user_id, entry_date)
        """)

        # Staff notes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS staff_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                team_abbr TEXT,
                tags TEXT,
                pinned INTEGER NOT NULL DEFAULT 0,
                author_id INTEGER,
                author_name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (author_id) REFERENCES users(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_staff_notes_team ON staff_notes(team_abbr)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_staff_notes_pinned ON staff_notes(pinned)")
        
        self.conn.commit()
    
    def upsert_team(self, team_data: Dict[str, Any]):
        """Insert or update team"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO teams 
            (team_id, abbreviation, name, city, league, division, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            team_data.get('id') or team_data.get('team_id'),
            team_data.get('abbreviation') or team_data.get('abbr'),
            team_data.get('name') or team_data.get('team_name'),
            team_data.get('city'),
            team_data.get('league'),
            team_data.get('division'),
            datetime.now().timestamp()
        ))
        self.conn.commit()
    
    def upsert_player(self, player_data: Dict[str, Any]):
        """Insert or update player"""
        cursor = self.conn.cursor()
        
        # Extract player ID (try multiple possible fields)
        player_id = (player_data.get('id') or 
                    player_data.get('player_id') or 
                    player_data.get('sportradar_id'))
        
        if not player_id:
            raise ValueError("Player data missing ID")
        
        cursor.execute("""
            INSERT OR REPLACE INTO players 
            (sportradar_id, mlbam_id, name, first_name, last_name, position, 
             primary_position, team_id, team_abbr, jersey_number, handedness,
             height, weight, birth_date, birth_place, debut_date, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_id,
            player_data.get('mlbam_id') or player_data.get('mlb_id'),
            player_data.get('name') or player_data.get('full_name') or 
                f"{player_data.get('first_name', '')} {player_data.get('last_name', '')}".strip(),
            player_data.get('first_name'),
            player_data.get('last_name'),
            player_data.get('position') or player_data.get('primary_position'),
            player_data.get('primary_position') or player_data.get('position'),
            player_data.get('team_id') or player_data.get('team', {}).get('id'),
            player_data.get('team_abbr') or player_data.get('team', {}).get('abbreviation'),
            player_data.get('jersey_number') or player_data.get('jersey'),
            player_data.get('handedness') or player_data.get('bats'),
            player_data.get('height'),
            player_data.get('weight'),
            player_data.get('birth_date') or player_data.get('date_of_birth'),
            player_data.get('birth_place'),
            player_data.get('debut_date') or player_data.get('mlb_debut'),
            datetime.now().timestamp()
        ))
        self.conn.commit()
    
    def upsert_player_season(self, player_id: str, season: str, stats: Dict[str, Any]):
        """Insert or update player season stats"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO player_seasons
            (sportradar_id, season, games, at_bats, hits, doubles, triples,
             home_runs, rbi, runs, stolen_bases, walks, strikeouts,
             avg, obp, slg, ops, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_id,
            season,
            stats.get('games') or stats.get('games_played'),
            stats.get('at_bats') or stats.get('ab'),
            stats.get('hits') or stats.get('h'),
            stats.get('doubles') or stats.get('2b') or stats.get('doubles'),
            stats.get('triples') or stats.get('3b') or stats.get('triples'),
            stats.get('home_runs') or stats.get('hr'),
            stats.get('rbi') or stats.get('runs_batted_in'),
            stats.get('runs') or stats.get('r'),
            stats.get('stolen_bases') or stats.get('sb'),
            stats.get('walks') or stats.get('bb'),
            stats.get('strikeouts') or stats.get('so') or stats.get('k'),
            stats.get('avg') or stats.get('batting_average'),
            stats.get('obp') or stats.get('on_base_percentage'),
            stats.get('slg') or stats.get('slugging_percentage'),
            stats.get('ops') or stats.get('on_base_plus_slugging'),
            datetime.now().timestamp()
        ))
        self.conn.commit()
    
    def search_players(self, search: Optional[str] = None, team: Optional[str] = None,
                      position: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Search players with filters"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM players WHERE 1=1"
        params = []
        
        if search:
            query += " AND (name LIKE ? OR first_name LIKE ? OR last_name LIKE ?)"
            search_term = f"%{search}%"
            params.extend([search_term, search_term, search_term])
        
        if team:
            query += " AND team_abbr = ?"
            params.append(team.upper())
        
        if position:
            query += " AND position LIKE ?"
            params.append(f"%{position}%")
        
        query += " ORDER BY name LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_player(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Get player by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM players WHERE sportradar_id = ?", (player_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_player_seasons(self, player_id: str) -> List[Dict[str, Any]]:
        """Get all season stats for a player"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM player_seasons 
            WHERE sportradar_id = ? 
            ORDER BY season DESC
        """, (player_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_player_current_season(self, player_id: str, season: str = "2024") -> Optional[Dict[str, Any]]:
        """Get current season stats for a player"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM player_seasons 
            WHERE sportradar_id = ? AND season = ?
        """, (player_id, season))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_teams(self) -> List[Dict[str, Any]]:
        """Get all teams"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM teams ORDER BY abbreviation")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def count_players(self, search: Optional[str] = None, team: Optional[str] = None,
                     position: Optional[str] = None) -> int:
        """Count players matching filters"""
        cursor = self.conn.cursor()
        
        query = "SELECT COUNT(*) FROM players WHERE 1=1"
        params = []
        
        if search:
            query += " AND (name LIKE ? OR first_name LIKE ? OR last_name LIKE ?)"
            search_term = f"%{search}%"
            params.extend([search_term, search_term, search_term])
        
        if team:
            query += " AND team_abbr = ?"
            params.append(team.upper())
        
        if position:
            query += " AND position LIKE ?"
            params.append(f"%{position}%")
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
    def close(self):
        """Close database connection"""
        self.conn.close()

    # ---------------------------
    # Authentication helpers
    # ---------------------------

    def create_user(self, email: str, password_hash: str, first_name: str, last_name: str, is_admin: bool = False) -> int:
        """Insert a new user and return the user ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO users (email, password_hash, first_name, last_name, created_at, updated_at, is_admin)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            (email or "").strip().lower(),
            password_hash,
            (first_name or "").strip(),
            (last_name or "").strip(),
            datetime.now().timestamp(),
            datetime.now().timestamp(),
            1 if is_admin else 0
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve a user record by email."""
        if not email:
            return None
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", ((email or "").strip().lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a user record by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """Return all users sorted by creation date."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, email, first_name, last_name, created_at, updated_at, is_admin
            FROM users
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def set_user_admin(self, user_id: int, is_admin: bool) -> None:
        """Toggle admin flag for a user."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET is_admin = ?, updated_at = ? WHERE id = ?",
            (1 if is_admin else 0, datetime.now().timestamp(), user_id)
        )
        self.conn.commit()

    def update_user_password(self, user_id: int, password_hash: str) -> None:
        """Update a user's password hash."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (password_hash, datetime.now().timestamp(), user_id)
        )
        self.conn.commit()

    def update_user_profile(self, user_id: int, **fields) -> bool:
        """Update user profile metadata."""
        allowed_fields = {
            "first_name",
            "last_name",
            "theme_preference",
            "profile_image_path",
            "bio",
            "job_title",
            "pronouns",
            "phone",
            "timezone",
            "notification_preferences",
        }
        updates = {key: value for key, value in fields.items() if key in allowed_fields}
        if not updates:
            return False

        assignments = []
        params = []
        for column, value in updates.items():
            assignments.append(f"{column} = ?")
            params.append(value)

        assignments.append("updated_at = ?")
        params.append(datetime.now().timestamp())
        params.append(user_id)

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE users SET {', '.join(assignments)} WHERE id = ?",
            params
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ---------------------------
    # Staff notes helpers
    # ---------------------------

    def create_staff_note(self, title: str, body: str, author_id: Optional[int], author_name: str,
                          team_abbr: Optional[str] = None, tags: Optional[List[str]] = None,
                          pinned: bool = False) -> int:
        """Create a staff note and return its ID."""
        cursor = self.conn.cursor()
        now_ts = datetime.now().timestamp()
        cursor.execute("""
            INSERT INTO staff_notes (title, body, team_abbr, tags, pinned, author_id, author_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            (title or "").strip(),
            (body or "").strip(),
            (team_abbr or "").strip().upper() or None,
            json.dumps(tags or []),
            1 if pinned else 0,
            author_id,
            author_name,
            now_ts,
            now_ts,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def update_staff_note(self, note_id: int, **fields) -> bool:
        """Update an existing staff note with provided fields."""
        allowed = {"title", "body", "team_abbr", "tags", "pinned"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False

        params = []
        assignments = []
        if "title" in updates:
            assignments.append("title = ?")
            params.append((updates["title"] or "").strip())
        if "body" in updates:
            assignments.append("body = ?")
            params.append((updates["body"] or "").strip())
        if "team_abbr" in updates:
            assignments.append("team_abbr = ?")
            value = (updates["team_abbr"] or "").strip().upper()
            params.append(value or None)
        if "tags" in updates:
            assignments.append("tags = ?")
            params.append(json.dumps(updates["tags"] or []))
        if "pinned" in updates:
            assignments.append("pinned = ?")
            params.append(1 if updates["pinned"] else 0)

        assignments.append("updated_at = ?")
        params.append(datetime.now().timestamp())
        params.append(note_id)

        cursor = self.conn.cursor()
        cursor.execute(f"""
            UPDATE staff_notes
            SET {", ".join(assignments)}
            WHERE id = ?
        """, params)
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_staff_note(self, note_id: int) -> bool:
        """Remove a staff note."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM staff_notes WHERE id = ?", (note_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_staff_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single staff note."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM staff_notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_staff_notes(self, team_abbr: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
        """List staff notes optionally filtered by team."""
        cursor = self.conn.cursor()
        params = []
        query = """
            SELECT id, title, body, team_abbr, tags, pinned, author_id, author_name, created_at, updated_at
            FROM staff_notes
        """
        if team_abbr:
            query += " WHERE team_abbr IS NULL OR team_abbr = ?"
            params.append(team_abbr.strip().upper())
        query += " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        notes = []
        for row in rows:
            data = dict(row)
            try:
                data["tags"] = json.loads(data.get("tags") or "[]")
            except json.JSONDecodeError:
                data["tags"] = []
            notes.append(data)
        return notes

    def create_player_document(self, player_id: int, filename: str, path: str,
                               uploaded_by: Optional[int],
                               category: Optional[str] = None,
                               series_opponent: Optional[str] = None,
                               series_label: Optional[str] = None,
                               series_start: Optional[float] = None,
                               series_end: Optional[float] = None) -> int:
        """Store metadata for an uploaded player document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO player_documents (
                player_id, filename, path, uploaded_by, uploaded_at, category,
                series_opponent, series_label, series_start, series_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(player_id),
            filename,
            path,
            uploaded_by,
            datetime.now().timestamp(),
            (category or "").strip().lower() or None,
            (series_opponent or "").strip().upper() or None,
            (series_label or "").strip() or None,
            float(series_start) if series_start is not None else None,
            float(series_end) if series_end is not None else None
        ))
        self.conn.commit()
        return cursor.lastrowid

    def delete_player_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Delete a player document and return the deleted record."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, player_id, filename, path, uploaded_by, uploaded_at,
                   category, series_opponent, series_label, series_start, series_end
            FROM player_documents
            WHERE id = ?
        """, (doc_id,))
        row = cursor.fetchone()
        if not row:
            return None
        cursor.execute("DELETE FROM player_documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return dict(row)

    def list_player_documents(self, player_id: int,
                              category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List uploaded documents for a player."""
        cursor = self.conn.cursor()
        params: List[Any] = [int(player_id)]
        category_clause = "category IS NULL"
        if category is not None:
            category_clause = "category = ?"
            params.append((category or "").strip().lower())
        cursor.execute(f"""
            SELECT id, player_id, filename, path, uploaded_by, uploaded_at,
                   category, series_opponent, series_label, series_start, series_end
            FROM player_documents
            WHERE player_id = ?
              AND {category_clause}
            ORDER BY uploaded_at DESC
        """, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_latest_player_document_by_category(self, player_id: int,
                                               category: str) -> Optional[Dict[str, Any]]:
        """Return newest document for a player within a category."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, player_id, filename, path, uploaded_by, uploaded_at,
                   category, series_opponent, series_label, series_start, series_end
            FROM player_documents
            WHERE player_id = ? AND category = ?
            ORDER BY uploaded_at DESC
            LIMIT 1
        """, (int(player_id), (category or "").strip().lower()))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_documents_by_category(self, category: str,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """List most recent documents by category."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, player_id, filename, path, uploaded_by, uploaded_at,
                   category, series_opponent, series_label, series_start, series_end
            FROM player_documents
            WHERE category = ?
            ORDER BY uploaded_at DESC
            LIMIT ?
        """, ((category or "").strip().lower(), int(limit)))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_latest_document_by_category(self, category: str) -> Optional[Dict[str, Any]]:
        """Return the newest document in a category."""
        docs = self.list_documents_by_category(category, limit=1)
        return docs[0] if docs else None

    def get_player_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single player document entry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, player_id, filename, path, uploaded_by, uploaded_at,
                   category, series_opponent, series_label, series_start, series_end
            FROM player_documents
            WHERE id = ?
        """, (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_expired_player_documents(self, reference_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return documents whose series window has elapsed."""
        cursor = self.conn.cursor()
        if reference_ts is None:
            reference_ts = datetime.now().timestamp()
        cursor.execute("""
            SELECT id, player_id, filename, path, uploaded_by, uploaded_at,
                   category,
                   series_opponent, series_label, series_start, series_end
            FROM player_documents
            WHERE series_end IS NOT NULL AND series_end <= ?
        """, (reference_ts,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def record_player_document_event(self, player_id: int, filename: str, action: str,
                                     performed_by: Optional[int]) -> int:
        """Log document actions such as upload/delete."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO player_document_log (player_id, filename, action, performed_by, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            int(player_id),
            filename,
            action,
            performed_by,
            datetime.now().timestamp()
        ))
        self.conn.commit()
        return cursor.lastrowid

    def list_player_document_events(self, player_id: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """Return document activity, optionally filtered by player."""
        cursor = self.conn.cursor()
        params = []
        query = """
            SELECT id, player_id, filename, action, performed_by, timestamp
            FROM player_document_log
        """
        if player_id:
            query += " WHERE player_id = ?"
            params.append(int(player_id))
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # ---------------------------
    # Journal entry helpers
    # ---------------------------

    def upsert_journal_entry(self, user_id: int, entry_date: str, visibility: str,
                             title: Optional[str], body: Optional[str]) -> int:
        """Create or update a journal entry for a user."""
        if not entry_date:
            raise ValueError("entry_date is required")

        normalized_visibility = (visibility or "private").strip().lower()
        if normalized_visibility not in {"public", "private"}:
            raise ValueError(f"Invalid visibility: {visibility}")

        sanitized_date = entry_date.strip()
        if len(sanitized_date) != 10:
            raise ValueError("entry_date must be formatted as YYYY-MM-DD")

        now_ts = datetime.now().timestamp()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO journal_entries (user_id, entry_date, visibility, title, body, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, entry_date, visibility) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                updated_at = excluded.updated_at
        """, (
            int(user_id),
            sanitized_date,
            normalized_visibility,
            (title or "").strip() or None,
            (body or "").strip() or "",
            now_ts,
            now_ts,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_journal_entry(self, user_id: int, entry_date: str,
                          visibility: str) -> Optional[Dict[str, Any]]:
        """Fetch a journal entry for a specific date and visibility."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, user_id, entry_date, visibility, title, body, created_at, updated_at
            FROM journal_entries
            WHERE user_id = ? AND entry_date = ? AND visibility = ?
        """, (int(user_id), entry_date.strip(), (visibility or "").strip().lower()))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_journal_entries(self, user_id: int,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             visibility: Optional[str] = None,
                             limit: int = 180) -> List[Dict[str, Any]]:
        """List journal entries for a user ordered newest first."""
        cursor = self.conn.cursor()
        clauses = ["user_id = ?"]
        params: List[Any] = [int(user_id)]

        if visibility:
            normalized_visibility = (visibility or "").strip().lower()
            clauses.append("visibility = ?")
            params.append(normalized_visibility)

        if start_date:
            clauses.append("entry_date >= ?")
            params.append(start_date.strip())

        if end_date:
            clauses.append("entry_date <= ?")
            params.append(end_date.strip())

        query = f"""
            SELECT id, user_id, entry_date, visibility, title, body, created_at, updated_at
            FROM journal_entries
            WHERE {' AND '.join(clauses)}
            ORDER BY entry_date DESC, visibility ASC
            LIMIT ?
        """
        params.append(int(limit) if limit and limit > 0 else 180)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def delete_journal_entry(self, entry_id: int, user_id: int) -> bool:
        """Remove a journal entry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM journal_entries
            WHERE id = ? AND user_id = ?
        """, (int(entry_id), int(user_id)))
        self.conn.commit()
        return cursor.rowcount > 0

