# src/cache.py
import json, hashlib, sqlite3, time
from pathlib import Path

class SnapshotCache:
    def __init__(self, root="build/cache"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root/"index.sqlite"); self._init()

    def _init(self):
        self.db.execute("""CREATE TABLE IF NOT EXISTS snapshots(
            key TEXT PRIMARY KEY, ts REAL, path TEXT, ttl REAL
        )"""); self.db.commit()

    def _key(self, url, params=None):
        raw = url + "|" + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, url, params=None):
        k = self._key(url, params); row = self.db.execute(
            "SELECT path, ts, ttl FROM snapshots WHERE key=?", (k,)
        ).fetchone()
        if not row: return None
        path, ts, ttl = row
        if ttl and (time.time() - ts) > ttl: return None
        return Path(path).read_bytes()

    def put(self, url, params, content:bytes, ttl=60*60*24):
        k = self._key(url, params)
        p = self.root/f"{k}.bin"; p.write_bytes(content)
        self.db.execute("REPLACE INTO snapshots(key, ts, path, ttl) VALUES(?,?,?,?)",
                        (k, time.time(), str(p), ttl))
        self.db.commit()
