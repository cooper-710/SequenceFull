#!/usr/bin/env python3
"""
Backup player database data before clearing
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')
from database import PlayerDB

def backup_database():
    """Export all player data to JSON backup"""
    db = PlayerDB()
    cursor = db.conn.cursor()
    
    # Create backup directory
    backup_dir = Path("build/database/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped backup file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"player_data_backup_{timestamp}.json"
    
    backup_data = {
        "backup_date": datetime.now().isoformat(),
        "players": [],
        "player_seasons": [],
        "player_stats": [],
        "player_history": [],
        "teams": []
    }
    
    # Backup players
    cursor.execute("SELECT * FROM players")
    for row in cursor.fetchall():
        backup_data["players"].append(dict(row))
    
    # Backup player seasons
    cursor.execute("SELECT * FROM player_seasons")
    for row in cursor.fetchall():
        backup_data["player_seasons"].append(dict(row))
    
    # Backup player stats
    cursor.execute("SELECT * FROM player_stats")
    for row in cursor.fetchall():
        backup_data["player_stats"].append(dict(row))
    
    # Backup player history
    cursor.execute("SELECT * FROM player_history")
    for row in cursor.fetchall():
        backup_data["player_history"].append(dict(row))
    
    # Backup teams
    cursor.execute("SELECT * FROM teams")
    for row in cursor.fetchall():
        backup_data["teams"].append(dict(row))
    
    # Write backup file
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2, default=str)
    
    print(f"Backup created: {backup_file}")
    print(f"  - Players: {len(backup_data['players'])}")
    print(f"  - Player seasons: {len(backup_data['player_seasons'])}")
    print(f"  - Player stats: {len(backup_data['player_stats'])}")
    print(f"  - Player history: {len(backup_data['player_history'])}")
    print(f"  - Teams: {len(backup_data['teams'])}")
    
    db.close()
    return str(backup_file)

if __name__ == "__main__":
    backup_database()

