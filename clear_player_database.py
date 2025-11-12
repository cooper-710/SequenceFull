#!/usr/bin/env python3
"""
Clear all player data from database (keeping schema)
"""
import sys

sys.path.insert(0, 'src')
from database import PlayerDB

def clear_database():
    """Delete all player data, keeping schema intact"""
    db = PlayerDB()
    cursor = db.conn.cursor()
    
    # Get counts before deletion
    cursor.execute("SELECT COUNT(*) FROM players")
    player_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM player_seasons")
    season_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM player_stats")
    stats_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM player_history")
    history_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM teams")
    team_count = cursor.fetchone()[0]
    
    print(f"Clearing database...")
    print(f"  - Players: {player_count}")
    print(f"  - Player seasons: {season_count}")
    print(f"  - Player stats: {stats_count}")
    print(f"  - Player history: {history_count}")
    print(f"  - Teams: {team_count}")
    
    # Delete all data (order matters due to foreign keys)
    cursor.execute("DELETE FROM player_stats")
    cursor.execute("DELETE FROM player_seasons")
    cursor.execute("DELETE FROM player_history")
    cursor.execute("DELETE FROM players")
    cursor.execute("DELETE FROM teams")
    
    db.conn.commit()
    
    # Verify deletion
    cursor.execute("SELECT COUNT(*) FROM players")
    remaining_players = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM player_seasons")
    remaining_seasons = cursor.fetchone()[0]
    
    print(f"\nDatabase cleared!")
    print(f"  - Remaining players: {remaining_players}")
    print(f"  - Remaining seasons: {remaining_seasons}")
    
    db.close()

if __name__ == "__main__":
    clear_database()

