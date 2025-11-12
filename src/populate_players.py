# src/populate_players.py
"""
Populate player database from Sportradar API
"""
import os
import sys
import asyncio
from pathlib import Path

from sportradar_client import SportradarClient
from database import PlayerDB
from cache import SnapshotCache

# Hitter positions to include
HITTER_POSITIONS = ['OF', '1B', '2B', '3B', 'SS', 'C', 'DH', 'INF', 'IF']

async def populate_teams(client: SportradarClient, db: PlayerDB):
    """Fetch and store all MLB teams"""
    print("Fetching teams...")
    teams = await client.get_teams()
    
    if not teams:
        print("Warning: No teams found. API response format may be different.")
        return
    
    print(f"Found {len(teams)} teams")
    for team in teams:
        # Handle different response formats
        team_data = {}
        if isinstance(team, dict):
            team_data = team
        else:
            print(f"Unexpected team format: {type(team)}")
            continue
        
        # Try to extract team info from various possible field names
        team_info = {
            'id': team_data.get('id') or team_data.get('team_id'),
            'abbreviation': team_data.get('abbreviation') or team_data.get('abbr') or team_data.get('alias'),
            'name': team_data.get('name') or team_data.get('team_name'),
            'city': team_data.get('city') or team_data.get('market'),
            'league': team_data.get('league'),
            'division': team_data.get('division'),
        }
        
        if team_info['id']:
            db.upsert_team(team_info)
            print(f"  Stored team: {team_info.get('abbreviation')} - {team_info.get('name')}")
    
    print(f"Stored {len(teams)} teams\n")

def is_hitter(position: str) -> bool:
    """Check if position is a hitter position"""
    if not position:
        return False
    position = position.upper()
    # Check if any hitter position is in the position string
    return any(pos in position for pos in HITTER_POSITIONS)

async def populate_players_from_team(client: SportradarClient, db: PlayerDB, team_id: str, team_name: str):
    """Fetch and store players from a team roster"""
    try:
        roster_data = await client.get_team_roster(team_id)
        
        if not roster_data:
            print(f"  No roster data for {team_name}")
            return 0
        
        # Handle different roster response formats
        players = []
        if isinstance(roster_data, dict):
            # Try common roster fields
            if 'players' in roster_data:
                players = roster_data['players']
            elif 'roster' in roster_data:
                players = roster_data['roster']
            elif 'team' in roster_data and 'players' in roster_data['team']:
                players = roster_data['team']['players']
            else:
                # Might be a list at root level
                players = [roster_data] if not isinstance(roster_data.get('id'), str) else []
        elif isinstance(roster_data, list):
            players = roster_data
        
        if not players:
            print(f"  No players found in roster for {team_name}")
            return 0
        
        count = 0
        for player_data in players:
            # Handle different player data formats
            player = {}
            if isinstance(player_data, dict):
                player = player_data
            else:
                continue
            
            # Extract position
            position = (player.get('position') or 
                       player.get('primary_position') or
                       player.get('positions', {}).get('primary') if isinstance(player.get('positions'), dict) else None)
            
            # Only include hitters
            if not is_hitter(position):
                continue
            
            # Extract player ID
            player_id = (player.get('id') or 
                        player.get('player_id') or
                        player.get('sportradar_id'))
            
            if not player_id:
                print(f"  Warning: Player missing ID: {player.get('name')}")
                continue
            
            # Add team info to player data
            player['team_id'] = team_id
            player['team_abbr'] = (roster_data.get('team', {}).get('abbreviation') if isinstance(roster_data, dict) 
                                  else None) or team_name
            
            # Store player
            try:
                db.upsert_player(player)
                count += 1
            except Exception as e:
                print(f"  Error storing player {player_id}: {e}")
                continue
        
        return count
    except Exception as e:
        print(f"  Error fetching roster for {team_name}: {e}")
        return 0

async def populate_all_players(client: SportradarClient, db: PlayerDB):
    """Fetch and store all hitters from all teams"""
    print("Fetching all players...")
    
    # Get all teams
    teams = await client.get_teams()
    if not teams:
        print("Error: Could not fetch teams")
        return
    
    total_players = 0
    for team in teams:
        if isinstance(team, dict):
            team_id = team.get('id') or team.get('team_id')
            team_name = team.get('abbreviation') or team.get('name') or 'Unknown'
            
            if team_id:
                print(f"Fetching roster for {team_name}...")
                count = await populate_players_from_team(client, db, team_id, team_name)
                total_players += count
                print(f"  Stored {count} hitters from {team_name}")
    
    print(f"\nTotal players stored: {total_players}")

async def populate_player_stats(client: SportradarClient, db: PlayerDB, player_id: str, season: str = "2024"):
    """Fetch and store stats for a player"""
    try:
        stats_data = await client.get_player_statistics(player_id, season)
        
        if not stats_data:
            return
        
        # Extract season stats from various possible formats
        season_stats = {}
        if isinstance(stats_data, dict):
            # Try common stat fields
            if 'seasons' in stats_data:
                # Multiple seasons
                for season_data in stats_data['seasons']:
                    if season_data.get('year') == season or season_data.get('season') == season:
                        season_stats = season_data.get('statistics', {}) or season_data
                        break
            elif 'statistics' in stats_data:
                season_stats = stats_data['statistics']
            elif 'stats' in stats_data:
                season_stats = stats_data['stats']
            elif 'batting' in stats_data:
                season_stats = stats_data['batting']
            else:
                # Stats might be at root level
                season_stats = stats_data
        
        if season_stats:
            db.upsert_player_season(player_id, season, season_stats)
    except Exception as e:
        print(f"Error fetching stats for player {player_id}: {e}")

async def main():
    """Main population function"""
    # Check for API key
    api_key = os.environ.get("SPORTRADAR_API_KEY")
    if not api_key:
        print("Error: SPORTRADAR_API_KEY environment variable not set")
        print("Set it with: export SPORTRADAR_API_KEY=your_key")
        return
    
    # Initialize client and database
    cache = SnapshotCache(root="build/cache/sportradar")
    client = SportradarClient(api_key=api_key, cache=cache)
    db = PlayerDB()
    
    try:
        # Populate teams
        await populate_teams(client, db)
        
        # Populate players
        await populate_all_players(client, db)
        
        print("\nDatabase population complete!")
        print(f"Total players in database: {db.count_players()}")
        
    except Exception as e:
        print(f"Error during population: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())

