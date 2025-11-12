#!/usr/bin/env python3
"""Add test players to database for testing"""
import sys
sys.path.insert(0, 'src')

from database import PlayerDB

# Test players to add
test_players = [
    {
        'sportradar_id': 'test-alonso-001',
        'mlbam_id': '624413',
        'name': 'Pete Alonso',
        'first_name': 'Pete',
        'last_name': 'Alonso',
        'position': '1B',
        'primary_position': '1B',
        'team_abbr': 'NYM',
        'jersey_number': '20',
        'handedness': 'R',
        'height': "6'3\"",
        'weight': 245,
    },
    {
        'sportradar_id': 'test-bader-001',
        'mlbam_id': '608070',
        'name': 'Harrison Bader',
        'first_name': 'Harrison',
        'last_name': 'Bader',
        'position': 'OF',
        'primary_position': 'OF',
        'team_abbr': 'NYM',
        'jersey_number': '22',
        'handedness': 'R',
        'height': "6'0\"",
        'weight': 195,
    },
    {
        'sportradar_id': 'test-judge-001',
        'mlbam_id': '592450',
        'name': 'Aaron Judge',
        'first_name': 'Aaron',
        'last_name': 'Judge',
        'position': 'OF',
        'primary_position': 'OF',
        'team_abbr': 'NYY',
        'jersey_number': '99',
        'handedness': 'R',
        'height': "6'7\"",
        'weight': 282,
    },
    {
        'sportradar_id': 'test-ohtani-001',
        'mlbam_id': '660271',
        'name': 'Shohei Ohtani',
        'first_name': 'Shohei',
        'last_name': 'Ohtani',
        'position': 'DH',
        'primary_position': 'DH',
        'team_abbr': 'LAD',
        'jersey_number': '17',
        'handedness': 'L',
        'height': "6'4\"",
        'weight': 210,
    },
    {
        'sportradar_id': 'test-acuna-001',
        'mlbam_id': '660670',
        'name': 'Ronald Acuña Jr.',
        'first_name': 'Ronald',
        'last_name': 'Acuña',
        'position': 'OF',
        'primary_position': 'OF',
        'team_abbr': 'ATL',
        'jersey_number': '13',
        'handedness': 'R',
        'height': "6'0\"",
        'weight': 205,
    },
]

def add_test_players():
    db = PlayerDB()
    
    print("Adding test players to database...")
    for player in test_players:
        try:
            db.upsert_player(player)
            print(f"  Added: {player['name']}")
        except Exception as e:
            print(f"  Error adding {player['name']}: {e}")
    
    count = db.count_players()
    print(f"\nTotal players in database: {count}")
    db.close()

if __name__ == "__main__":
    add_test_players()

