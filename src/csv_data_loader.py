# src/csv_data_loader.py
"""
CSV Data Loader for Player Statistics
Loads and searches data from fangraphs.csv, Positions.csv, and statscast.csv
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

class CSVDataLoader:
    """Load and search player data from CSV files"""
    
    def __init__(self, base_path: str = None):
        """
        Initialize the CSV data loader
        
        Args:
            base_path: Base path to the project directory. If None, assumes current directory.
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent
        else:
            base_path = Path(base_path)
        
        self.fangraphs_path = base_path / "fangraphs.csv"
        self.fangraphs_pitchers_path = base_path / "fangraphs_pitchers.csv"
        self.positions_path = base_path / "Positions.csv"
        self.statscast_path = base_path / "statscast.csv"
        
        self._fangraphs_df = None
        self._fangraphs_pitchers_df = None
        self._positions_df = None
        self._statscast_df = None
    
    def _load_fangraphs(self):
        """Lazy load fangraphs.csv"""
        if self._fangraphs_df is None and self.fangraphs_path.exists():
            self._fangraphs_df = pd.read_csv(self.fangraphs_path)
        return self._fangraphs_df
    
    def _load_fangraphs_pitchers(self):
        """Lazy load fangraphs_pitchers.csv"""
        if self._fangraphs_pitchers_df is None and self.fangraphs_pitchers_path.exists():
            self._fangraphs_pitchers_df = pd.read_csv(self.fangraphs_pitchers_path)
        return self._fangraphs_pitchers_df
    
    def _load_positions(self):
        """Lazy load Positions.csv"""
        if self._positions_df is None and self.positions_path.exists():
            self._positions_df = pd.read_csv(self.positions_path)
        return self._positions_df
    
    def _load_statscast(self):
        """Lazy load statscast.csv"""
        if self._statscast_df is None and self.statscast_path.exists():
            # Handle quoted column names - the first column is "last_name, first_name"
            self._statscast_df = pd.read_csv(self.statscast_path)
            # Clean up column names - remove quotes if present
            self._statscast_df.columns = [col.strip('"') for col in self._statscast_df.columns]
        return self._statscast_df
    
    def search_players(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for players by name across all CSV files
        
        Args:
            search_term: Player name to search for
            
        Returns:
            List of matching players with their basic info
        """
        if not search_term or len(search_term.strip()) < 2:
            return []
        
        search_term = search_term.strip().lower()
        players = {}
        
        # Search in fangraphs
        fg_df = self._load_fangraphs()
        if fg_df is not None and 'Name' in fg_df.columns:
            matches = fg_df[fg_df['Name'].str.lower().str.contains(search_term, na=False)]
            for _, row in matches.iterrows():
                name = row['Name']
                if name not in players:
                    players[name] = {
                        'name': name,
                        'seasons': []
                    }
                if 'Season' in row:
                    players[name]['seasons'].append(row['Season'])
        
        # Search in positions
        pos_df = self._load_positions()
        if pos_df is not None and 'player_name' in pos_df.columns:
            matches = pos_df[pos_df['player_name'].str.lower().str.contains(search_term, na=False)]
            for _, row in matches.iterrows():
                name = row['player_name']
                if name not in players:
                    players[name] = {
                        'name': name,
                        'seasons': []
                    }
        
        # Search in fangraphs_pitchers
        fg_pitchers_df = self._load_fangraphs_pitchers()
        if fg_pitchers_df is not None and 'Name' in fg_pitchers_df.columns:
            matches = fg_pitchers_df[fg_pitchers_df['Name'].str.lower().str.contains(search_term, na=False)]
            for _, row in matches.iterrows():
                name = row['Name']
                if name not in players:
                    players[name] = {
                        'name': name,
                        'seasons': []
                    }
                if 'Season' in row:
                    players[name]['seasons'].append(row['Season'])
        
        # Search in statscast
        sc_df = self._load_statscast()
        if sc_df is not None:
            # Handle quoted column name - typically "last_name, first_name"
            name_col = None
            for col in sc_df.columns:
                col_lower = col.lower()
                if 'last_name' in col_lower or 'first_name' in col_lower or (col == '"last_name, first_name"'):
                    name_col = col
                    break
            
            if name_col:
                # Handle cases where name might be in "Last, First" format
                try:
                    matches = sc_df[sc_df[name_col].astype(str).str.lower().str.contains(search_term, na=False)]
                    for _, row in matches.iterrows():
                        name = str(row[name_col])
                        if name not in players:
                            players[name] = {
                                'name': name,
                                'seasons': []
                            }
                except Exception as e:
                    # If search fails, continue without this data source
                    pass
        
        return list(players.values())
    
    def get_player_data(self, player_name: str) -> Dict[str, Any]:
        """
        Get all data for a specific player from all CSV files
        
        Args:
            player_name: Exact player name
            
        Returns:
            Dictionary with all player data from all sources
        """
        result = {
            'name': player_name,
            'fangraphs': [],
            'positions': [],
            'statscast': []
        }
        
        # Get fangraphs data
        fg_df = self._load_fangraphs()
        if fg_df is not None and 'Name' in fg_df.columns:
            # Normalize names for comparison (strip whitespace, handle case)
            player_name_normalized = player_name.strip().lower()
            # Use contains first for partial matches, then exact match
            fg_data = fg_df[fg_df['Name'].astype(str).str.strip().str.lower() == player_name_normalized]
            if fg_data.empty:
                # Try partial match as fallback
                fg_data = fg_df[fg_df['Name'].astype(str).str.strip().str.lower().str.contains(player_name_normalized, na=False, regex=False)]
            if not fg_data.empty:
                # Convert to list of dictionaries, handling NaN values
                for _, row in fg_data.iterrows():
                    row_dict = row.to_dict()
                    # Convert NaN to None for JSON serialization
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            row_dict[key] = None
                    result['fangraphs'].append(row_dict)
        
        # Get fangraphs_pitchers data
        fg_pitchers_df = self._load_fangraphs_pitchers()
        if fg_pitchers_df is not None and 'Name' in fg_pitchers_df.columns:
            player_name_normalized = player_name.strip().lower()
            fg_pitchers_data = fg_pitchers_df[fg_pitchers_df['Name'].astype(str).str.strip().str.lower() == player_name_normalized]
            if fg_pitchers_data.empty:
                # Try partial match as fallback
                fg_pitchers_data = fg_pitchers_df[fg_pitchers_df['Name'].astype(str).str.strip().str.lower().str.contains(player_name_normalized, na=False, regex=False)]
            if not fg_pitchers_data.empty:
                # Sort by season descending to get most recent data first
                if 'Season' in fg_pitchers_data.columns:
                    fg_pitchers_data = fg_pitchers_data.sort_values('Season', ascending=False)
                
                # Get most recent team for positions data
                most_recent_team = None
                if 'Season' in fg_pitchers_data.columns and len(fg_pitchers_data) > 0:
                    first_row = fg_pitchers_data.iloc[0]
                    most_recent_team = first_row.get('fg_Team') if 'fg_Team' in fg_pitchers_data.columns else None
                    if pd.isna(most_recent_team):
                        most_recent_team = first_row.get('Team') if 'Team' in fg_pitchers_data.columns else None
                    if pd.isna(most_recent_team):
                        most_recent_team = None
                
                # Convert to list of dictionaries, handling NaN values
                for _, row in fg_pitchers_data.iterrows():
                    row_dict = row.to_dict()
                    # Convert NaN to None for JSON serialization
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            row_dict[key] = None
                    # Add pitcher data to fangraphs array (merge with hitter data)
                    result['fangraphs'].append(row_dict)
                
                # Create positions data entry for pitchers if not already present
                if not result['positions']:
                    # Use most recent team or try to get from any row
                    team = most_recent_team
                    if not team and len(fg_pitchers_data) > 0:
                        first_row_dict = fg_pitchers_data.iloc[0].to_dict()
                        team = first_row_dict.get('fg_Team') or first_row_dict.get('Team')
                    
                    # Create a positions entry
                    position_entry = {
                        'player_name': player_name,
                        'Player Name': player_name,
                        'team_name': team,
                        'Team Name': team,
                        'Team': team,
                        'position_name': 'Pitcher',
                        'Position Name': 'Pitcher',
                        'Position': 'Pitcher',
                        'Pos': 'P'
                    }
                    result['positions'].append(position_entry)
        
        # Get positions data
        pos_df = self._load_positions()
        if pos_df is not None and 'player_name' in pos_df.columns:
            pos_data = pos_df[pos_df['player_name'].str.lower() == player_name.lower()]
            if not pos_data.empty:
                for _, row in pos_data.iterrows():
                    row_dict = row.to_dict()
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            row_dict[key] = None
                    result['positions'].append(row_dict)
        
        # Get statscast data
        sc_df = self._load_statscast()
        if sc_df is not None:
            # Find name column
            name_col = None
            for col in sc_df.columns:
                col_lower = col.lower()
                if 'last_name' in col_lower or 'first_name' in col_lower or (col == '"last_name, first_name"'):
                    name_col = col
                    break
            
            if name_col:
                try:
                    # Try to match the player name - handle "Last, First" format
                    sc_data = sc_df[sc_df[name_col].astype(str).str.lower().str.contains(player_name.lower(), na=False)]
                    if not sc_data.empty:
                        for _, row in sc_data.iterrows():
                            row_dict = row.to_dict()
                            for key, value in row_dict.items():
                                if pd.isna(value):
                                    row_dict[key] = None
                            result['statscast'].append(row_dict)
                except Exception as e:
                    # If matching fails, continue without this data source
                    pass
        
        return result
    
    def get_all_players_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all players with basic info for filtering"""
        players = {}
        
        # Get from fangraphs
        fg_df = self._load_fangraphs()
        if fg_df is not None and 'Name' in fg_df.columns:
            # Process players even if Team column doesn't exist
            for _, row in fg_df.iterrows():
                name = row['Name']
                if pd.isna(name) or name == '':
                    continue
                if name not in players:
                    players[name] = {
                        'name': name,
                        'team': row.get('Team', None) if 'Team' in fg_df.columns else None,
                        'position': None,
                        'seasons': set()
                    }
                if 'Season' in row and pd.notna(row['Season']):
                    try:
                        season_int = int(row['Season'])
                        players[name]['seasons'].add(season_int)
                    except (ValueError, TypeError):
                        pass
        
        # Get from fangraphs_pitchers
        fg_pitchers_df = self._load_fangraphs_pitchers()
        if fg_pitchers_df is not None and 'Name' in fg_pitchers_df.columns:
            for _, row in fg_pitchers_df.iterrows():
                name = row['Name']
                if pd.isna(name) or name == '':
                    continue
                if name not in players:
                    # Get team from fg_Team column
                    team = row.get('fg_Team', None) if 'fg_Team' in fg_pitchers_df.columns else None
                    players[name] = {
                        'name': name,
                        'team': team,
                        'position': 'Pitcher',
                        'seasons': set()
                    }
                elif players[name]['position'] is None:
                    # If player already exists but no position, set to Pitcher
                    players[name]['position'] = 'Pitcher'
                # Update team if not set
                if players[name]['team'] is None:
                    team = row.get('fg_Team', None) if 'fg_Team' in fg_pitchers_df.columns else None
                    if team:
                        players[name]['team'] = team
                if 'Season' in row and pd.notna(row['Season']):
                    try:
                        season_int = int(row['Season'])
                        players[name]['seasons'].add(season_int)
                    except (ValueError, TypeError):
                        pass
        
        # Get position info
        pos_df = self._load_positions()
        if pos_df is not None and 'player_name' in pos_df.columns:
            for _, row in pos_df.iterrows():
                name = row['player_name']
                if name in players:
                    if 'position_name' in row:
                        players[name]['position'] = row['position_name']
                    elif 'Position Name' in row:
                        players[name]['position'] = row['Position Name']
        
        # Convert sets to sorted lists
        result = []
        for name, data in players.items():
            result.append({
                'name': name,
                'team': data['team'],
                'position': data['position'],
                'seasons': sorted(list(data['seasons'])) if data['seasons'] else []
            })
        
        return sorted(result, key=lambda x: x['name'])
    
    def get_player_trends(self, player_name: str, stats: List[str] = None, 
                          season_start: int = None, season_end: int = None) -> Dict[str, Any]:
        """Get player performance trends over seasons"""
        fg_df = self._load_fangraphs()
        fg_pitchers_df = self._load_fangraphs_pitchers()
        
        player_data = None
        data_source = 'hitter'
        
        if fg_df is not None and 'Name' in fg_df.columns:
            hitter_data = fg_df[fg_df['Name'].str.lower() == player_name.lower()]
            if not hitter_data.empty:
                player_data = hitter_data
                data_source = 'hitter'
        
        if (player_data is None or player_data.empty) and fg_pitchers_df is not None and 'Name' in fg_pitchers_df.columns:
            pitcher_data = fg_pitchers_df[fg_pitchers_df['Name'].str.lower() == player_name.lower()]
            if not pitcher_data.empty:
                player_data = pitcher_data
                data_source = 'pitcher'
        
        if player_data is None or player_data.empty:
            return {'player': player_name, 'trends': []}
        
        # Filter by season range
        if 'Season' in player_data.columns:
            if season_start:
                player_data = player_data[player_data['Season'] >= season_start]
            if season_end:
                player_data = player_data[player_data['Season'] <= season_end]
        
        # Sort by season
        if 'Season' in player_data.columns:
            player_data = player_data.sort_values('Season')
        
        # Map user-friendly stat names to CSV column names
        stat_column_map = {
            # Basic Stats
            'AVG': 'fg_AVG',
            'OBP': 'fg_OBP',
            'SLG': 'fg_SLG',
            'OPS': 'fg_OPS',
            'HR': 'fg_HR',
            'RBI': 'fg_RBI',
            'R': 'fg_R',
            'H': 'fg_H',
            '2B': 'fg_2B',
            '3B': 'fg_3B',
            'XBH': None,  # Will calculate from 2B + 3B + HR
            'SB': 'fg_SB',
            # Advanced Metrics
            'wRC+': 'fg_wRC+',
            'WAR': 'fg_WAR',
            'wOBA': 'fg_wOBA',
            'ISO': 'fg_ISO',
            'BABIP': 'fg_BABIP',
            'wRAA': 'fg_wRAA',
            # Plate Discipline
            'BB': 'fg_BB',
            'BB%': 'fg_BB%',
            'K': 'fg_SO',  # Strikeouts are stored as SO in CSV
            'K%': 'fg_K%',
            'BB/K': 'fg_BB/K',
            'PA': 'fg_PA',
            'AB': 'fg_AB',
            # Batted Ball
            'GB%': 'fg_GB%',
            'FB%': 'fg_FB%',
            'LD%': 'fg_LD%',
            'GB/FB': 'fg_GB/FB',
            'HR/FB': 'fg_HR/FB',
            'Barrel%': 'fg_Barrel%',
            # Expected Stats
            'xBA': 'fg_xBA',
            'xSLG': 'fg_xSLG',
            'xwOBA': 'fg_xwOBA',
            # Pitching - Run Prevention
            'ERA': 'fg_ERA',
            'WHIP': 'fg_WHIP',
            'FIP': 'fg_FIP',
            'xFIP': 'fg_xFIP',
            'xERA': 'fg_xERA',
            'SIERA': 'fg_SIERA',
            # Pitching - Strikeout & Walk
            'K/9': 'fg_K/9',
            'BB/9': 'fg_BB/9',
            'K/BB': 'fg_K/BB',
            'K-BB%': 'fg_K-BB%',
            # Pitching - Volume & Results
            'IP': 'fg_IP',
            'G': 'fg_G',
            'GS': 'fg_GS',
            'SV': 'fg_SV',
            'W': 'fg_W',
            'L': 'fg_L',
            # Pitching - Index & Percentages
            'ERA-': 'fg_ERA-',
            'FIP-': 'fg_FIP-',
            'LOB%': 'fg_LOB%',
            'CSW%': 'fg_CSW%',
            'Pitches': 'fg_Pitches'
        }
        
        trends = []
        for _, row in player_data.iterrows():
            trend_point = {
                'season': int(row['Season']) if 'Season' in row and pd.notna(row['Season']) else None,
                'stats': {}
            }
            
            if stats:
                for stat in stats:
                    # Handle special calculated stats
                    if stat == 'XBH':
                        # Calculate XBH from 2B + 3B + HR
                        val_2b = row.get('fg_2B', 0) if 'fg_2B' in row and pd.notna(row.get('fg_2B')) else 0
                        val_3b = row.get('fg_3B', 0) if 'fg_3B' in row and pd.notna(row.get('fg_3B')) else 0
                        val_hr = row.get('fg_HR', 0) if 'fg_HR' in row and pd.notna(row.get('fg_HR')) else 0
                        xbh_value = val_2b + val_3b + val_hr
                        trend_point['stats'][stat] = float(xbh_value) if xbh_value > 0 else None
                        continue
                    
                    # Try direct match first, then mapped column name
                    column_name = stat
                    if stat in stat_column_map:
                        column_name = stat_column_map[stat]
                    
                    # If mapping is None, skip this stat
                    if column_name is None:
                        trend_point['stats'][stat] = None
                        continue
                    
                    # Try both the mapped column name and the original stat name
                    if column_name in row:
                        val = row[column_name]
                        trend_point['stats'][stat] = float(val) if pd.notna(val) else None
                    elif stat in row:
                        val = row[stat]
                        trend_point['stats'][stat] = float(val) if pd.notna(val) else None
                    else:
                        # Stat not found, set to None
                        trend_point['stats'][stat] = None
            else:
                # Include all numeric columns
                for col in player_data.columns:
                    if col not in ['Name', 'Season', 'Team']:
                        try:
                            val = row[col]
                            if pd.notna(val):
                                trend_point['stats'][col] = float(val)
                        except (ValueError, TypeError):
                            pass
            
            trends.append(trend_point)
        
        return {'player': player_name, 'trends': trends}
    
    def compare_players(self, player_names: List[str], stats: List[str],
                       season: int = None) -> Dict[str, Any]:
        """Compare multiple players across selected metrics"""
        fg_df = self._load_fangraphs()
        if fg_df is None or 'Name' not in fg_df.columns:
            return {'players': [], 'comparison': {}}
        
        comparison = {}
        players_found = []
        
        for player_name in player_names:
            player_data = fg_df[fg_df['Name'].str.lower() == player_name.lower()]
            
            if season:
                if 'Season' in player_data.columns:
                    player_data = player_data[player_data['Season'] == season]
            
            if not player_data.empty:
                players_found.append(player_name)
                comparison[player_name] = {}
                
                # Get latest season if no season specified
                if not season and 'Season' in player_data.columns:
                    latest_season = player_data['Season'].max()
                    player_data = player_data[player_data['Season'] == latest_season]
                
                for stat in stats:
                    if stat in player_data.columns:
                        val = player_data[stat].iloc[0] if len(player_data) > 0 else None
                        comparison[player_name][stat] = float(val) if pd.notna(val) else None
                    else:
                        comparison[player_name][stat] = None
        
        return {'players': players_found, 'comparison': comparison}
    
    def get_league_leaders(self, stat: str, limit: int = 10, season: int = None,
                          position: str = None, team: str = None) -> List[Dict[str, Any]]:
        """Get league leaders for a specific stat"""
        fg_df = self._load_fangraphs()
        if fg_df is None or 'Name' not in fg_df.columns:
            return []
        
        # Filter by season
        if season and 'Season' in fg_df.columns:
            fg_df = fg_df[fg_df['Season'] == season]
        
        # Filter by position (if available in positions data)
        if position:
            pos_df = self._load_positions()
            if pos_df is not None and 'player_name' in pos_df.columns:
                pos_col = 'position_name' if 'position_name' in pos_df.columns else 'Position Name'
                if pos_col in pos_df.columns:
                    matching_players = pos_df[pos_df[pos_col].str.lower() == position.lower()]['player_name'].tolist()
                    fg_df = fg_df[fg_df['Name'].isin(matching_players)]
        
        # Filter by team
        if team and 'Team' in fg_df.columns:
            fg_df = fg_df[fg_df['Team'].str.lower() == team.lower()]
        
        # Check if stat exists
        if stat not in fg_df.columns:
            return []
        
        # Remove rows with NaN values for the stat
        fg_df = fg_df[fg_df[stat].notna()]
        
        # Sort by stat value (descending)
        fg_df = fg_df.sort_values(stat, ascending=False)
        
        # Get top N
        leaders = []
        for _, row in fg_df.head(limit).iterrows():
            leaders.append({
                'name': row['Name'],
                'team': row.get('Team', None),
                'season': int(row['Season']) if 'Season' in row and pd.notna(row['Season']) else None,
                'value': float(row[stat])
            })
        
        return leaders
    
    def get_stat_distribution(self, stat: str, season: int = None,
                             position: str = None, team: str = None,
                             bins: int = 20) -> Dict[str, Any]:
        """Get statistical distribution for a metric"""
        fg_df = self._load_fangraphs()
        if fg_df is None or 'Name' not in fg_df.columns:
            return {'distribution': [], 'stats': {}}
        
        # Filter by season
        if season and 'Season' in fg_df.columns:
            fg_df = fg_df[fg_df['Season'] == season]
        
        # Filter by position
        if position:
            pos_df = self._load_positions()
            if pos_df is not None and 'player_name' in pos_df.columns:
                pos_col = 'position_name' if 'position_name' in pos_df.columns else 'Position Name'
                if pos_col in pos_df.columns:
                    matching_players = pos_df[pos_df[pos_col].str.lower() == position.lower()]['player_name'].tolist()
                    fg_df = fg_df[fg_df['Name'].isin(matching_players)]
        
        # Filter by team
        if team and 'Team' in fg_df.columns:
            fg_df = fg_df[fg_df['Team'].str.lower() == team.lower()]
        
        # Check if stat exists
        if stat not in fg_df.columns:
            return {'distribution': [], 'stats': {}}
        
        # Get values, removing NaN
        values = fg_df[stat].dropna().astype(float).tolist()
        
        if not values:
            return {'distribution': [], 'stats': {}}
        
        # Calculate statistics
        stats_summary = {
            'count': len(values),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
        
        # Create histogram bins
        hist, bin_edges = np.histogram(values, bins=bins)
        distribution = []
        for i in range(len(hist)):
            distribution.append({
                'bin_start': float(bin_edges[i]),
                'bin_end': float(bin_edges[i + 1]),
                'count': int(hist[i])
            })
        
        return {'distribution': distribution, 'stats': stats_summary}

