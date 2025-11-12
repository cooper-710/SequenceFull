# src/sportradar_client.py
"""
Sportradar MLB API Client
"""
import os
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from fetch import get_json, RateLimiter
    from cache import SnapshotCache
except ImportError:
    # Handle case where running from different directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from fetch import get_json, RateLimiter
    from cache import SnapshotCache

class SportradarClient:
    """Client for Sportradar MLB API"""
    
    def __init__(self, api_key: Optional[str] = None, cache: Optional[SnapshotCache] = None):
        """
        Initialize Sportradar client.
        
        Args:
            api_key: Sportradar API key. If None, reads from SPORTRADAR_API_KEY env var.
            cache: Optional cache instance for API responses
        """
        self.api_key = api_key or os.environ.get("SPORTRADAR_API_KEY")
        if not self.api_key:
            raise ValueError("Sportradar API key required. Set SPORTRADAR_API_KEY environment variable.")
        
        # Sportradar MLB API base URL - correct format for trial API
        self.base_url = "https://api.sportradar.com/mlb/trial/v7/en"
        
        self.cache = cache or SnapshotCache(root="build/cache/sportradar")
        self.rate_limiter = RateLimiter(rps=2.0)  # Conservative rate limit for Sportradar
    
    def _build_url(self, endpoint: str) -> str:
        """Build full API URL with API key"""
        # Sportradar trial API format: https://api.sportradar.com/mlb/trial/v7/en/{endpoint}.json?api_key={key}
        endpoint = endpoint.lstrip('/')
        if not endpoint.endswith('.json'):
            endpoint = endpoint + '.json'
        url = f"{self.base_url}/{endpoint}"
        if '?' in url:
            url += f"&api_key={self.api_key}"
        else:
            url += f"?api_key={self.api_key}"
        return url
    
    async def get_teams(self) -> List[Dict[str, Any]]:
        """
        Get all MLB teams.
        
        Returns:
            List of team dictionaries
        """
        # Try different endpoint formats
        endpoints = [
            "/league/teams",
            "/teams",
            "/seasons/2024/teams",
        ]
        
        for endpoint_path in endpoints:
            try:
                url = self._build_url(endpoint_path)
                data = await get_json(url, cache=self.cache, cache_ttl=60*60*24, rl=self.rate_limiter)
                
                if not data:
                    continue
                
                # Handle different response formats
                if isinstance(data, dict):
                    # Common format: {"teams": [...]} or {"league": {"teams": [...]}}
                    if "teams" in data:
                        return data["teams"]
                    elif "league" in data and "teams" in data["league"]:
                        return data["league"]["teams"]
                    elif "conferences" in data:
                        # Some APIs return conferences -> divisions -> teams
                        teams = []
                        for conf in data.get("conferences", []):
                            for div in conf.get("divisions", []):
                                teams.extend(div.get("teams", []))
                        if teams:
                            return teams
                    elif "season" in data and "teams" in data["season"]:
                        return data["season"]["teams"]
                    # Try direct list
                    for key in ["data", "results", "items"]:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                elif isinstance(data, list):
                    return data
            except Exception as e:
                # Try next endpoint
                continue
        
        return []
    
    async def get_team_roster(self, team_id: str) -> Dict[str, Any]:
        """
        Get team roster with all players.
        
        Args:
            team_id: Sportradar team ID
            
        Returns:
            Roster data dictionary
        """
        # Try different endpoint formats
        endpoints = [
            f"/teams/{team_id}/roster",
            f"/teams/{team_id}/players",
            f"/seasons/2024/teams/{team_id}/roster",
        ]
        
        for endpoint_path in endpoints:
            try:
                url = self._build_url(endpoint_path)
                data = await get_json(url, cache=self.cache, cache_ttl=60*60*24, rl=self.rate_limiter)
                if data:
                    return data
            except Exception as e:
                continue
        
        return {}
    
    async def get_player_profile(self, player_id: str) -> Dict[str, Any]:
        """
        Get detailed player profile.
        
        Args:
            player_id: Sportradar player ID
            
        Returns:
            Player profile dictionary
        """
        # Try different endpoint formats
        endpoints = [
            f"/players/{player_id}/profile",
            f"/players/{player_id}",
        ]
        
        for endpoint_path in endpoints:
            try:
                url = self._build_url(endpoint_path)
                data = await get_json(url, cache=self.cache, cache_ttl=60*60*12, rl=self.rate_limiter)
                if data:
                    return data
            except Exception as e:
                continue
        
        return {}
    
    async def get_player_statistics(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get player statistics.
        
        Args:
            player_id: Sportradar player ID
            season: Optional season year (e.g., "2024")
            
        Returns:
            Player statistics dictionary
        """
        # Try different endpoint formats
        if season:
            endpoints = [
                f"/players/{player_id}/statistics/{season}",
                f"/seasons/{season}/players/{player_id}/statistics",
            ]
        else:
            endpoints = [
                f"/players/{player_id}/statistics",
            ]
        
        for endpoint_path in endpoints:
            try:
                url = self._build_url(endpoint_path)
                data = await get_json(url, cache=self.cache, cache_ttl=60*60*6, rl=self.rate_limiter)
                if data:
                    return data
            except Exception as e:
                continue
        
        return {}
    
    def get_team_roster_sync(self, team_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for get_team_roster"""
        return asyncio.run(self.get_team_roster(team_id))
    
    def get_teams_sync(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for get_teams"""
        return asyncio.run(self.get_teams())
    
    def get_player_profile_sync(self, player_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for get_player_profile"""
        return asyncio.run(self.get_player_profile(player_id))
    
    def get_player_statistics_sync(self, player_id: str, season: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for get_player_statistics"""
        return asyncio.run(self.get_player_statistics(player_id, season))
