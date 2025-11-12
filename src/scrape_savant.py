from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd
from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup
import statsapi

CACHE_DIR = Path("build/cache/savant")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _hash_key(*parts) -> Path:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return CACHE_DIR / f"{h}.parquet"

def fetch_pitcher_statcast(pitcher_id: int, start: str, end: str) -> pd.DataFrame:
    key = _hash_key("pitcher", pitcher_id, start, end)
    if key.exists():
        return pd.read_parquet(key)
    df = statcast_pitcher(start, end, pitcher_id)
    if df is None:
        df = pd.DataFrame()
    df.to_parquet(key, index=False)
    return df

def lookup_batter_id(name: str) -> int:
    """
    Look up MLBAM ID for a batter by name.
    Tries multiple strategies to find the player.
    """
    # Clean the name
    name = name.strip()
    
    # First try: exact name match with statsapi
    try:
        people = statsapi.lookup_player(name)
    except Exception as e:
        # If API call fails, try variations
        people = []
    
    # If statsapi didn't work, try pybaseball as fallback (more reliable)
    if not people:
        try:
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_name = parts[0]
                # Try pybaseball lookup
                result = playerid_lookup(last_name, first_name)
                if not result.empty and 'key_mlbam' in result.columns:
                    mlbam_id = result.iloc[0]['key_mlbam']
                    if pd.notna(mlbam_id):
                        return int(mlbam_id)
        except Exception:
            pass
    
    # If no results, try some common variations
    if not people:
        # Try with different case variations
        variations = [
            name.title(),  # Title case: "Pete Alonso"
            name.lower(),  # Lower case: "pete alonso"
            name.upper(),  # Upper case: "PETE ALONSO"
        ]
        
        # Try common name variations (e.g., "Pete" -> "Peter")
        parts = name.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            # Common nickname variations
            if first.lower() == "pete":
                variations.append(f"Peter {last}")
            elif first.lower() == "peter":
                variations.append(f"Pete {last}")
            elif first.lower() == "jim":
                variations.append(f"James {last}")
            elif first.lower() == "james":
                variations.append(f"Jim {last}")
            elif first.lower() == "mike":
                variations.append(f"Michael {last}")
            elif first.lower() == "michael":
                variations.append(f"Mike {last}")
            elif first.lower() == "bob":
                variations.append(f"Robert {last}")
            elif first.lower() == "robert":
                variations.append(f"Bob {last}")
        
        for variation in variations:
            if variation != name:  # Don't retry the original
                try:
                    people = statsapi.lookup_player(variation)
                    if people:
                        break
                except Exception:
                    continue
    
    # If still no results, try a partial match (last name only)
    if not people:
        parts = name.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            try:
                people = statsapi.lookup_player(last_name)
                # Filter to exact last name matches if multiple results
                if people:
                    filtered = [p for p in people if last_name.lower() in p.get("fullName", "").lower()]
                    if filtered:
                        people = filtered
            except Exception:
                pass
    
    if not people:
        # Try with common MLB player name patterns
        # Some players might be listed differently in the API
        # Last resort: try searching with just last name
        parts = name.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            # Try with "lastname, firstname" format
            first_name = parts[0]
            try:
                people = statsapi.lookup_player(f"{last_name}, {first_name}")
            except Exception:
                pass
    
    # Final fallback: try pybaseball with variations if statsapi still didn't work
    if not people:
        try:
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_name = parts[0]
                
                # Try pybaseball with original name
                result = playerid_lookup(last_name, first_name)
                if not result.empty and 'key_mlbam' in result.columns:
                    mlbam_id = result.iloc[0]['key_mlbam']
                    if pd.notna(mlbam_id):
                        return int(mlbam_id)
                
                # Try common nickname variations with pybaseball
                if first_name.lower() == "pete":
                    result = playerid_lookup(last_name, "peter")
                    if not result.empty and 'key_mlbam' in result.columns:
                        mlbam_id = result.iloc[0]['key_mlbam']
                        if pd.notna(mlbam_id):
                            return int(mlbam_id)
        except Exception:
            pass
    
    if not people:
        raise ValueError(
            f"Could not locate MLBAM id for hitter: {name}\n"
            f"Tried variations: {name}, {name.title()}, {name.lower()}\n"
            f"Please check the spelling or try a different name format."
        )
    
    # If multiple results, try to find the best match
    if len(people) > 1:
        # Try to find exact match first
        name_lower = name.lower()
        exact_match = next((p for p in people if p.get("fullName", "").lower() == name_lower), None)
        if exact_match:
            return int(exact_match["id"])
        
        # Try partial match (first and last name match)
        parts = name.split()
        if len(parts) >= 2:
            first, last = parts[0].lower(), parts[-1].lower()
            partial_match = next(
                (p for p in people 
                 if first in p.get("fullName", "").lower() and last in p.get("fullName", "").lower()),
                None
            )
            if partial_match:
                return int(partial_match["id"])
    
    # Return the first result
    return int(people[0]["id"])

def fetch_batter_statcast(batter_id: int, start: str, end: str) -> pd.DataFrame:
    key = _hash_key("batter", batter_id, start, end)
    if key.exists():
        return pd.read_parquet(key)
    df = statcast_batter(start, end, batter_id)
    if df is None:
        df = pd.DataFrame()
    df.to_parquet(key, index=False)
    return df
