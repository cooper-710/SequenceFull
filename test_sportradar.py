#!/usr/bin/env python3
"""Test Sportradar API connection"""
import os
import asyncio
import sys
sys.path.insert(0, 'src')

from sportradar_client import SportradarClient

async def test_api():
    api_key = os.environ.get("SPORTRADAR_API_KEY", "Z2ZMn59qIGMtzMrDKQB9smyd2ANvPxj98FXZicIp")
    client = SportradarClient(api_key=api_key)
    
    # Try different endpoint formats
    print("Testing Sportradar API...")
    print(f"Base URL: {client.base_url}/{client.version}")
    print(f"API Key: {api_key[:10]}...")
    
    # Test 1: Current format
    try:
        print("\n1. Testing /league/teams endpoint...")
        teams = await client.get_teams()
        print(f"Success! Got {len(teams)} teams")
        return True
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 2: Try without version
    try:
        print("\n2. Testing without version...")
        client.version = ""
        teams = await client.get_teams()
        print(f"Success! Got {len(teams)} teams")
        return True
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 3: Try different endpoint
    try:
        print("\n3. Testing /teams endpoint...")
        client.version = "v7"
        url = client._build_url("/teams")
        print(f"URL: {url}")
        from fetch import get_json
        data = await get_json(url, cache=client.cache, cache_ttl=60*60*24, rl=client.rate_limiter)
        print(f"Success! Got data: {type(data)}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
    
    return False

if __name__ == "__main__":
    result = asyncio.run(test_api())
    sys.exit(0 if result else 1)

