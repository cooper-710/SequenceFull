#!/bin/bash
# Setup script for Player Database
# This script sets the Sportradar API key and runs the initial database population

# Set the API key
export SPORTRADAR_API_KEY="Z2ZMn59qIGMtzMrDKQB9smyd2ANvPxj98FXZicIp"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the population script
echo "Starting player database population..."
echo "This may take a few minutes..."
python3 src/populate_players.py

echo "Done! Database populated at build/database/players.db"

