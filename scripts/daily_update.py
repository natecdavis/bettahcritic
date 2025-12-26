#!/usr/bin/env python3
"""
Daily Update Script for Adjusted Metacritic Scores

This script runs the full data pipeline:
1. Fetch new movies and reviews from Metacritic
2. Re-compute hierarchical critic/outlet effects
3. Re-compute adjusted scores
4. Convert to JSON for the webapp

Usage:
    python daily_update.py --data-dir /path/to/metacritic_data --webapp-dir /path/to/webapp

For GitHub Actions, set up as a scheduled workflow (see .github/workflows/daily-update.yml)
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime, timedelta
import json


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"stderr: {result.stderr}")
        return False
    
    print(f"✓ {description} completed successfully")
    return True


def fetch_recent_movies(data_dir: str, days: int = 7):
    """
    Fetch movies released in the last N days.
    
    This is a placeholder - you'll need to implement based on your scraper.
    """
    print(f"\nFetching movies from the last {days} days...")
    
    # Your scraper command here, e.g.:
    # cmd = ['python', 'scrape_metacritic.py', '--days', str(days), '--output', data_dir]
    # return run_command(cmd, f"Fetch movies from last {days} days")
    
    # For now, just check if the data exists
    movies_path = os.path.join(data_dir, 'movies.csv')
    reviews_path = os.path.join(data_dir, 'reviews.csv')
    
    if not os.path.exists(movies_path) or not os.path.exists(reviews_path):
        print(f"WARNING: Data files not found in {data_dir}")
        print("Skipping fetch step - using existing data")
        return True
    
    print(f"✓ Data files found in {data_dir}")
    return True


def compute_hierarchical_effects(data_dir: str, effects_dir: str, scripts_dir: str):
    """Re-compute hierarchical critic/outlet effects."""
    script_path = os.path.join(scripts_dir, 'compute_hierarchical_effects.py')
    
    if not os.path.exists(script_path):
        print(f"WARNING: {script_path} not found, skipping effects computation")
        return True
    
    cmd = [
        'python', script_path,
        '--input-dir', data_dir,
        '--output-dir', effects_dir,
    ]
    
    return run_command(cmd, "Compute hierarchical effects")


def compute_adjusted_scores(data_dir: str, effects_dir: str, output_dir: str, scripts_dir: str):
    """Re-compute adjusted scores."""
    script_path = os.path.join(scripts_dir, 'metacritic_adjusted_scores_v2.py')
    
    if not os.path.exists(script_path):
        print(f"WARNING: {script_path} not found, skipping score computation")
        return True
    
    cmd = [
        'python', script_path,
        '--input-dir', data_dir,
        '--effects-dir', effects_dir,
        '--output-dir', output_dir,
    ]
    
    return run_command(cmd, "Compute adjusted scores")


def fix_dates(output_dir: str, data_dir: str, scripts_dir: str):
    """Fix any date parsing issues."""
    script_path = os.path.join(scripts_dir, 'fix_dates.py')
    adjusted_path = os.path.join(output_dir, 'adjusted_scores.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')
    
    if not os.path.exists(script_path):
        print(f"WARNING: {script_path} not found, skipping date fix")
        return True
    
    if not os.path.exists(adjusted_path):
        print(f"WARNING: {adjusted_path} not found, skipping date fix")
        return True
    
    cmd = [
        'python', script_path,
        '--adjusted', adjusted_path,
        '--movies', movies_path,
    ]
    
    return run_command(cmd, "Fix dates")


def convert_to_json(output_dir: str, webapp_data_dir: str, scripts_dir: str):
    """Convert CSV to JSON for the webapp."""
    script_path = os.path.join(scripts_dir, 'csv_to_json.py')
    adjusted_path = os.path.join(output_dir, 'adjusted_scores.csv')
    
    if not os.path.exists(script_path):
        print(f"WARNING: {script_path} not found, skipping JSON conversion")
        return True
    
    if not os.path.exists(adjusted_path):
        print(f"WARNING: {adjusted_path} not found, skipping JSON conversion")
        return True
    
    cmd = [
        'python', script_path,
        '--input', adjusted_path,
        '--output', webapp_data_dir,
    ]
    
    return run_command(cmd, "Convert to JSON")


def update_metadata(webapp_data_dir: str):
    """Update the metadata file with current timestamp."""
    meta_path = os.path.join(webapp_data_dir, 'metadata.json')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    metadata['updated'] = datetime.now().isoformat()
    metadata['last_update_status'] = 'success'
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Updated metadata timestamp")
    return True


def main():
    parser = argparse.ArgumentParser(description='Daily update for Adjusted Metacritic Scores')
    parser.add_argument('--data-dir', type=str, default='./metacritic_data',
                        help='Directory containing movies.csv and reviews.csv')
    parser.add_argument('--effects-dir', type=str, default='./hierarchical_effects',
                        help='Directory for hierarchical effects output')
    parser.add_argument('--scores-dir', type=str, default='./adjusted_scores_v2',
                        help='Directory for adjusted scores output')
    parser.add_argument('--webapp-dir', type=str, default='./public/data',
                        help='Directory for webapp JSON files')
    parser.add_argument('--scripts-dir', type=str, default='.',
                        help='Directory containing Python scripts')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetching new data (use existing)')
    parser.add_argument('--skip-effects', action='store_true',
                        help='Skip recomputing effects (use existing)')
    parser.add_argument('--fetch-days', type=int, default=7,
                        help='Number of days to fetch new movies for')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DAILY UPDATE - Adjusted Metacritic Scores")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    # Ensure directories exist
    os.makedirs(args.effects_dir, exist_ok=True)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.webapp_dir, exist_ok=True)
    
    success = True
    
    # Step 1: Fetch new data (optional)
    if not args.skip_fetch:
        if not fetch_recent_movies(args.data_dir, args.fetch_days):
            print("WARNING: Fetch failed, continuing with existing data")
    
    # Step 2: Compute hierarchical effects
    if not args.skip_effects:
        if not compute_hierarchical_effects(args.data_dir, args.effects_dir, args.scripts_dir):
            success = False
    
    # Step 3: Compute adjusted scores
    if success:
        if not compute_adjusted_scores(args.data_dir, args.effects_dir, args.scores_dir, args.scripts_dir):
            success = False
    
    # Step 4: Fix dates
    if success:
        if not fix_dates(args.scores_dir, args.data_dir, args.scripts_dir):
            success = False
    
    # Step 5: Convert to JSON
    if success:
        if not convert_to_json(args.scores_dir, args.webapp_dir, args.scripts_dir):
            success = False
    
    # Step 6: Update metadata
    if success:
        update_metadata(args.webapp_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print(f"\nJSON files updated in: {args.webapp_dir}")
        print("Ready to deploy!")
    else:
        print("\nSome steps failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
