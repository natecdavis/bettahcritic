#!/usr/bin/env python3
"""
Daily Update Script for Adjusted Metacritic Scores

This script runs the full data pipeline:
1. Fetch new movies and reviews from Metacritic
2. Re-compute hierarchical critic/outlet effects
3. Re-compute adjusted scores (dates come clean from the scraper)
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


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'categories.json')


def load_category_config(category: str) -> dict:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    if category not in config:
        raise SystemExit(f"Unknown category {category!r}; known: "
                         f"{[k for k in config if not k.startswith('_')]}")
    return config[category]


def write_model_info(category: str, cfg: dict, effects_dir: str, scores_dir: str) -> str:
    """Collect what the app should say about this category's methodology:
    the configured adjustments, effect-model fit, and the file-drawer
    diagnostics (if analyze_categories.py has been run)."""
    info = {
        'label': cfg.get('label', category),
        'coverage_prior': bool(cfg.get('coverage_prior')),
        'excluded_release_types': cfg.get('exclude_release_types') or [],
    }
    effects_meta = os.path.join(effects_dir, 'metadata.json')
    if os.path.exists(effects_meta):
        with open(effects_meta) as f:
            em = json.load(f)
        info['share_reviews_named_critic'] = em.get('share_reviews_named_critic')
        info['hierarchical_variance_explained'] = em.get('hierarchical_variance_explained')
    diag_path = os.path.join(cfg['data_root'], 'diagnostics.json')
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            info['diagnostics'] = json.load(f)
    path = os.path.join(scores_dir, 'model_info.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return path


def compute_hierarchical_effects(data_dir: str, effects_dir: str, scripts_dir: str,
                                 items_file: str = None, exclude_types=None):
    """Re-compute hierarchical critic/outlet effects."""
    script_path = os.path.join(scripts_dir, 'compute_hierarchical_effects.py')
    
    if not os.path.exists(script_path):
        print(f"WARNING: {script_path} not found, skipping effects computation")
        return True
    
    cmd = [
        sys.executable, script_path,
        '--input-dir', data_dir,
        '--output-dir', effects_dir,
    ]
    if items_file:
        cmd += ['--items-file', items_file]
    if exclude_types:
        cmd += ['--exclude-release-types', ','.join(exclude_types)]

    return run_command(cmd, "Compute hierarchical effects")


def compute_adjusted_scores(data_dir: str, effects_dir: str, output_dir: str, scripts_dir: str,
                            cfg: dict = None):
    """Re-compute adjusted scores."""
    script_path = os.path.join(scripts_dir, 'metacritic_adjusted_scores_v2.py')
    
    if not os.path.exists(script_path):
        print(f"WARNING: {script_path} not found, skipping score computation")
        return True
    
    cmd = [
        sys.executable, script_path,
        '--input-dir', data_dir,
        '--effects-dir', effects_dir,
        '--output-dir', output_dir,
        '--recompute-shrinkage',
    ]
    cfg = cfg or {}
    if cfg.get('items_file'):
        cmd += ['--items-file', cfg['items_file']]
    if cfg.get('exclude_release_types'):
        cmd += ['--exclude-release-types', ','.join(cfg['exclude_release_types'])]
    if cfg.get('include_unscored'):
        cmd += ['--include-unscored']
    if cfg.get('coverage_prior'):
        cmd += ['--coverage-prior']
    if cfg.get('max_review_lag_days') is not None:
        cmd += ['--max-review-lag-days', str(cfg['max_review_lag_days'])]

    return run_command(cmd, "Compute adjusted scores")


def convert_to_json(output_dir: str, webapp_data_dir: str, scripts_dir: str,
                    category: str = 'movies', model_info_path: str = None):
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
        sys.executable, script_path,
        '--input', adjusted_path,
        '--output', webapp_data_dir,
        '--category', category,
    ]
    if model_info_path:
        cmd += ['--model-info', model_info_path]

    return run_command(cmd, "Convert to JSON")


def update_metadata(webapp_data_dir: str, category: str = 'movies'):
    """Update the metadata file with current timestamp."""
    meta_path = os.path.join(webapp_data_dir, f'{category}_metadata.json')
    
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
    parser.add_argument('--category', type=str, default='movies',
                        help='Category in config/categories.json; sets the data/effects/scores '
                             'dirs (under its data_root) and model settings')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing the items CSV and reviews.csv')
    parser.add_argument('--effects-dir', type=str, default=None,
                        help='Directory for hierarchical effects output')
    parser.add_argument('--scores-dir', type=str, default=None,
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

    cfg = load_category_config(args.category)
    root = cfg['data_root']
    args.data_dir = args.data_dir or os.path.join(root, 'metacritic_data')
    args.effects_dir = args.effects_dir or os.path.join(root, 'hierarchical_effects')
    args.scores_dir = args.scores_dir or os.path.join(root, 'adjusted_scores')

    print("="*60)
    print(f"DAILY UPDATE - Adjusted Metacritic Scores [{args.category}]")
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
        if not compute_hierarchical_effects(args.data_dir, args.effects_dir, args.scripts_dir,
                                            cfg.get('items_file'),
                                            cfg.get('exclude_release_types')):
            success = False
    
    # Step 3: Compute adjusted scores
    if success:
        if not compute_adjusted_scores(args.data_dir, args.effects_dir, args.scores_dir,
                                       args.scripts_dir, cfg):
            success = False
    
    # Step 4: Convert to JSON
    if success:
        model_info_path = write_model_info(args.category, cfg, args.effects_dir, args.scores_dir)
        if not convert_to_json(args.scores_dir, args.webapp_dir, args.scripts_dir,
                               args.category, model_info_path):
            success = False

    # Step 4b: TV show-level aggregates (Shows tab)
    if success and cfg.get('show_aggregates'):
        cmd = [sys.executable, os.path.join(args.scripts_dir, 'tv_show_scores.py'),
               '--scores-dir', args.scores_dir, '--webapp-dir', args.webapp_dir,
               '--items-path', os.path.join(args.data_dir, cfg.get('items_file') or 'items.csv')]
        if not run_command(cmd, "Aggregate TV show scores"):
            success = False

    # Step 5: Update metadata
    if success:
        update_metadata(args.webapp_dir, args.category)
    
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
