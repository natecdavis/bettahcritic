"""
Convert adjusted_scores.csv to JSON for the webapp.

This should be run daily as part of your data pipeline.
"""

import pandas as pd
import json
import argparse
import os
from datetime import datetime


def csv_to_json(csv_path: str, output_path: str, recent_years: int = 2):
    """
    Convert adjusted scores CSV to JSON for the webapp.
    
    Creates two files:
    - movies_recent.json: Last N years of movies
    - movies_all.json: All movies
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df):,} movies")
    
    # Ensure required columns exist
    required_cols = ['movie_slug', 'title', 'release_date', 'genre', 'raw_score', 
                     'adjusted_score', 'n_reviews', 'total_adjustment']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
    
    # Extract year if not present
    if 'year' not in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    # Clean up data
    df = df.dropna(subset=['adjusted_score', 'title'])
    
    # Select and rename columns for webapp
    webapp_cols = {
        'movie_slug': 'movie_slug',
        'title': 'title',
        'release_date': 'release_date',
        'year': 'year',
        'genre': 'genre',
        'raw_score': 'raw_score',
        'adjusted_score': 'adjusted_score',
        'n_reviews': 'n_reviews',
        'total_adjustment': 'total_adjustment',
    }
    
    # Only keep columns that exist
    cols_to_use = {k: v for k, v in webapp_cols.items() if k in df.columns}
    df_out = df[list(cols_to_use.keys())].rename(columns=cols_to_use)
    
    # Round scores for cleaner JSON
    if 'adjusted_score' in df_out.columns:
        df_out['adjusted_score'] = df_out['adjusted_score'].round(1)
    if 'total_adjustment' in df_out.columns:
        df_out['total_adjustment'] = df_out['total_adjustment'].round(2)
    
    # Handle NaN values - replace with None for valid JSON
    # For numeric columns, NaN -> None
    # For string columns, NaN -> empty string or 'Unknown'
    df_out = df_out.fillna({
        'genre': 'Unknown',
        'title': 'Unknown',
        'release_date': '',
        'movie_slug': '',
    })
    
    # Convert any remaining NaN to None (which becomes null in JSON)
    df_out = df_out.where(pd.notnull(df_out), None)
    
    # Ensure year is integer or None
    if 'year' in df_out.columns:
        df_out['year'] = df_out['year'].apply(lambda x: int(x) if pd.notnull(x) and x == x else None)
    
    # Fill missing genres
    if 'genre' in df_out.columns:
        df_out['genre'] = df_out['genre'].fillna('Unknown')
        # Take first genre if multiple
        df_out['genre'] = df_out['genre'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
    
    # Sort by adjusted score
    df_out = df_out.sort_values('adjusted_score', ascending=False)
    
    # Convert to records
    # Use a custom function to ensure no NaN values slip through
    def clean_record(record):
        """Clean a single record, replacing NaN with None."""
        cleaned = {}
        for k, v in record.items():
            if pd.isna(v) or (isinstance(v, float) and (v != v)):  # NaN check
                cleaned[k] = None
            elif isinstance(v, float):
                # Keep as float but ensure it's not NaN/Inf
                if v != v or v == float('inf') or v == float('-inf'):
                    cleaned[k] = None
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = v
        return cleaned
    
    all_movies = [clean_record(r) for r in df_out.to_dict('records')]
    
    # Filter for recent movies (must have valid release date and year)
    current_year = datetime.now().year
    cutoff_year = current_year - recent_years + 1
    recent_movies = [m for m in all_movies 
                     if m.get('year') is not None 
                     and m.get('year') >= cutoff_year
                     and m.get('release_date')]
    
    print(f"Recent movies ({cutoff_year}+): {len(recent_movies):,}")
    print(f"All movies: {len(all_movies):,}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save recent movies
    recent_path = os.path.join(output_path, 'movies_recent.json')
    with open(recent_path, 'w') as f:
        json.dump({
            'updated': datetime.now().isoformat(),
            'count': len(recent_movies),
            'movies': recent_movies
        }, f, indent=2)
    print(f"Saved: {recent_path}")
    
    # Save all movies
    all_path = os.path.join(output_path, 'movies_all.json')
    with open(all_path, 'w') as f:
        json.dump({
            'updated': datetime.now().isoformat(),
            'count': len(all_movies),
            'movies': all_movies
        }, f, indent=2)
    print(f"Saved: {all_path}")
    
    # Save metadata
    genres = df_out['genre'].unique().tolist() if 'genre' in df_out.columns else []
    years = sorted(df_out['year'].dropna().unique().tolist()) if 'year' in df_out.columns else []
    
    metadata = {
        'updated': datetime.now().isoformat(),
        'total_movies': len(all_movies),
        'recent_movies': len(recent_movies),
        'recent_cutoff_year': cutoff_year,
        'genres': sorted([g for g in genres if g and g != 'Unknown']),
        'year_range': {'min': int(min(years)) if years else None, 'max': int(max(years)) if years else None},
    }
    
    meta_path = os.path.join(output_path, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")
    
    # Print sample
    print(f"\nSample of recent movies:")
    for m in recent_movies[:10]:
        print(f"  {m.get('adjusted_score', 0):.1f}  {m.get('title', 'Unknown')[:50]}")
    
    return all_movies, recent_movies


def main():
    parser = argparse.ArgumentParser(description='Convert adjusted scores to JSON for webapp')
    parser.add_argument('--input', type=str, default='./adjusted_scores_v2/adjusted_scores.csv',
                        help='Path to adjusted_scores.csv')
    parser.add_argument('--output', type=str, default='./webapp_data',
                        help='Output directory for JSON files')
    parser.add_argument('--recent-years', type=int, default=2,
                        help='Number of years to include in recent movies')
    
    args = parser.parse_args()
    
    csv_to_json(args.input, args.output, args.recent_years)
    
    print("\nDone!")


if __name__ == "__main__":
    main()