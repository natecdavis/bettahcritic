"""
Merge multiple movie/review datasets without duplicates.

Where datasets overlap, LATER-listed files win — list the old dataset first
and the freshest scrape last, so re-scraped metadata (metascores, release
dates) replaces stale values.

Usage:
    python merge_data.py --output-dir ./data/metacritic_data \
        --movies ./data/metacritic_data/movies.csv ./data/metacritic_fresh/movies.csv \
        --reviews ./data/metacritic_data/reviews.csv ./data/metacritic_fresh/reviews.csv
"""

import pandas as pd
import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Merge movie/review datasets')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for merged files')
    parser.add_argument('--movies', nargs='+', required=True,
                        help='List of movies.csv files to merge')
    parser.add_argument('--reviews', nargs='+', required=True,
                        help='List of reviews.csv files to merge')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MERGE DATASETS")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Merge movies
    print("\n" + "="*60)
    print("MERGING MOVIES")
    print("="*60)
    
    all_movies = []
    for path in args.movies:
        print(f"  Loading {path}...")
        df = pd.read_csv(path)
        print(f"    {len(df):,} movies")
        all_movies.append(df)
    
    combined_movies = pd.concat(all_movies, ignore_index=True)
    print(f"\nCombined: {len(combined_movies):,} movies")
    
    # Dedupe by slug: later-listed (fresher) files win per column, but a
    # missing value never overwrites a real one (scrapes have transient
    # gaps, e.g. a listing item returned without its metascore)
    before = len(combined_movies)
    combined_movies = combined_movies.groupby('movie_slug', sort=False).last().reset_index()
    combined_movies = combined_movies[all_movies[0].columns.tolist()]
    after = len(combined_movies)
    print(f"After dedupe: {after:,} movies ({before - after:,} duplicates removed)")
    
    movies_path = os.path.join(args.output_dir, 'movies_merged.csv')
    combined_movies.to_csv(movies_path, index=False)
    print(f"Saved to: {movies_path}")
    
    # Merge reviews
    print("\n" + "="*60)
    print("MERGING REVIEWS")
    print("="*60)
    
    all_reviews = []
    for path in args.reviews:
        print(f"  Loading {path}...")
        df = pd.read_csv(path)
        print(f"    {len(df):,} reviews")
        all_reviews.append(df)
    
    combined_reviews = pd.concat(all_reviews, ignore_index=True)
    print(f"\nCombined: {len(combined_reviews):,} reviews")
    
    # Dedupe by (movie, outlet, critic): later-listed files win per column,
    # missing values never overwrite real ones. Key fields are normalized
    # to '' so NaN (from CSV) and '' (from scrapes) match.
    key_cols = ['movie_slug', 'outlet_slug', 'critic_slug']
    before = len(combined_reviews)
    for c in key_cols:
        combined_reviews[c] = combined_reviews[c].fillna('')
    combined_reviews = combined_reviews.groupby(key_cols, sort=False).last().reset_index()
    combined_reviews = combined_reviews[all_reviews[0].columns.tolist()]
    after = len(combined_reviews)
    print(f"After dedupe: {after:,} reviews ({before - after:,} duplicates removed)")
    
    reviews_path = os.path.join(args.output_dir, 'reviews_merged.csv')
    combined_reviews.to_csv(reviews_path, index=False)
    print(f"Saved to: {reviews_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Movies: {len(combined_movies):,}")
    print(f"Reviews: {len(combined_reviews):,}")
    print(f"Movies with reviews: {combined_reviews['movie_slug'].nunique():,}")
    print(f"\nOutput files:")
    print(f"  {movies_path}")
    print(f"  {reviews_path}")
    print(f"\nTo use merged data, rename files:")
    print(f"  mv {movies_path} {os.path.join(args.output_dir, 'movies.csv')}")
    print(f"  mv {reviews_path} {os.path.join(args.output_dir, 'reviews.csv')}")


if __name__ == "__main__":
    main()