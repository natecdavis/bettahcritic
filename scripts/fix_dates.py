"""
Fix dates in adjusted_scores.csv

Replaces dates with properly formatted four-digit year dates from the original
movies.csv file to avoid Excel/parsing issues with two-digit years.
"""

import pandas as pd
import argparse
import os


def fix_dates(adjusted_path: str, movies_path: str, output_path: str = None):
    """
    Fix dates in adjusted_scores.csv using original movies.csv dates.
    """
    print("Loading data...")
    adjusted_df = pd.read_csv(adjusted_path)
    movies_df = pd.read_csv(movies_path)
    
    print(f"Adjusted scores: {len(adjusted_df):,} rows")
    print(f"Movies: {len(movies_df):,} rows")
    
    # Check current date format in adjusted_scores
    print(f"\nCurrent release_date sample in adjusted_scores:")
    print(adjusted_df['release_date'].head(10).tolist())
    
    # Check original date format in movies.csv
    print(f"\nOriginal release_date sample in movies.csv:")
    print(movies_df['release_date'].head(10).tolist())
    
    # Parse original dates with explicit format handling
    # Try to detect the format
    sample_date = movies_df['release_date'].dropna().iloc[0]
    print(f"\nSample date string: '{sample_date}'")
    
    # Parse dates, being careful about year interpretation
    def parse_date_safe(date_str):
        """Parse date string ensuring four-digit year output."""
        if pd.isna(date_str):
            return None
        
        try:
            # Try parsing with pandas
            dt = pd.to_datetime(date_str)
            
            # Check if year seems wrong (future date for old movie)
            # If year > 2030, it's probably a parsing error for a 19xx date
            if dt.year > 2030:
                # Subtract 100 years
                dt = dt.replace(year=dt.year - 100)
            
            # Return as YYYY-MM-DD format
            return dt.strftime('%Y-%m-%d')
        except:
            return str(date_str)
    
    # Create a lookup from movie_slug to properly formatted date
    movies_df['release_date_fixed'] = movies_df['release_date'].apply(parse_date_safe)
    
    date_lookup = movies_df.set_index('movie_slug')['release_date_fixed'].to_dict()
    
    # Also get year from the fixed date
    def extract_year(date_str):
        if pd.isna(date_str):
            return None
        try:
            return pd.to_datetime(date_str).year
        except:
            return None
    
    movies_df['year_fixed'] = movies_df['release_date_fixed'].apply(extract_year)
    year_lookup = movies_df.set_index('movie_slug')['year_fixed'].to_dict()
    
    # Apply fixes to adjusted_scores
    print("\nApplying date fixes...")
    
    adjusted_df['release_date'] = adjusted_df['movie_slug'].map(date_lookup)
    
    # Add a year column if not present, or fix it
    if 'year' in adjusted_df.columns:
        adjusted_df['year'] = adjusted_df['movie_slug'].map(year_lookup)
    else:
        adjusted_df['year'] = adjusted_df['movie_slug'].map(year_lookup)
    
    # Verify the fix
    print(f"\nFixed release_date sample:")
    print(adjusted_df['release_date'].head(10).tolist())
    
    # Check for any remaining future dates
    adjusted_df['year_check'] = pd.to_datetime(adjusted_df['release_date'], errors='coerce').dt.year
    future_dates = adjusted_df[adjusted_df['year_check'] > 2026]
    
    if len(future_dates) > 0:
        print(f"\nWARNING: {len(future_dates)} rows still have future dates:")
        print(future_dates[['movie_slug', 'title', 'release_date', 'year_check']].head(20))
    else:
        print(f"\n✓ No future dates found - fix successful!")
    
    # Check for very old dates that might have been incorrectly parsed
    old_dates = adjusted_df[(adjusted_df['year_check'] < 1900) & (adjusted_df['year_check'].notna())]
    if len(old_dates) > 0:
        print(f"\nWARNING: {len(old_dates)} rows have dates before 1900:")
        print(old_dates[['movie_slug', 'title', 'release_date', 'year_check']].head(20))
    
    # Drop the check column
    adjusted_df = adjusted_df.drop(columns=['year_check'])
    
    # Reorder columns to put year near release_date
    cols = adjusted_df.columns.tolist()
    if 'year' in cols and 'release_date' in cols:
        # Move year to right after release_date
        cols.remove('year')
        release_idx = cols.index('release_date')
        cols.insert(release_idx + 1, 'year')
        adjusted_df = adjusted_df[cols]
    
    # Save
    if output_path is None:
        output_path = adjusted_path  # Overwrite
    
    adjusted_df.to_csv(output_path, index=False)
    print(f"\nSaved fixed file to: {output_path}")
    
    # Show some examples of old movies to verify
    print(f"\nVerification - oldest movies in dataset:")
    oldest = adjusted_df.sort_values('release_date').head(15)
    for _, row in oldest.iterrows():
        print(f"  {row['release_date']}  {row['title'][:50]}")
    
    return adjusted_df


def main():
    parser = argparse.ArgumentParser(description='Fix dates in adjusted_scores.csv')
    parser.add_argument('--adjusted', type=str, default='./adjusted_scores_v2/adjusted_scores.csv',
                        help='Path to adjusted_scores.csv')
    parser.add_argument('--movies', type=str, default='./metacritic_data/movies.csv',
                        help='Path to original movies.csv')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: overwrite adjusted)')
    
    args = parser.parse_args()
    
    fix_dates(args.adjusted, args.movies, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
