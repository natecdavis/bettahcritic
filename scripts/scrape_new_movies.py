"""
Incremental Metacritic Scraper

Fetches new movies and reviews since your last update.
Stops when it encounters movies already in your dataset.

Usage:
    # Dry run - see what would be fetched
    python scrape_new_movies.py --input-dir ./data/metacritic_data --dry-run
    
    # Actually fetch new data
    python scrape_new_movies.py --input-dir ./data/metacritic_data
    
    # Fetch more pages (if you haven't updated in a while)
    python scrape_new_movies.py --input-dir ./data/metacritic_data --max-pages 20
"""

import requests
import pandas as pd
import json
import os
import time
import argparse
from datetime import datetime
from tqdm import tqdm


# API endpoints
MOVIES_API = "https://backend.metacritic.com/finder/metacritic/web"
REVIEWS_API = "https://backend.metacritic.com/reviews/metacritic/critic/movies/{slug}/web"

# Request headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.metacritic.com/',
    'Origin': 'https://www.metacritic.com',
}


def get_recent_movies(page: int = 0, page_size: int = 24, delay: float = 0.5) -> list[dict]:
    """
    Fetch a page of recent movies from Metacritic, sorted by release date (newest first).
    
    Returns list of movie dicts with basic info.
    """
    params = {
        'sortBy': '-releaseDate',  # Newest first
        'productType': 'movies',
        'page': page,
        'limit': page_size,
        'offset': page * page_size,
    }
    
    time.sleep(delay)
    
    try:
        response = requests.get(MOVIES_API, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        items = data.get('data', {}).get('items', [])
        
        movies = []
        for item in items:
            # Handle genres - can be list of strings or list of dicts
            genres = item.get('genres', [])
            if genres and isinstance(genres[0], dict):
                genre_str = ', '.join(g.get('name', '') for g in genres if g.get('name'))
            elif genres:
                genre_str = ', '.join(genres)
            else:
                genre_str = ''
            
            movie = {
                'movie_slug': item.get('slug', ''),
                'title': item.get('title', ''),
                'release_date': item.get('releaseDate', ''),
                'metascore': item.get('criticScoreSummary', {}).get('score'),
                'user_score': item.get('userScoreSummary', {}).get('score'),
                'genre': genre_str,
                'rating': item.get('rating', ''),
                'description': item.get('description', '')[:500] if item.get('description') else '',
            }
            
            if movie['movie_slug']:
                movies.append(movie)
        
        return movies
        
    except requests.RequestException as e:
        print(f"Error fetching movies page {page}: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing movies page {page}: {e}")
        return []


def get_movie_reviews(slug: str, delay: float = 0.3) -> list[dict]:
    """
    Fetch all critic reviews for a movie.
    Paginates through all results.
    """
    url = REVIEWS_API.format(slug=slug)
    all_reviews = []
    offset = 0
    limit = 10
    
    while True:
        params = {
            'offset': offset,
            'limit': limit,
            'filterBySentiment': 'all',
            'sort': 'date',  # Sort by date to get chronological order
            'componentName': 'critic-reviews',
            'componentDisplayName': 'Critic Reviews',
            'componentType': 'ReviewList',
        }
        
        try:
            time.sleep(delay)
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            total_results = data.get('data', {}).get('totalResults', 0)
            items = data.get('data', {}).get('items', [])
            
            if not items:
                break
            
            for item in items:
                score = item.get('score')
                
                if score is not None:
                    if score >= 61:
                        sentiment = 'positive'
                    elif score >= 40:
                        sentiment = 'mixed'
                    else:
                        sentiment = 'negative'
                else:
                    sentiment = None
                
                review = {
                    'movie_slug': slug,
                    'score': score,
                    'sentiment': sentiment,
                    'outlet': item.get('publicationName', ''),
                    'outlet_slug': item.get('publicationSlug', ''),
                    'critic': item.get('author', ''),
                    'critic_slug': item.get('authorSlug', ''),
                    'excerpt': item.get('quote', '')[:500] if item.get('quote') else '',
                    'review_url': item.get('url', ''),
                    'date': item.get('date'),
                }
                
                if review['score'] is not None and review['outlet']:
                    all_reviews.append(review)
            
            offset += limit
            if offset >= total_results:
                break
                
        except requests.RequestException as e:
            print(f"  Error fetching reviews for {slug}: {e}")
            break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error parsing reviews for {slug}: {e}")
            break
    
    return all_reviews


def fetch_new_movies_incrementally(
    existing_slugs: set,
    max_pages: int = 10,
    stop_after_n_existing: int = 5,
    delay: float = 0.5
) -> list[dict]:
    """
    Fetch new movies until we hit several existing ones in a row.
    
    Args:
        existing_slugs: Set of movie slugs already in the database
        max_pages: Maximum pages to fetch
        stop_after_n_existing: Stop after seeing this many existing movies in a row
        delay: Delay between requests
    
    Returns:
        List of new movie dicts
    """
    new_movies = []
    consecutive_existing = 0
    
    print(f"Fetching new movies (will stop after {stop_after_n_existing} consecutive existing movies)...")
    
    for page in range(max_pages):
        print(f"  Page {page + 1}/{max_pages}...", end=" ")
        
        movies = get_recent_movies(page=page, delay=delay)
        
        if not movies:
            print("no results")
            break
        
        page_new = 0
        page_existing = 0
        
        for movie in movies:
            slug = movie['movie_slug']
            
            if slug in existing_slugs:
                consecutive_existing += 1
                page_existing += 1
                
                if consecutive_existing >= stop_after_n_existing:
                    print(f"{page_new} new, {page_existing} existing (stopping)")
                    return new_movies
            else:
                consecutive_existing = 0
                new_movies.append(movie)
                page_new += 1
        
        print(f"{page_new} new, {page_existing} existing")
        
        # If entire page was existing movies, probably safe to stop
        if page_new == 0:
            print("  Entire page was existing movies, stopping.")
            break
    
    return new_movies


def check_for_new_reviews(
    movies_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    months_back: int = 3,
    delay: float = 0.3
) -> list[dict]:
    """
    Check recent movies for new reviews that aren't in our dataset.
    
    Args:
        movies_df: Existing movies DataFrame
        reviews_df: Existing reviews DataFrame
        months_back: How many months back to check
        delay: Delay between API requests
    
    Returns:
        List of new review dicts
    """
    # Filter to recent movies
    movies_df = movies_df.copy()
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
    
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
    recent_movies = movies_df[movies_df['release_date'] >= cutoff]
    
    print(f"Checking {len(recent_movies)} movies from the last {months_back} months for new reviews...")
    
    # Count existing reviews per movie
    existing_review_counts = reviews_df.groupby('movie_slug').size().to_dict()
    
    # Build a set of existing (movie_slug, outlet_slug, critic_slug) tuples for deduplication
    existing_reviews_set = set()
    for _, row in reviews_df.iterrows():
        key = (row.get('movie_slug'), row.get('outlet_slug'), row.get('critic_slug'))
        existing_reviews_set.add(key)
    
    new_reviews = []
    movies_with_new_reviews = 0
    
    for _, movie in tqdm(recent_movies.iterrows(), total=len(recent_movies), desc="Checking for new reviews"):
        slug = movie['movie_slug']
        existing_count = existing_review_counts.get(slug, 0)
        
        # Fetch current reviews from API
        current_reviews = get_movie_reviews(slug, delay=delay)
        
        if len(current_reviews) > existing_count:
            # There might be new reviews - check which ones we don't have
            movie_new_reviews = []
            
            for review in current_reviews:
                key = (review['movie_slug'], review.get('outlet_slug'), review.get('critic_slug'))
                
                if key not in existing_reviews_set:
                    movie_new_reviews.append(review)
                    existing_reviews_set.add(key)  # Don't add duplicates within this run
            
            if movie_new_reviews:
                new_reviews.extend(movie_new_reviews)
                movies_with_new_reviews += 1
    
    print(f"Found {len(new_reviews)} new reviews across {movies_with_new_reviews} movies")
    
    return new_reviews


def main():
    parser = argparse.ArgumentParser(description='Incrementally fetch new movies and reviews')
    parser.add_argument('--input-dir', type=str, default='./data/metacritic_data',
                        help='Directory with movies.csv and reviews.csv')
    parser.add_argument('--max-pages', type=int, default=10,
                        help='Maximum pages of movies to check')
    parser.add_argument('--stop-after', type=int, default=5,
                        help='Stop after N consecutive existing movies')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between API requests')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be fetched without actually fetching')
    parser.add_argument('--skip-reviews', action='store_true',
                        help='Only fetch movies, not reviews')
    parser.add_argument('--months-back', type=int, default=3,
                        help='How many months back to check for new reviews on existing movies')
    parser.add_argument('--skip-new-review-check', action='store_true',
                        help='Skip checking existing movies for new reviews')
    
    args = parser.parse_args()
    
    print("="*60)
    print("INCREMENTAL METACRITIC SCRAPER")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    # Load existing data
    movies_path = os.path.join(args.input_dir, 'movies.csv')
    reviews_path = os.path.join(args.input_dir, 'reviews.csv')
    
    if os.path.exists(movies_path):
        print(f"\nLoading existing movies from {movies_path}...")
        existing_movies_df = pd.read_csv(movies_path)
        existing_slugs = set(existing_movies_df['movie_slug'].dropna())
        print(f"Found {len(existing_slugs):,} existing movies")
    else:
        print(f"\nNo existing movies file found at {movies_path}")
        print("Will create new file.")
        existing_movies_df = pd.DataFrame()
        existing_slugs = set()
    
    if os.path.exists(reviews_path):
        existing_reviews_df = pd.read_csv(reviews_path)
        print(f"Found {len(existing_reviews_df):,} existing reviews")
    else:
        existing_reviews_df = pd.DataFrame()
        print("No existing reviews file found, will create new file.")
    
    # Fetch new movies
    print("\n" + "="*60)
    print("STEP 1: FETCH NEW MOVIES")
    print("="*60)
    
    new_movies = fetch_new_movies_incrementally(
        existing_slugs=existing_slugs,
        max_pages=args.max_pages,
        stop_after_n_existing=args.stop_after,
        delay=args.delay
    )
    
    print(f"\nFound {len(new_movies)} new movies")
    
    if new_movies:
        print("\nNew movies:")
        for movie in new_movies[:20]:
            score = movie.get('metascore', 'N/A')
            score_str = f"{score:3.0f}" if score else "N/A"
            print(f"  [{score_str}] {movie['title'][:50]} ({movie['release_date'][:10] if movie['release_date'] else 'N/A'})")
        
        if len(new_movies) > 20:
            print(f"  ... and {len(new_movies) - 20} more")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would add {len(new_movies)} movies and fetch their reviews")
        if not args.skip_new_review_check and len(existing_movies_df) > 0:
            # Count how many existing movies are in the check window
            existing_movies_df_temp = existing_movies_df.copy()
            existing_movies_df_temp['release_date'] = pd.to_datetime(existing_movies_df_temp['release_date'], errors='coerce')
            cutoff = pd.Timestamp.now() - pd.DateOffset(months=args.months_back)
            recent_count = len(existing_movies_df_temp[existing_movies_df_temp['release_date'] >= cutoff])
            print(f"[DRY RUN] Would also check {recent_count} existing movies (last {args.months_back} months) for new reviews")
        return
    
    if not new_movies:
        print("\nNo new movies found.")
        if args.skip_new_review_check or args.skip_reviews:
            print("Data is up to date!")
            return
        # Continue to check for new reviews on existing movies
    
    # Fetch reviews for new movies
    if not args.skip_reviews:
        print("\n" + "="*60)
        print("STEP 2: FETCH REVIEWS FOR NEW MOVIES")
        print("="*60)
        
        all_new_reviews = []
        movies_with_reviews = 0
        
        for movie in tqdm(new_movies, desc="Fetching reviews"):
            slug = movie['movie_slug']
            reviews = get_movie_reviews(slug, delay=args.delay)
            
            if reviews:
                all_new_reviews.extend(reviews)
                movies_with_reviews += 1
        
        print(f"\nFetched {len(all_new_reviews):,} reviews for {movies_with_reviews} movies")
        
        if all_new_reviews:
            avg_reviews = len(all_new_reviews) / movies_with_reviews if movies_with_reviews > 0 else 0
            print(f"Average reviews per movie: {avg_reviews:.1f}")
    else:
        all_new_reviews = []
        print("\n[Skipping reviews as requested]")
    
    # Check for new reviews on existing recent movies
    if not args.skip_reviews and not args.skip_new_review_check and len(existing_movies_df) > 0:
        print("\n" + "="*60)
        print("STEP 3: CHECK FOR NEW REVIEWS ON EXISTING MOVIES")
        print("="*60)
        
        additional_reviews = check_for_new_reviews(
            movies_df=existing_movies_df,
            reviews_df=existing_reviews_df,
            months_back=args.months_back,
            delay=args.delay
        )
        
        if additional_reviews:
            all_new_reviews.extend(additional_reviews)
            print(f"Total new reviews (new movies + existing movies): {len(all_new_reviews)}")
    
    # Save updated data
    print("\n" + "="*60)
    print("STEP 4: SAVE UPDATED DATA")
    print("="*60)
    
    # Create backup of movies if we have new ones
    if new_movies:
        if os.path.exists(movies_path):
            backup_path = movies_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d")}.csv')
            print(f"Backing up movies to: {backup_path}")
            existing_movies_df.to_csv(backup_path, index=False)
        
        # Add new movies
        new_movies_df = pd.DataFrame(new_movies)
        
        # Ensure columns match
        if len(existing_movies_df) > 0:
            for col in existing_movies_df.columns:
                if col not in new_movies_df.columns:
                    new_movies_df[col] = None
            new_movies_df = new_movies_df[existing_movies_df.columns]
        
        combined_movies_df = pd.concat([existing_movies_df, new_movies_df], ignore_index=True)
        combined_movies_df.to_csv(movies_path, index=False)
        print(f"Saved {len(combined_movies_df):,} movies to {movies_path}")
    else:
        combined_movies_df = existing_movies_df
        print("No new movies to add.")
    
    # Add new reviews
    if all_new_reviews:
        new_reviews_df = pd.DataFrame(all_new_reviews)
        
        if len(existing_reviews_df) > 0:
            for col in existing_reviews_df.columns:
                if col not in new_reviews_df.columns:
                    new_reviews_df[col] = None
            new_reviews_df = new_reviews_df[existing_reviews_df.columns]
        
        combined_reviews_df = pd.concat([existing_reviews_df, new_reviews_df], ignore_index=True)
        combined_reviews_df.to_csv(reviews_path, index=False)
        print(f"Saved {len(combined_reviews_df):,} reviews to {reviews_path}")
    else:
        combined_reviews_df = existing_reviews_df
        print("No new reviews to add.")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"New movies added: {len(new_movies)}")
    print(f"New reviews added: {len(all_new_reviews)}")
    print(f"Total movies: {len(combined_movies_df):,}")
    print(f"Total reviews: {len(combined_reviews_df):,}")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()