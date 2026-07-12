"""
Full Metacritic Scraper

Fetches ALL movies and reviews from Metacritic from scratch.
This is a long-running process - expect several hours for ~28k movies.

Usage:
    # Full scrape (will take hours)
    python scrape_full.py --output-dir ./data/metacritic_data
    
    # Resume from where you left off (if interrupted)
    python scrape_full.py --output-dir ./data/metacritic_data --resume
    
    # Just fetch movies (no reviews)
    python scrape_full.py --output-dir ./data/metacritic_data --skip-reviews
    
    # Fetch reviews for existing movies.csv
    python scrape_full.py --output-dir ./data/metacritic_data --reviews-only
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


def get_movies_page(page: int = 0, page_size: int = 24, delay: float = 0.3) -> tuple[list[dict], int]:
    """
    Fetch a page of movies from Metacritic.
    
    Returns (list of movie dicts, total count)
    """
    params = {
        'sortBy': '-releaseDate',
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
        
        total_count = data.get('data', {}).get('totalResults', 0)
        items = data.get('data', {}).get('items', [])
        
        movies = []
        for item in items:
            # Handle genres
            genres = item.get('genres', [])
            if genres and isinstance(genres[0], dict):
                genre_str = ', '.join(g.get('name', '') for g in genres if g.get('name'))
            elif genres:
                genre_str = ', '.join(genres)
            else:
                genre_str = ''
            
            # Extract year
            release_date = item.get('releaseDate', '')
            year = None
            if release_date:
                try:
                    year = int(release_date[:4])
                except:
                    pass
            
            movie = {
                'movie_slug': item.get('slug', ''),
                'title': item.get('title', ''),
                'release_date': release_date,
                'year': year,
                'metascore': item.get('criticScoreSummary', {}).get('score'),
                'user_score': item.get('userScoreSummary', {}).get('score'),
                'genre': genre_str,
                'rating': item.get('rating', ''),
                'description': item.get('description', '')[:500] if item.get('description') else '',
            }
            
            if movie['movie_slug']:
                movies.append(movie)
        
        return movies, total_count
        
    except requests.RequestException as e:
        print(f"\nError fetching movies page {page}: {e}")
        return [], 0
    except (json.JSONDecodeError, KeyError) as e:
        print(f"\nError parsing movies page {page}: {e}")
        return [], 0


def get_movie_reviews(slug: str, year: int = None, delay: float = 0.3) -> list[dict]:
    """
    Fetch all critic reviews for a movie.
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
            'sort': 'date',
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
                    'year': year,
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
            print(f"\n  Error fetching reviews for {slug}: {e}")
            break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"\n  Error parsing reviews for {slug}: {e}")
            break
    
    return all_reviews


def fetch_all_movies(output_dir: str, delay: float = 0.3, resume: bool = False) -> pd.DataFrame:
    """
    Fetch all movies from Metacritic.
    """
    movies_path = os.path.join(output_dir, 'movies.csv')
    
    # Check for existing data if resuming
    existing_movies = []
    start_page = 0
    
    if resume and os.path.exists(movies_path):
        print(f"Resuming from existing {movies_path}...")
        existing_df = pd.read_csv(movies_path)
        existing_movies = existing_df.to_dict('records')
        # Estimate where we left off (24 movies per page)
        start_page = len(existing_movies) // 24
        print(f"Found {len(existing_movies)} existing movies, starting from page {start_page}")
    
    # Get first page to find total count
    print("Fetching movie count...")
    _, total_count = get_movies_page(page=0, delay=delay)
    
    if total_count == 0:
        print("ERROR: Could not get movie count from API")
        return pd.DataFrame()
    
    total_pages = (total_count // 24) + 1
    print(f"Total movies: {total_count:,}")
    print(f"Total pages: {total_pages:,}")
    
    all_movies = existing_movies.copy()
    existing_slugs = set(m['movie_slug'] for m in existing_movies)
    
    print(f"\nFetching movies from page {start_page} to {total_pages}...")
    
    for page in tqdm(range(start_page, total_pages), initial=start_page, total=total_pages, desc="Pages"):
        movies, _ = get_movies_page(page=page, delay=delay)
        
        for movie in movies:
            slug = movie['movie_slug']
            if slug not in existing_slugs:
                all_movies.append(movie)
                existing_slugs.add(slug)
        
        # Save progress every 50 pages
        if page > 0 and page % 50 == 0:
            df = pd.DataFrame(all_movies)
            df.to_csv(movies_path, index=False)
            tqdm.write(f"  Saved {len(all_movies):,} movies")
    
    # Final save; keep='last' so the most recently fetched metadata wins
    df = pd.DataFrame(all_movies)
    df = df.drop_duplicates(subset=['movie_slug'], keep='last')
    df.to_csv(movies_path, index=False)
    
    print(f"\nSaved {len(df):,} movies to {movies_path}")
    return df


def fetch_all_reviews(movies_df: pd.DataFrame, output_dir: str, delay: float = 0.3, resume: bool = False) -> pd.DataFrame:
    """
    Fetch reviews for all movies.
    """
    reviews_path = os.path.join(output_dir, 'reviews.csv')
    progress_path = os.path.join(output_dir, '.reviews_progress.txt')
    
    # Check for existing progress if resuming
    existing_reviews = []
    completed_slugs = set()
    
    if resume:
        if os.path.exists(reviews_path):
            print(f"Loading existing reviews from {reviews_path}...")
            existing_df = pd.read_csv(reviews_path)
            existing_reviews = existing_df.to_dict('records')
            completed_slugs = set(existing_df['movie_slug'].unique())
            print(f"Found {len(existing_reviews):,} existing reviews for {len(completed_slugs):,} movies")
        
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                for line in f:
                    completed_slugs.add(line.strip())
    
    # Filter to movies that need reviews
    movies_to_fetch = movies_df[~movies_df['movie_slug'].isin(completed_slugs)]
    
    print(f"\nMovies to fetch reviews for: {len(movies_to_fetch):,}")
    print(f"Movies already completed: {len(completed_slugs):,}")
    
    if len(movies_to_fetch) == 0:
        print("All movies already have reviews fetched!")
        return pd.read_csv(reviews_path) if os.path.exists(reviews_path) else pd.DataFrame()
    
    all_reviews = existing_reviews.copy()
    
    # Open progress file for appending
    with open(progress_path, 'a') as progress_file:
        for idx, (_, movie) in enumerate(tqdm(movies_to_fetch.iterrows(), total=len(movies_to_fetch), desc="Movies")):
            slug = movie['movie_slug']
            year = movie.get('year')
            
            reviews = get_movie_reviews(slug, year=year, delay=delay)
            all_reviews.extend(reviews)
            
            # Mark as completed
            progress_file.write(f"{slug}\n")
            progress_file.flush()
            
            # Save progress every 100 movies
            if (idx + 1) % 100 == 0:
                df = pd.DataFrame(all_reviews)
                df.to_csv(reviews_path, index=False)
                tqdm.write(f"  Saved {len(all_reviews):,} reviews ({idx + 1}/{len(movies_to_fetch)} movies)")
    
    # Final save; dedupe on (movie, outlet, critic) with missing keys
    # normalized so NaN (from CSV) and '' (from fresh scrapes) match
    df = pd.DataFrame(all_reviews)
    review_keys = df[['movie_slug', 'outlet_slug', 'critic_slug']].fillna('')
    df = df.loc[~review_keys.duplicated(keep='last')]
    df.to_csv(reviews_path, index=False)
    
    # Clean up progress file
    if os.path.exists(progress_path):
        os.remove(progress_path)
    
    print(f"\nSaved {len(df):,} reviews to {reviews_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Full Metacritic scraper')
    parser.add_argument('--output-dir', type=str, default='./data/metacritic_data',
                        help='Output directory for movies.csv and reviews.csv')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay between API requests')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing progress')
    parser.add_argument('--skip-reviews', action='store_true',
                        help='Only fetch movies, skip reviews')
    parser.add_argument('--reviews-only', action='store_true',
                        help='Only fetch reviews for existing movies.csv')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FULL METACRITIC SCRAPER")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    movies_path = os.path.join(args.output_dir, 'movies.csv')
    
    # Fetch movies (unless reviews-only mode)
    if args.reviews_only:
        if not os.path.exists(movies_path):
            print(f"ERROR: {movies_path} not found. Cannot do reviews-only mode.")
            return
        print(f"Loading existing movies from {movies_path}...")
        movies_df = pd.read_csv(movies_path)
        print(f"Found {len(movies_df):,} movies")
    else:
        print("\n" + "="*60)
        print("STEP 1: FETCH ALL MOVIES")
        print("="*60)
        movies_df = fetch_all_movies(args.output_dir, delay=args.delay, resume=args.resume)
    
    if len(movies_df) == 0:
        print("No movies found. Exiting.")
        return
    
    # Fetch reviews (unless skip-reviews mode)
    if not args.skip_reviews:
        print("\n" + "="*60)
        print("STEP 2: FETCH ALL REVIEWS")
        print("="*60)
        reviews_df = fetch_all_reviews(movies_df, args.output_dir, delay=args.delay, resume=args.resume)
    else:
        print("\n[Skipping reviews as requested]")
        reviews_df = pd.DataFrame()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total movies: {len(movies_df):,}")
    print(f"Total reviews: {len(reviews_df):,}")
    if len(movies_df) > 0 and len(reviews_df) > 0:
        movies_with_reviews = reviews_df['movie_slug'].nunique()
        print(f"Movies with reviews: {movies_with_reviews:,}")
        print(f"Avg reviews per movie: {len(reviews_df) / movies_with_reviews:.1f}")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()