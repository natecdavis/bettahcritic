"""
Incremental Metacritic Scraper

Walks the newest-first movie listing back to the --months-back cutoff:
- Movies not yet in the dataset are added (and their reviews fetched).
- Movies already in the dataset get their metadata refreshed in place
  (metascore, release_date, etc.), since Metacritic keeps updating these
  after our first scrape.

Movies are identified by slug alone: Metacritic slugs are unique page IDs,
so a slug that reappears with different metadata is the same film updated,
not a new film.

Usage:
    # Dry run - see what would be fetched
    python scrape_new_movies.py --input-dir ./data/metacritic_data --dry-run

    # Actually fetch new data
    python scrape_new_movies.py --input-dir ./data/metacritic_data

    # Fetch more pages (if you haven't updated in a while)
    python scrape_new_movies.py --input-dir ./data/metacritic_data --max-pages 40
"""

import requests
import pandas as pd
import glob
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
            
            # Extract year from release date
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
        
        return movies
        
    except requests.RequestException as e:
        print(f"Error fetching movies page {page}: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing movies page {page}: {e}")
        return []


def _parse_review_items(items: list[dict], slug: str) -> list[dict]:
    """Parse raw API review items, keeping only scored reviews with an outlet."""
    reviews = []
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
            reviews.append(review)

    return reviews


def fetch_review_page(slug: str, offset: int = 0, delay: float = 0.3) -> tuple[list[dict], int, int]:
    """
    Fetch one page of critic reviews (the API caps pages at 10 items;
    page 0 is newest-first).

    Returns (parsed reviews, total review count, raw item count). The raw
    count can exceed the parsed count when unscored reviews are filtered out.
    """
    url = REVIEWS_API.format(slug=slug)
    params = {
        'offset': offset,
        'limit': 10,
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
        return _parse_review_items(items, slug), total_results, len(items)
    except requests.RequestException as e:
        print(f"  Error fetching reviews for {slug}: {e}")
        return [], 0, 0
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Error parsing reviews for {slug}: {e}")
        return [], 0, 0


def get_movie_reviews(slug: str, delay: float = 0.3) -> list[dict]:
    """
    Fetch all critic reviews for a movie.
    Paginates through all results.
    """
    all_reviews = []
    offset = 0

    while True:
        page_reviews, total_results, n_items = fetch_review_page(slug, offset=offset, delay=delay)
        all_reviews.extend(page_reviews)

        if n_items == 0:
            break
        offset += 10
        if offset >= total_results:
            break

    return all_reviews


def fetch_new_movies_incrementally(
    existing_slugs: set,
    months_back: int = 3,
    max_pages: int = 25,
    delay: float = 0.5
) -> tuple[list[dict], list[dict]]:
    """
    Walk the newest-first listing until release dates fall behind the cutoff.

    Args:
        existing_slugs: Set of movie slugs already in the database
        months_back: Walk back until every release date on a page is older than this
        max_pages: Safety cap on pages fetched
        delay: Delay between requests

    Returns:
        (new_movies, updated_movies): movies with unseen slugs, and fresh
        metadata for movies we already have.
    """
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
    new_movies = []
    updated_movies = []

    print(f"Walking movie listing back to {cutoff.date()} (max {max_pages} pages)...")

    for page in range(max_pages):
        print(f"  Page {page + 1}/{max_pages}...", end=" ")

        movies = get_recent_movies(page=page, delay=delay)

        if not movies:
            print("no results")
            break

        page_new = 0
        page_known = 0

        for movie in movies:
            if movie['movie_slug'] in existing_slugs:
                updated_movies.append(movie)
                page_known += 1
            else:
                new_movies.append(movie)
                existing_slugs.add(movie['movie_slug'])
                page_new += 1

        print(f"{page_new} new, {page_known} known (metadata refreshed)")

        # Stop once every dated movie on the page is older than the cutoff.
        # The listing is sorted newest-first, so nothing newer follows.
        page_dates = pd.to_datetime(
            [m['release_date'] for m in movies if m['release_date']],
            errors='coerce'
        ).dropna()
        if len(page_dates) > 0 and page_dates.max() < cutoff:
            print(f"  Reached movies older than {cutoff.date()}, stopping.")
            break

    return new_movies, updated_movies


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

    # Build a set of existing (movie_slug, outlet_slug, critic_slug) tuples for
    # deduplication. Missing values are normalized to '' so keys built from the
    # CSV (NaN) match keys built from fresh API responses ('').
    existing_reviews_set = set(zip(
        reviews_df['movie_slug'].fillna(''),
        reviews_df['outlet_slug'].fillna(''),
        reviews_df['critic_slug'].fillna(''),
    ))
    
    def review_key(review: dict) -> tuple:
        return (
            review['movie_slug'],
            review.get('outlet_slug') or '',
            review.get('critic_slug') or '',
        )

    new_reviews = []
    movies_with_new_reviews = 0
    full_fetches = 0

    for _, movie in tqdm(recent_movies.iterrows(), total=len(recent_movies), desc="Checking for new reviews"):
        slug = movie['movie_slug']
        movie_year = movie.get('year')
        existing_count = existing_review_counts.get(slug, 0)

        # Cheap gate: one request for the newest-first first page. Only fetch
        # the remaining pages when it shows an unseen review, or the API's
        # total exceeds what we have stored.
        page1, total_results, _ = fetch_review_page(slug, offset=0, delay=delay)
        page1_new = any(review_key(r) not in existing_reviews_set for r in page1)

        if not page1_new and total_results <= existing_count:
            continue

        if total_results > 10:
            current_reviews = get_movie_reviews(slug, delay=delay)
            full_fetches += 1
        else:
            current_reviews = page1

        movie_new_reviews = []
        for review in current_reviews:
            # Denormalized year column, kept consistent with movies.csv
            review['year'] = movie_year
            key = review_key(review)

            if key not in existing_reviews_set:
                movie_new_reviews.append(review)
                existing_reviews_set.add(key)  # Don't add duplicates within this run

        if movie_new_reviews:
            new_reviews.extend(movie_new_reviews)
            movies_with_new_reviews += 1

    print(f"Found {len(new_reviews)} new reviews across {movies_with_new_reviews} movies "
          f"({full_fetches} movies needed a full re-fetch)")

    return new_reviews


def main():
    parser = argparse.ArgumentParser(description='Incrementally fetch new movies and reviews')
    parser.add_argument('--input-dir', type=str, default='./data/metacritic_data',
                        help='Directory with movies.csv and reviews.csv')
    parser.add_argument('--max-pages', type=int, default=25,
                        help='Maximum pages of movies to check')
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
    parser.add_argument('--dedupe-only', action='store_true',
                        help='Only deduplicate existing data, do not fetch anything new')
    
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
        print(f"Found {len(existing_movies_df):,} existing movies")
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
    
    # Handle dedupe-only mode
    if args.dedupe_only:
        print("\n" + "="*60)
        print("DEDUPE-ONLY MODE")
        print("="*60)
        
        # Dedupe movies by slug; keep='last' so the most recently scraped
        # (freshest) metadata wins
        if len(existing_movies_df) > 0:
            before = len(existing_movies_df)
            existing_movies_df = existing_movies_df.drop_duplicates(subset=['movie_slug'], keep='last')
            after = len(existing_movies_df)
            if before != after:
                print(f"Removed {before - after} duplicate movies")
                existing_movies_df.to_csv(movies_path, index=False)
                print(f"Saved {len(existing_movies_df):,} movies")
            else:
                print("No duplicate movies found")

        # Dedupe reviews; normalize missing key fields so NaN and '' match
        if len(existing_reviews_df) > 0:
            before = len(existing_reviews_df)
            review_keys = existing_reviews_df[['movie_slug', 'outlet_slug', 'critic_slug']].fillna('')
            existing_reviews_df = existing_reviews_df.loc[~review_keys.duplicated(keep='last')]
            after = len(existing_reviews_df)
            if before != after:
                print(f"Removed {before - after} duplicate reviews")
                existing_reviews_df.to_csv(reviews_path, index=False)
                print(f"Saved {len(existing_reviews_df):,} reviews")
            else:
                print("No duplicate reviews found")
        
        print(f"\nCompleted: {datetime.now().isoformat()}")
        return
    
    # Fetch new movies
    print("\n" + "="*60)
    print("STEP 1: FETCH NEW MOVIES")
    print("="*60)
    
    new_movies, updated_movies = fetch_new_movies_incrementally(
        existing_slugs=existing_slugs,
        months_back=args.months_back,
        max_pages=args.max_pages,
        delay=args.delay
    )

    print(f"\nFound {len(new_movies)} new movies, {len(updated_movies)} known movies to refresh")
    
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
        print(f"[DRY RUN] Would refresh metadata for {len(updated_movies)} existing movies")
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
            movie_year = movie.get('year')
            reviews = get_movie_reviews(slug, delay=args.delay)
            
            if reviews:
                # Add year to each review
                for review in reviews:
                    review['year'] = movie_year
                
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
    
    # Combine and dedupe movies
    if new_movies or updated_movies:
        if os.path.exists(movies_path):
            backup_path = movies_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d")}.csv')
            print(f"Backing up movies to: {backup_path}")
            existing_movies_df.to_csv(backup_path, index=False)

            # Keep only the newest backups (dated names sort chronologically)
            backups = sorted(glob.glob(movies_path.replace('.csv', '_backup_*.csv')))
            for stale in backups[:-5]:
                os.remove(stale)
            if len(backups) > 5:
                print(f"Pruned {len(backups) - 5} old backups (kept 5 newest)")

    if new_movies:
        new_movies_df = pd.DataFrame(new_movies)

        if len(existing_movies_df) > 0:
            for col in existing_movies_df.columns:
                if col not in new_movies_df.columns:
                    new_movies_df[col] = None
            new_movies_df = new_movies_df[existing_movies_df.columns]

        combined_movies_df = pd.concat([existing_movies_df, new_movies_df], ignore_index=True)
    else:
        combined_movies_df = existing_movies_df.copy()

    # Always dedupe movies by slug; keep='last' so freshest metadata wins
    if len(combined_movies_df) > 0:
        before_dedupe = len(combined_movies_df)
        combined_movies_df = combined_movies_df.drop_duplicates(subset=['movie_slug'], keep='last')
        after_dedupe = len(combined_movies_df)
        dupes_removed = before_dedupe - after_dedupe

        # Refresh metadata of movies we re-encountered on the listing pages
        if updated_movies:
            updates_df = pd.DataFrame(updated_movies).drop_duplicates(subset=['movie_slug'], keep='last')
            updates_df = updates_df.set_index('movie_slug')
            updates_df = updates_df[[c for c in updates_df.columns if c in combined_movies_df.columns]]
            combined_movies_df = combined_movies_df.set_index('movie_slug')
            combined_movies_df.update(updates_df)
            combined_movies_df = combined_movies_df.reset_index()
            print(f"Refreshed metadata for {len(updates_df)} existing movies")

        if new_movies or updated_movies or dupes_removed > 0:
            if dupes_removed > 0:
                print(f"Removed {dupes_removed} duplicate movies")
            combined_movies_df.to_csv(movies_path, index=False)
            print(f"Saved {len(combined_movies_df):,} movies to {movies_path}")
        else:
            print("No new movies to add.")
    
    # Combine and dedupe reviews
    if all_new_reviews:
        new_reviews_df = pd.DataFrame(all_new_reviews)
        
        if len(existing_reviews_df) > 0:
            for col in existing_reviews_df.columns:
                if col not in new_reviews_df.columns:
                    new_reviews_df[col] = None
            new_reviews_df = new_reviews_df[existing_reviews_df.columns]
        
        combined_reviews_df = pd.concat([existing_reviews_df, new_reviews_df], ignore_index=True)
    else:
        combined_reviews_df = existing_reviews_df.copy()
    
    # Always dedupe reviews on (movie, outlet, critic), normalizing missing
    # key fields so NaN (from CSV) and '' (from fresh scrapes) match
    if len(combined_reviews_df) > 0:
        before_dedupe = len(combined_reviews_df)
        review_keys = combined_reviews_df[['movie_slug', 'outlet_slug', 'critic_slug']].fillna('')
        combined_reviews_df = combined_reviews_df.loc[~review_keys.duplicated(keep='last')]
        after_dedupe = len(combined_reviews_df)
        dupes_removed = before_dedupe - after_dedupe
        
        if all_new_reviews or dupes_removed > 0:
            if dupes_removed > 0:
                print(f"Removed {dupes_removed} duplicate reviews")
            combined_reviews_df.to_csv(reviews_path, index=False)
            print(f"Saved {len(combined_reviews_df):,} reviews to {reviews_path}")
        else:
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