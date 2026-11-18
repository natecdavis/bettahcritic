"""
Metacritic scraper for the non-movie categories: TV, games, music.

Movies keep their own scraper (scrape_new_movies.py); this one writes the
same two-file layout for each other category under data/<category>/:

    data/<category>/metacritic_data/items.csv     one row per scored item
    data/<category>/metacritic_data/reviews.csv   one row per critic review

What counts as an "item" differs by category, because that is the unit
Metacritic assigns a metascore to:
  - games: a game on its primary platform. The reviews API only serves the
    default platform's reviews, and the metascore we keep is that platform's.
  - music: an album. Keyed "<album-slug>/<artist-slug>" (album slugs alone
    collide across artists).
  - tv: a *season*. Keyed "<show-slug>/<season-slug>". The show endpoint mixes
    reviews from all seasons; each review is tagged with its season, and each
    season has its own metascore on the show page.

Both CSVs key items by `item_slug`. The downstream scripts accept either
`item_slug` or the movies pipeline's `movie_slug`.

Modes:
    --mode full         Walk the whole scored catalog (sorted by metascore) and
                        fetch every item's reviews. Resumable: items already in
                        items.csv are skipped, progress is appended in chunks.
    --mode incremental  Daily: walk the newest-first listing back --months-back
                        months, plus stored items released in that window.
                        New items get their reviews; known ones get metadata
                        refreshed and are checked for new reviews.

Usage:
    python scrape_category.py --category games --mode full
    python scrape_category.py --category tv --mode incremental --months-back 3
"""

import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

FINDER_API = "https://backend.metacritic.com/finder/metacritic/web"
REVIEWS_API = "https://backend.metacritic.com/reviews/metacritic/critic/{path}/web"
SHOW_PAGE_API = "https://backend.metacritic.com/composer/metacritic/pages/shows/{slug}/web"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.metacritic.com/',
    'Origin': 'https://www.metacritic.com',
}

# Finder productType and reviews-API path segment per category
CATEGORIES = {
    'games': {'product_type': 'games', 'review_kind': 'games'},
    'music': {'product_type': 'music', 'review_kind': 'music'},
    'tv': {'product_type': 'tv', 'review_kind': 'shows'},
}

ITEM_COLUMNS = ['item_slug', 'metacritic_id', 'title', 'subtitle', 'release_date', 'year',
                'metascore', 'user_score', 'listed_review_count', 'genre', 'url']
REVIEW_COLUMNS = ['item_slug', 'score', 'outlet', 'outlet_slug', 'critic', 'critic_slug',
                  'date', 'platform', 'excerpt', 'review_url']

LISTING_PAGE_SIZE = 100   # the finder API accepts up to 100
REVIEW_PAGE_SIZE = 10     # the reviews API ignores larger limits
MAX_ATTEMPTS = 4

_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, 'session'):
        s = requests.Session()
        s.headers.update(HEADERS)
        _thread_local.session = s
    return _thread_local.session


def get_json(url: str, params: dict, delay: float):
    """GET with retry/backoff on transient failures. Returns parsed JSON, or
    None on a permanent failure (404) or after exhausting retries."""
    for attempt in range(MAX_ATTEMPTS):
        time.sleep(delay + (2 ** attempt + random.random() if attempt else 0))
        try:
            r = _session().get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == MAX_ATTEMPTS - 1:
                tqdm.write(f"  giving up on {url}: {e}")
    return None


# ---------------------------------------------------------------- listings

def _genre_str(item: dict) -> str:
    genres = item.get('genres') or []
    names = [g.get('name', '') if isinstance(g, dict) else str(g) for g in genres]
    return ', '.join(n for n in names if n)


def fetch_listing_page(category: str, offset: int, sort_by: str, delay: float,
                       extra_params: dict | None = None) -> list[dict]:
    params = {
        'sortBy': sort_by,
        'productType': CATEGORIES[category]['product_type'],
        'limit': LISTING_PAGE_SIZE,
        'offset': offset,
        **(extra_params or {}),
    }
    data = get_json(FINDER_API, params, delay)
    if not data:
        return []
    return data.get('data', {}).get('items', []) or []


def listing_entry(category: str, item: dict) -> dict:
    """Normalize a finder item into what the per-item fetchers need."""
    summary = item.get('criticScoreSummary') or {}
    entry = {
        'slug': item.get('slug', ''),
        'metacritic_id': item.get('id'),
        'title': item.get('title', ''),
        'release_date': item.get('releaseDate') or '',
        'metascore': summary.get('score'),
        'listed_review_count': summary.get('reviewCount'),
        'user_score': (item.get('userScore') or {}).get('score'),
        'genre': _genre_str(item),
        'platforms': [p.get('name') for p in (item.get('platforms') or []) if p.get('name')],
    }
    if category == 'music':
        artists = item.get('artists') or []
        entry['artist_slug'] = item.get('artistSlug') or (artists[0].get('slug') if artists else '')
        entry['artist'] = ', '.join(a.get('name', '') for a in artists if a.get('name'))
    return entry


def listing_key(category: str, entry: dict) -> str:
    """The unit we fetch reviews for: show slug for TV (expands to seasons)."""
    if category == 'music':
        return f"{entry['slug']}/{entry['artist_slug']}"
    return entry['slug']


def walk_listing_full(category: str, delay: float) -> list[dict]:
    """Every item with a metascore, walking the -metaScore listing until scores run out."""
    entries, offset = [], 0
    pbar = tqdm(desc=f"{category} listing", unit="items")
    while True:
        page = fetch_listing_page(category, offset, '-metaScore', delay)
        if not page:
            break
        scored = [listing_entry(category, it) for it in page]
        scored = [e for e in scored if e['slug'] and e['metascore'] is not None]
        entries.extend(scored)
        pbar.update(len(page))
        # Sorted by score, so a page with nothing scored means the unscored
        # tail has begun. (A single unscored item can appear mid-listing, so
        # a partly scored page is not the end.)
        if not scored:
            break
        offset += LISTING_PAGE_SIZE
    pbar.close()
    return entries


def walk_listing_recent(category: str, months_back: int, delay: float, max_pages: int) -> list[dict]:
    """Items released since the cutoff, newest first. Unscored items are kept for
    TV (a show's newest season may be unscored while older ones are scored)."""
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
    entries = []

    if category == 'games':
        # The games release-date listing is ~100 unscored titles a day, so a
        # date walk would need hundreds of pages. Instead, list scored games
        # from the cutoff's year onward (the year filter + metascore sort
        # returns scored titles only) and keep those released since the cutoff.
        years = {'releaseYearMin': cutoff.year, 'releaseYearMax': pd.Timestamp.now().year + 1}
        for page_no in range(max_pages):
            page = fetch_listing_page(category, page_no * LISTING_PAGE_SIZE, '-metaScore',
                                      delay, years)
            page_entries = [listing_entry(category, it) for it in page]
            scored = [e for e in page_entries if e['slug'] and e['metascore'] is not None]
            entries.extend(e for e in scored
                           if pd.to_datetime(e['release_date'], errors='coerce') >= cutoff)
            if not scored:
                break
        return entries

    for page_no in range(max_pages):
        page = fetch_listing_page(category, page_no * LISTING_PAGE_SIZE, '-releaseDate', delay)
        if not page:
            break
        page_entries = [listing_entry(category, it) for it in page]
        entries.extend(e for e in page_entries
                       if e['slug'] and (e['metascore'] is not None or category == 'tv'))
        dates = pd.to_datetime([e['release_date'] for e in page_entries if e['release_date']],
                               errors='coerce').dropna()
        if len(dates) and dates.max() < cutoff:
            break
    return entries


# ----------------------------------------------------------------- reviews

def parse_review(item_slug: str, raw: dict) -> dict | None:
    if raw.get('score') is None or not raw.get('publicationName'):
        return None
    return {
        'item_slug': item_slug,
        'score': raw.get('score'),
        'outlet': raw.get('publicationName', ''),
        'outlet_slug': raw.get('publicationSlug') or '',
        'critic': raw.get('author') or '',
        'critic_slug': raw.get('authorSlug') or '',
        'date': raw.get('date'),
        'platform': raw.get('platform') or '',
        'excerpt': (raw.get('quote') or '')[:300],
        'review_url': raw.get('url') or '',
    }


def fetch_review_page(category: str, path: str, offset: int, delay: float):
    """Returns (raw items, totalResults); ([], 0) on failure."""
    params = {
        'offset': offset,
        'limit': REVIEW_PAGE_SIZE,
        'filterBySentiment': 'all',
        'sort': 'date',
        'componentName': 'critic-reviews',
        'componentDisplayName': 'Critic Reviews',
        'componentType': 'ReviewList',
    }
    url = REVIEWS_API.format(path=f"{CATEGORIES[category]['review_kind']}/{path}")
    data = get_json(url, params, delay)
    if not data:
        return [], 0
    d = data.get('data', {}) or {}
    return d.get('items', []) or [], d.get('totalResults', 0) or 0


def fetch_all_reviews(category: str, path: str, delay: float, first_page=None) -> list[dict]:
    raw, total = first_page if first_page else fetch_review_page(category, path, 0, delay)
    items = list(raw)
    offset = REVIEW_PAGE_SIZE
    while raw and offset < total:
        raw, _ = fetch_review_page(category, path, offset, delay)
        items.extend(raw)
        offset += REVIEW_PAGE_SIZE
    return items


def fetch_show_seasons(slug: str, delay: float) -> list[dict]:
    """Per-season metadata and metascores from the show page."""
    data = get_json(SHOW_PAGE_API.format(slug=slug), {'contentOnly': 'true'}, delay)
    if not data:
        return []
    for comp in data.get('components', []):
        if (comp.get('meta') or {}).get('componentName') == 'seasons':
            return (comp.get('data') or {}).get('items', []) or []
    return []


# ------------------------------------------------------- per-item fetching

def build_items_and_reviews(category: str, entry: dict, raw_reviews: list[dict],
                            seasons: list[dict] | None) -> tuple[list[dict], list[dict]]:
    """Turn one listing entry + its raw reviews into item rows and review rows."""
    key = listing_key(category, entry)

    if category != 'tv':
        reviews = [r for r in (parse_review(key, x) for x in raw_reviews) if r]
        if category == 'games':
            platforms = [r['platform'] for r in reviews if r['platform']]
            subtitle = (pd.Series(platforms).mode().iloc[0] if platforms
                        else (entry['platforms'][0] if entry['platforms'] else ''))
            url = f"/game/{entry['slug']}/"
        else:
            subtitle = entry.get('artist', '')
            url = f"/music/{entry['slug']}/{entry['artist_slug']}/"
        item = {
            'item_slug': key,
            'metacritic_id': entry['metacritic_id'],
            'title': entry['title'],
            'subtitle': subtitle,
            'release_date': entry['release_date'],
            'year': int(entry['release_date'][:4]) if entry['release_date'][:4].isdigit() else None,
            'metascore': entry['metascore'],
            'user_score': entry['user_score'],
            'listed_review_count': entry['listed_review_count'],
            'genre': entry['genre'],
            'url': url,
        }
        return [item], reviews

    # TV: expand the show into scored seasons; route each review by season id
    items, reviews = [], []
    season_by_id = {}
    for s in seasons or []:
        summary = s.get('criticScoreSummary') or {}
        season_slug = s.get('seasonSlug') or f"season-{s.get('seasonNumber')}"
        season_id = ((s.get('tvTaxonomy') or {}).get('season') or {}).get('id')
        item_slug = f"{entry['slug']}/{season_slug}"
        if season_id is not None:
            season_by_id[season_id] = item_slug
        if summary.get('score') is None:
            continue
        premiere = s.get('premiereDate') or ''
        if not premiere and s.get('releaseYear'):
            premiere = f"{s['releaseYear']}-01-01"
        if not premiere and s.get('seasonNumber') == 1:
            premiere = entry['release_date']
        n = s.get('seasonNumber')
        items.append({
            'item_slug': item_slug,
            'metacritic_id': season_id,
            'title': entry['title'],
            'subtitle': f"Season {n}" if n is not None else season_slug,
            'release_date': premiere,
            'year': int(premiere[:4]) if premiere[:4].isdigit() else None,
            'metascore': summary.get('score'),
            'user_score': None,
            'listed_review_count': summary.get('reviewCount'),
            'genre': entry['genre'],
            'url': f"/tv/{entry['slug']}/{season_slug}/",
        })
    for raw in raw_reviews:
        season_id = (((raw.get('reviewedProduct') or {}).get('tvTaxonomy') or {})
                     .get('season') or {}).get('id')
        item_slug = season_by_id.get(season_id)
        if item_slug:
            r = parse_review(item_slug, raw)
            if r:
                reviews.append(r)
    return items, reviews


def fetch_entry(category: str, entry: dict, delay: float):
    """Full fetch of one listing entry (all review pages, plus seasons for TV)."""
    path = listing_key(category, entry)
    seasons = fetch_show_seasons(entry['slug'], delay) if category == 'tv' else None
    raw = fetch_all_reviews(category, path, delay)
    return build_items_and_reviews(category, entry, raw, seasons)


def refresh_entry(category: str, entry: dict, delay: float, stored_counts: dict,
                  stored_keys: set):
    """Incremental check: one review page first; full fetch only when it shows
    an unseen review or the API total exceeds what we have stored."""
    path = listing_key(category, entry)
    seasons = fetch_show_seasons(entry['slug'], delay) if category == 'tv' else None
    first_raw, total = fetch_review_page(category, path, 0, delay)
    items, first_reviews = build_items_and_reviews(category, entry, first_raw, seasons)
    stored_total = sum(stored_counts.get(it['item_slug'], 0) for it in items)
    unseen = any(review_key(r) not in stored_keys for r in first_reviews)
    if not unseen and total <= stored_total:
        return items, []
    raw = fetch_all_reviews(category, path, delay, first_page=(first_raw, total))
    return build_items_and_reviews(category, entry, raw, seasons)


def review_key(r) -> tuple:
    return (r['item_slug'], r.get('outlet_slug') or '', r.get('critic_slug') or '',
            r.get('outlet') or '')


# ------------------------------------------------------------------ storage

def load_csv(path: str, columns: list[str]) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame(columns=columns)


def append_rows(path: str, rows: list[dict], columns: list[str]):
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=columns)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def dedupe_reviews(df: pd.DataFrame) -> pd.DataFrame:
    keys = df[['item_slug', 'outlet_slug', 'critic_slug', 'outlet']].fillna('')
    return df.loc[~keys.duplicated(keep='last')]


def run_full(category: str, data_dir: str, workers: int, delay: float, chunk: int, limit: int | None):
    items_path = os.path.join(data_dir, 'items.csv')
    reviews_path = os.path.join(data_dir, 'reviews.csv')
    done_path = os.path.join(data_dir, '.full_scrape_done.txt')

    entries = walk_listing_full(category, delay)
    print(f"{len(entries):,} scored {category} entries in the listing")

    # Resume: listing keys already fetched (TV seasons map back to shows)
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = {line.strip() for line in f if line.strip()}
    todo = [e for e in entries if listing_key(category, e) not in done]
    # De-duplicate within the listing (finder occasionally repeats an item)
    seen, unique = set(), []
    for e in todo:
        k = listing_key(category, e)
        if k not in seen:
            seen.add(k)
            unique.append(e)
    todo = unique[:limit] if limit else unique
    print(f"{len(done):,} already fetched; {len(todo):,} to fetch with {workers} workers")

    pbar = tqdm(total=len(todo), desc=f"{category} reviews", unit="item")
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        item_rows, review_rows, keys = [], [], []
        with ThreadPoolExecutor(workers) as ex:
            futures = {ex.submit(fetch_entry, category, e, delay): e for e in batch}
            for fut in as_completed(futures):
                e = futures[fut]
                try:
                    items, reviews = fut.result()
                except Exception as exc:  # keep the run alive; retried on resume
                    tqdm.write(f"  failed {listing_key(category, e)}: {exc}")
                    continue
                # An item with a metascore but zero fetched reviews is a fetch
                # failure; leave it out of `done` so a resume retries it
                if items and not reviews:
                    pbar.update(1)
                    continue
                item_rows.extend(items)
                review_rows.extend(reviews)
                keys.append(listing_key(category, e))
                pbar.update(1)
        append_rows(items_path, item_rows, ITEM_COLUMNS)
        append_rows(reviews_path, review_rows, REVIEW_COLUMNS)
        with open(done_path, 'a') as f:
            f.writelines(k + '\n' for k in keys)
    pbar.close()
    finalize(items_path, reviews_path)


def run_incremental(category: str, data_dir: str, workers: int, delay: float,
                    months_back: int, max_pages: int):
    items_path = os.path.join(data_dir, 'items.csv')
    reviews_path = os.path.join(data_dir, 'reviews.csv')
    items_df = load_csv(items_path, ITEM_COLUMNS)
    reviews_df = load_csv(reviews_path, REVIEW_COLUMNS)

    entries = walk_listing_recent(category, months_back, delay, max_pages)
    print(f"{len(entries):,} {category} entries released in the last {months_back} months")

    # Stored items released in the window may have dropped off the listing walk
    # (e.g. a TV season premiering after the show's listing date): re-check them
    # too. Their listing entry is rebuilt from what we stored.
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
    listed = {listing_key(category, e) for e in entries}
    recent = items_df[pd.to_datetime(items_df['release_date'], errors='coerce') >= cutoff]
    for _, row in recent.iterrows():
        slug = row['item_slug'].split('/')[0] if category == 'tv' else row['item_slug']
        if category == 'music':
            album, artist = row['item_slug'].split('/', 1)
            entry_key = row['item_slug']
        else:
            album, artist, entry_key = slug, '', slug
        if entry_key in listed:
            continue
        listed.add(entry_key)
        entries.append({
            'slug': album, 'artist_slug': artist, 'artist': row.get('subtitle', ''),
            'metacritic_id': row.get('metacritic_id'), 'title': row['title'],
            'release_date': str(row['release_date']), 'metascore': row['metascore'],
            'listed_review_count': row.get('listed_review_count'),
            'user_score': row.get('user_score'), 'genre': row.get('genre', '') or '',
            'platforms': [row['subtitle']] if category == 'games' and pd.notna(row.get('subtitle')) else [],
        })

    stored_counts = reviews_df.groupby('item_slug').size().to_dict()
    stored_keys = set(map(review_key, reviews_df.to_dict('records')))

    new_items, new_reviews = [], []
    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(refresh_entry, category, e, delay, stored_counts, stored_keys)
                   for e in entries]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{category} refresh"):
            try:
                items, reviews = fut.result()
            except Exception as exc:
                tqdm.write(f"  failed: {exc}")
                continue
            new_items.extend(items)
            new_reviews.extend(r for r in reviews if review_key(r) not in stored_keys)

    known = set(items_df['item_slug'])
    n_new = sum(1 for it in new_items if it['item_slug'] not in known)
    print(f"{n_new} new items, {len(new_items) - n_new} refreshed, {len(new_reviews)} new reviews")

    if new_items:
        merged = pd.concat([items_df, pd.DataFrame(new_items)], ignore_index=True)
        merged = merged.drop_duplicates('item_slug', keep='last').reindex(columns=ITEM_COLUMNS)
        merged.to_csv(items_path, index=False)
    if new_reviews:
        merged = pd.concat([reviews_df, pd.DataFrame(new_reviews)], ignore_index=True)
        dedupe_reviews(merged).reindex(columns=REVIEW_COLUMNS).to_csv(reviews_path, index=False)
    print(f"Total: {len(load_csv(items_path, ITEM_COLUMNS)):,} items, "
          f"{len(load_csv(reviews_path, REVIEW_COLUMNS)):,} reviews")


def finalize(items_path: str, reviews_path: str):
    """Dedupe the appended files once a full pass completes."""
    if os.path.exists(items_path):
        items = load_csv(items_path, ITEM_COLUMNS).drop_duplicates('item_slug', keep='last')
        items.to_csv(items_path, index=False)
        print(f"{len(items):,} items in {items_path}")
    if os.path.exists(reviews_path):
        reviews = dedupe_reviews(load_csv(reviews_path, REVIEW_COLUMNS))
        reviews.to_csv(reviews_path, index=False)
        print(f"{len(reviews):,} reviews in {reviews_path}")


def main():
    p = argparse.ArgumentParser(description='Scrape Metacritic TV / games / music')
    p.add_argument('--category', required=True, choices=sorted(CATEGORIES))
    p.add_argument('--data-dir', help='Default: data/<category>/metacritic_data')
    p.add_argument('--mode', choices=['full', 'incremental'], default='incremental')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--delay', type=float, default=0.25, help='Per-request delay per worker')
    p.add_argument('--months-back', type=int, default=3)
    p.add_argument('--max-pages', type=int, default=60,
                   help='Incremental listing cap (100 items per page)')
    p.add_argument('--chunk', type=int, default=200, help='Full mode: items per checkpoint')
    p.add_argument('--limit', type=int, help='Full mode: fetch at most N entries (testing)')
    args = p.parse_args()

    data_dir = args.data_dir or os.path.join('data', args.category, 'metacritic_data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {args.category} {args.mode} -> {data_dir}")

    if args.mode == 'full':
        run_full(args.category, data_dir, args.workers, args.delay, args.chunk, args.limit)
    else:
        run_incremental(args.category, data_dir, args.workers, args.delay,
                        args.months_back, args.max_pages)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] done")


if __name__ == "__main__":
    main()
