"""
Shared loading for the scoring pipeline across categories.

The movies pipeline stores data/metacritic_data/{movies,reviews}.csv keyed by
`movie_slug`. The TV / games / music scraper stores
data/<category>/metacritic_data/{items,reviews}.csv keyed by `item_slug`.
The effects and scoring code was written against `movie_slug`, so items are
normalized to that column name on load (it means "item id" everywhere
downstream).
"""

import os

import pandas as pd


def resolve_items_path(input_dir: str, items_file: str | None = None) -> str:
    if items_file:
        return os.path.join(input_dir, items_file)
    for name in ('movies.csv', 'items.csv'):
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No movies.csv or items.csv in {input_dir}")


def load_items_and_reviews(input_dir: str, items_file: str | None = None,
                           exclude_release_types=None):
    """exclude_release_types: e.g. ['reissue', 'compilation', 'archival'] to
    drop those music releases (see release_types.py) and their reviews, so
    they are neither scored nor used to estimate outlet effects."""
    items = pd.read_csv(resolve_items_path(input_dir, items_file), low_memory=False)
    reviews = pd.read_csv(os.path.join(input_dir, 'reviews.csv'), low_memory=False)
    if exclude_release_types:
        from release_types import classify_items
        slug_col = 'item_slug' if 'item_slug' in items.columns else 'movie_slug'
        types = classify_items(items, slug_col)
        drop = types.isin(set(exclude_release_types))
        print(f"Excluding {int(drop.sum()):,} items by release type "
              f"({types[drop].value_counts().to_dict()})")
        dropped = set(items.loc[drop, slug_col])
        items = items.loc[~drop]
        reviews = reviews[~reviews[slug_col].isin(dropped)]
    items = items.rename(columns={'item_slug': 'movie_slug'})
    reviews = reviews.rename(columns={'item_slug': 'movie_slug'})
    # The new scraper writes '' for unnamed critics; the movie CSVs have NaN.
    # Downstream code treats NaN as "no critic", so normalize.
    for col in ('critic_slug', 'outlet_slug', 'critic'):
        if col in reviews.columns:
            reviews[col] = reviews[col].replace('', pd.NA)
    return items, reviews


def parse_type_list(value: str | None):
    return [t.strip() for t in value.split(',') if t.strip()] if value else None
