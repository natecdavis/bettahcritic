"""
Metacritic Adjusted Scores Calculator (v2)

Applies:
1. Hierarchical critic/outlet adjustment (critic effect with shrinkage to outlet)
2. Bayesian shrinkage toward grand mean

Does NOT apply:
- Genre adjustment (removed by request)

Usage:
    python metacritic_adjusted_scores_v2.py --input-dir ./metacritic_data \
        --effects-dir ./hierarchical_effects
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import argparse
from tqdm import tqdm


class ExponentialWeightedStats:
    """Compute exponentially weighted statistics with backward-looking only."""
    
    def __init__(self, halflife_days: float = 730):
        self.halflife = halflife_days
        self.decay = np.log(2) / halflife_days
    
    def get_weights(self, dates: pd.Series, as_of_date: pd.Timestamp) -> np.ndarray:
        days_ago = (as_of_date - dates).dt.days
        weights = np.where(days_ago > 0, np.exp(-self.decay * days_ago), 0)
        return weights
    
    def weighted_mean(self, values: np.ndarray, weights: np.ndarray) -> float:
        valid = ~np.isnan(values) & (weights > 0)
        if valid.sum() == 0:
            return np.nan
        return np.average(values[valid], weights=weights[valid])
    
    def weighted_var(self, values: np.ndarray, weights: np.ndarray) -> float:
        valid = ~np.isnan(values) & (weights > 0)
        if valid.sum() < 2:
            return np.nan
        mean = self.weighted_mean(values, weights)
        return np.average((values[valid] - mean) ** 2, weights=weights[valid])


def load_hierarchical_effects(effects_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load precomputed hierarchical critic and outlet effects."""
    
    critic_path = os.path.join(effects_dir, 'critic_effects_hierarchical.csv')
    outlet_path = os.path.join(effects_dir, 'outlet_effects_hierarchical.csv')
    
    print(f"Loading effects from {effects_dir}...")
    
    critic_effects = pd.read_csv(critic_path)
    critic_effects['date'] = pd.to_datetime(critic_effects['date'])
    print(f"  Loaded {len(critic_effects):,} critic effect observations")
    
    outlet_effects = pd.read_csv(outlet_path)
    outlet_effects['date'] = pd.to_datetime(outlet_effects['date'])
    print(f"  Loaded {len(outlet_effects):,} outlet effect observations")
    
    return critic_effects, outlet_effects


def compute_shrinkage_params_ewa(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    halflife_days: float = 2000
) -> pd.DataFrame:
    """
    Compute time-varying shrinkage parameters (sigma², tau², grand_mean).
    """
    ewa = ExponentialWeightedStats(halflife_days)
    
    movies = movies_df.copy()
    movies['date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies = movies.dropna(subset=['date', 'metascore'])
    
    # Compute within-movie variance for each movie
    movie_vars = reviews_df.groupby('movie_slug')['score'].var().dropna()
    movies = movies.merge(
        movie_vars.reset_index().rename(columns={'score': 'within_var'}),
        on='movie_slug',
        how='left'
    )
    
    # Quarterly time points
    date_range = pd.date_range(
        start=movies['date'].min(),
        end=movies['date'].max(),
        freq='Q'
    )
    
    results = []
    
    print(f"Computing shrinkage params at {len(date_range)} time points...")
    
    for as_of_date in tqdm(date_range, desc="Shrinkage params"):
        weights = ewa.get_weights(movies['date'], as_of_date)
        
        if weights.sum() < 50:
            continue
        
        # Grand mean
        grand_mean = ewa.weighted_mean(movies['metascore'].values, weights)
        
        # Tau² (between-movie variance)
        tau_sq = ewa.weighted_var(movies['metascore'].values, weights)
        
        # Sigma² (within-movie variance, averaged)
        valid_var = movies['within_var'].notna()
        sigma_sq = ewa.weighted_mean(
            movies.loc[valid_var, 'within_var'].values,
            weights[valid_var]
        )
        
        results.append({
            'date': as_of_date,
            'grand_mean': grand_mean,
            'tau_sq': tau_sq,
            'sigma_sq': sigma_sq,
        })
    
    return pd.DataFrame(results)


def process_all_movies(
    movies_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    critic_effects: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    shrinkage_params: pd.DataFrame,
    output_path: str
) -> pd.DataFrame:
    """
    Compute adjusted scores for all movies (vectorized).

    Pipeline per movie:
    1. Start with raw metascore
    2. Adjust each review for critic/outlet effect (as of the review date,
       critic effect preferred, outlet as fallback), then re-average
    3. Apply Bayesian shrinkage toward the time-varying grand mean

    Movies with no parseable release date keep their raw score.
    """
    # Filter to movies with metascores
    movies = movies_df.dropna(subset=['metascore']).copy()
    movies['_movie_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

    print(f"\nProcessing {len(movies):,} movies with metascores...")

    # --- Step 1: per-review effects, looked up as-of the review date ---
    reviews = reviews_df.loc[
        reviews_df['movie_slug'].isin(set(movies['movie_slug'])),
        ['movie_slug', 'score', 'outlet', 'critic_slug', 'date']
    ].copy()
    reviews['_review_date'] = pd.to_datetime(reviews['date'], errors='coerce')
    release_map = movies.set_index('movie_slug')['_movie_date']
    reviews['_review_date'] = reviews['_review_date'].fillna(reviews['movie_slug'].map(release_map))

    n_reviews_map = reviews.groupby('movie_slug').size()

    # Movies without a parseable release date are not adjusted (raw score kept),
    # so their reviews are skipped
    valid_slugs = set(movies.loc[movies['_movie_date'].notna(), 'movie_slug'])
    reviews = reviews[reviews['movie_slug'].isin(valid_slugs)]
    reviews = reviews.dropna(subset=['_review_date'])

    # As-of lookups (last effect with date <= review date) via merge_asof.
    # Missing by-keys are normalized to '' which matches no effect row.
    reviews = reviews.sort_values('_review_date', kind='stable')
    reviews['_critic_key'] = reviews['critic_slug'].fillna('')
    reviews['_outlet_key'] = reviews['outlet'].fillna('')

    ce = critic_effects.sort_values('date', kind='stable')[['critic', 'date', 'final_effect']]
    ce = ce.rename(columns={'critic': '_critic_key', 'date': '_review_date',
                            'final_effect': '_critic_eff'})
    reviews = pd.merge_asof(reviews, ce, on='_review_date', by='_critic_key',
                            direction='backward')

    oe = outlet_effects.sort_values('date', kind='stable')[['outlet', 'date', 'effect']]
    oe = oe.rename(columns={'outlet': '_outlet_key', 'date': '_review_date',
                            'effect': '_outlet_eff'})
    reviews = pd.merge_asof(reviews, oe, on='_review_date', by='_outlet_key',
                            direction='backward')

    # Critic effect preferred, outlet as fallback, else unadjusted
    has_critic = reviews['_critic_eff'].notna()
    has_outlet = reviews['_outlet_eff'].notna()
    effect = np.where(has_critic, reviews['_critic_eff'],
                      np.where(has_outlet, reviews['_outlet_eff'], 0.0))
    reviews['_adj_score'] = reviews['score'] - effect
    reviews['_src_critic'] = has_critic
    reviews['_src_outlet'] = ~has_critic & has_outlet

    grp = reviews.groupby('movie_slug')
    per_movie = pd.DataFrame({
        '_adj_sum': grp['_adj_score'].sum(),
        '_adj_count': grp['_adj_score'].count(),
        '_n_rows': grp.size(),
        '_n_critic': grp['_src_critic'].sum(),
        '_n_outlet': grp['_src_outlet'].sum(),
    })
    # np.mean semantics: any NaN review score makes the movie's mean NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        per_movie['_adj_mean'] = np.where(
            per_movie['_adj_count'] == per_movie['_n_rows'],
            per_movie['_adj_sum'] / per_movie['_adj_count'],
            np.nan,
        )

    results = movies.merge(per_movie, left_on='movie_slug', right_index=True, how='left')
    results = results.reset_index(drop=True)
    results['n_reviews'] = results['movie_slug'].map(n_reviews_map).fillna(0).astype(int)

    # Plain arrays: later merges reset the index, so labeled alignment would break
    valid = results['_movie_date'].notna().to_numpy()
    has_reviews = (results['n_reviews'] > 0).to_numpy()
    raw = results['metascore']

    critic_adj = np.where(valid & has_reviews, results['_adj_mean'], raw)
    results['critic_outlet_adjusted_score'] = np.where(valid, critic_adj, np.nan)
    results['critic_outlet_adjustment'] = np.where(valid, critic_adj - raw, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        pct_c = np.where(has_reviews, results['_n_critic'] / results['n_reviews'], 0.0)
        pct_o = np.where(has_reviews, results['_n_outlet'] / results['n_reviews'], 0.0)
    results['pct_critic_adjusted'] = np.where(valid, pct_c, np.nan)
    results['pct_outlet_adjusted'] = np.where(valid, pct_o, np.nan)

    # --- Step 2: Bayesian shrinkage toward the grand mean as of release ---
    sp = shrinkage_params.sort_values('date', kind='stable')[
        ['date', 'grand_mean', 'tau_sq', 'sigma_sq']
    ].rename(columns={'date': '_movie_date'})
    sp['_params_found'] = True
    dated = results.loc[valid, ['movie_slug', '_movie_date']].sort_values('_movie_date', kind='stable')
    dated = pd.merge_asof(dated, sp, on='_movie_date', direction='backward')
    results = results.merge(dated.drop(columns=['_movie_date']), on='movie_slug', how='left')

    n = results['n_reviews']
    sigma_sq, tau_sq, gm = results['sigma_sq'], results['tau_sq'], results['grand_mean']
    with np.errstate(invalid='ignore', divide='ignore'):
        B = (sigma_sq / n) / (sigma_sq / n + tau_sq)
        shrunk = B * gm + (1 - B) * critic_adj
    do_shrink = (
        results['_params_found'].fillna(False).astype(bool).to_numpy()
        & has_reviews & sigma_sq.notna().to_numpy() & tau_sq.notna().to_numpy()
        & (tau_sq > 0).to_numpy()
    )
    results['shrinkage_factor'] = np.where(valid, np.where(do_shrink, B, 0.0), np.nan)
    results['shrunk_score'] = np.where(valid, np.where(do_shrink, shrunk, critic_adj), np.nan)
    results['shrinkage_adjustment'] = np.where(valid, np.where(do_shrink, shrunk - critic_adj, 0.0), np.nan)
    results['grand_mean_at_time'] = np.where(valid & do_shrink, gm, np.nan)

    results['adjusted_score'] = np.where(valid, results['shrunk_score'], raw)
    results['total_adjustment'] = np.where(valid, results['adjusted_score'] - raw, np.nan)

    results_df = pd.DataFrame({
        'movie_slug': results['movie_slug'],
        'title': results['title'],
        'release_date': results['release_date'],
        'genre': results['genre'].apply(lambda g: str(g).split(',')[0].strip()),
        'raw_score': raw,
        'n_reviews': results['n_reviews'],
        'critic_outlet_adjusted_score': results['critic_outlet_adjusted_score'],
        'critic_outlet_adjustment': results['critic_outlet_adjustment'],
        'pct_critic_adjusted': results['pct_critic_adjusted'],
        'pct_outlet_adjusted': results['pct_outlet_adjusted'],
        'shrinkage_factor': results['shrinkage_factor'],
        'shrunk_score': results['shrunk_score'],
        'shrinkage_adjustment': results['shrinkage_adjustment'],
        'grand_mean_at_time': results['grand_mean_at_time'],
        'adjusted_score': results['adjusted_score'],
        'total_adjustment': results['total_adjustment'],
    })

    # Sort by adjusted score descending
    results_df = results_df.sort_values('adjusted_score', ascending=False)

    # Save
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved adjusted scores to: {output_path}")

    return results_df


def summarize_adjustments(df: pd.DataFrame):
    """Print summary statistics of adjustments."""
    
    print("\n" + "="*70)
    print("ADJUSTMENT SUMMARY")
    print("="*70)
    
    print(f"\nMovies processed: {len(df):,}")
    
    # Critic/outlet adjustment stats
    print(f"\nCritic/Outlet Adjustment:")
    print(f"  Mean:   {df['critic_outlet_adjustment'].mean():+.2f}")
    print(f"  Std:    {df['critic_outlet_adjustment'].std():.2f}")
    print(f"  Min:    {df['critic_outlet_adjustment'].min():+.2f}")
    print(f"  Max:    {df['critic_outlet_adjustment'].max():+.2f}")
    
    print(f"\nAdjustment source breakdown:")
    print(f"  Mean % from critic effect: {df['pct_critic_adjusted'].mean()*100:.1f}%")
    print(f"  Mean % from outlet effect: {df['pct_outlet_adjusted'].mean()*100:.1f}%")
    print(f"  Mean % unadjusted:         {(1 - df['pct_critic_adjusted'] - df['pct_outlet_adjusted']).mean()*100:.1f}%")
    
    # Shrinkage stats
    print(f"\nShrinkage:")
    print(f"  Mean shrinkage factor: {df['shrinkage_factor'].mean():.3f}")
    print(f"  Mean shrinkage adjustment: {df['shrinkage_adjustment'].mean():+.2f}")
    
    # Total adjustment
    print(f"\nTotal Adjustment (critic/outlet + shrinkage):")
    print(f"  Mean:   {df['total_adjustment'].mean():+.2f}")
    print(f"  Std:    {df['total_adjustment'].std():.2f}")
    print(f"  Min:    {df['total_adjustment'].min():+.2f}")
    print(f"  Max:    {df['total_adjustment'].max():+.2f}")
    
    # Score comparison
    print(f"\nScore Statistics:")
    print(f"  Raw scores:      mean={df['raw_score'].mean():.1f}, std={df['raw_score'].std():.1f}")
    print(f"  Adjusted scores: mean={df['adjusted_score'].mean():.1f}, std={df['adjusted_score'].std():.1f}")
    
    # Correlation
    corr = df['raw_score'].corr(df['adjusted_score'])
    print(f"  Correlation (raw vs adjusted): {corr:.3f}")
    
    # Biggest movers
    print(f"\nBiggest positive adjustments (underrated by raw score):")
    top_up = df.nlargest(10, 'total_adjustment')
    for _, row in top_up.iterrows():
        print(f"  {row['title'][:45]:45s} {row['raw_score']:.0f} → {row['adjusted_score']:.1f} ({row['total_adjustment']:+.1f})")
    
    print(f"\nBiggest negative adjustments (overrated by raw score):")
    top_down = df.nsmallest(10, 'total_adjustment')
    for _, row in top_down.iterrows():
        print(f"  {row['title'][:45]:45s} {row['raw_score']:.0f} → {row['adjusted_score']:.1f} ({row['total_adjustment']:+.1f})")


def main():
    parser = argparse.ArgumentParser(
        description='Compute adjusted Metacritic scores with hierarchical critic effects'
    )
    
    parser.add_argument('--input-dir', type=str, default='./metacritic_data',
                        help='Directory with movies.csv and reviews.csv')
    parser.add_argument('--effects-dir', type=str, default='./hierarchical_effects',
                        help='Directory with precomputed hierarchical effects')
    parser.add_argument('--output-dir', type=str, default='./adjusted_scores_v2',
                        help='Directory for output files')
    parser.add_argument('--halflife-shrinkage', type=float, default=2000,
                        help='Halflife for shrinkage params (default: 2000 days)')
    parser.add_argument('--recompute-shrinkage', action='store_true',
                        help='Recompute shrinkage params even if cached')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    movies_df = pd.read_csv(os.path.join(args.input_dir, 'movies.csv'))
    reviews_df = pd.read_csv(os.path.join(args.input_dir, 'reviews.csv'))
    print(f"Loaded {len(movies_df):,} movies and {len(reviews_df):,} reviews")
    
    # Load hierarchical effects
    critic_effects, outlet_effects = load_hierarchical_effects(args.effects_dir)
    
    # Compute or load shrinkage params
    shrinkage_cache = os.path.join(args.output_dir, f'shrinkage_params_hl{int(args.halflife_shrinkage)}.csv')
    
    if args.recompute_shrinkage or not os.path.exists(shrinkage_cache):
        print(f"\nComputing shrinkage params (halflife={args.halflife_shrinkage} days)...")
        shrinkage_params = compute_shrinkage_params_ewa(
            reviews_df, movies_df, halflife_days=args.halflife_shrinkage
        )
        shrinkage_params.to_csv(shrinkage_cache, index=False)
    else:
        print(f"\nLoading cached shrinkage params...")
        shrinkage_params = pd.read_csv(shrinkage_cache)
        shrinkage_params['date'] = pd.to_datetime(shrinkage_params['date'])
    
    # Process all movies
    adjusted_df = process_all_movies(
        movies_df,
        reviews_df,
        critic_effects,
        outlet_effects,
        shrinkage_params,
        output_path=os.path.join(args.output_dir, 'adjusted_scores.csv')
    )
    
    # Summary
    summarize_adjustments(adjusted_df)
    
    # Save metadata
    metadata = {
        'adjustments_applied': ['hierarchical_critic_outlet', 'bayesian_shrinkage'],
        'adjustments_not_applied': ['genre'],
        'halflife_shrinkage_days': args.halflife_shrinkage,
        'n_movies': len(movies_df),
        'n_movies_with_scores': len(adjusted_df),
        'n_reviews': len(reviews_df),
        'n_critics': len(critic_effects['critic'].unique()),
        'n_outlets': len(outlet_effects['outlet'].unique()),
        'mean_raw_score': adjusted_df['raw_score'].mean(),
        'mean_adjusted_score': adjusted_df['adjusted_score'].mean(),
        'correlation_raw_adjusted': adjusted_df['raw_score'].corr(adjusted_df['adjusted_score']),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
