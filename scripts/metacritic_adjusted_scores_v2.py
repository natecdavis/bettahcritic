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


def get_critic_effect_as_of(critic_effects: pd.DataFrame, critic: str, 
                            as_of_date: pd.Timestamp) -> dict:
    """Get critic's hierarchical effect as of a given date."""
    if pd.isna(critic):
        return None
    
    mask = (critic_effects['critic'] == critic) & (critic_effects['date'] <= as_of_date)
    matching = critic_effects[mask]
    
    if len(matching) == 0:
        return None
    
    row = matching.iloc[-1]
    return {
        'effect': row['final_effect'],
        'outlet_effect': row['outlet_effect'],
        'critic_deviation': row['shrunk_deviation'],
        'shrinkage_weight': row['shrinkage_weight'],
        'effective_n': row['effective_n'],
    }


def get_outlet_effect_as_of(outlet_effects: pd.DataFrame, outlet: str,
                            as_of_date: pd.Timestamp) -> dict:
    """Get outlet effect as of a given date."""
    if pd.isna(outlet):
        return None
    
    mask = (outlet_effects['outlet'] == outlet) & (outlet_effects['date'] <= as_of_date)
    matching = outlet_effects[mask]
    
    if len(matching) == 0:
        return None
    
    row = matching.iloc[-1]
    return {
        'effect': row['effect'],
        'effective_n': row['effective_n'],
    }


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


def get_shrinkage_params_as_of(shrinkage_params: pd.DataFrame, 
                               as_of_date: pd.Timestamp) -> dict:
    """Get shrinkage parameters as of a given date."""
    mask = shrinkage_params['date'] <= as_of_date
    matching = shrinkage_params[mask]
    
    if len(matching) == 0:
        return None
    
    row = matching.iloc[-1]
    return {
        'grand_mean': row['grand_mean'],
        'tau_sq': row['tau_sq'],
        'sigma_sq': row['sigma_sq'],
    }


def compute_adjusted_score(
    movie_row: pd.Series,
    movie_reviews: pd.DataFrame,
    critic_effects: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    shrinkage_params: pd.DataFrame,
) -> dict:
    """
    Compute adjusted score for a single movie.
    
    Pipeline:
    1. Start with raw metascore
    2. Adjust each review for critic/outlet effect, then re-average
    3. Apply Bayesian shrinkage toward grand mean
    """
    movie_date = pd.to_datetime(movie_row['release_date'])
    raw_score = movie_row['metascore']
    
    result = {
        'movie_slug': movie_row['movie_slug'],
        'title': movie_row.get('title', ''),
        'release_date': movie_row['release_date'],
        'genre': str(movie_row.get('genre', '')).split(',')[0].strip(),
        'raw_score': raw_score,
        'n_reviews': len(movie_reviews),
    }
    
    if pd.isna(movie_date) or pd.isna(raw_score):
        result['adjusted_score'] = raw_score
        return result
    
    # Step 1: Critic/outlet adjusted score
    if len(movie_reviews) > 0:
        adjusted_review_scores = []
        adjustment_sources = {'critic': 0, 'outlet': 0, 'none': 0}
        
        for _, review in movie_reviews.iterrows():
            review_score = review['score']
            critic = review.get('critic_slug')
            outlet = review.get('outlet')
            
            # Use review date if available, else movie release date
            review_date = pd.to_datetime(review.get('date'), errors='coerce')
            if pd.isna(review_date):
                review_date = movie_date
            
            # Try critic effect first
            effect = 0
            source = 'none'
            
            critic_params = get_critic_effect_as_of(critic_effects, critic, review_date)
            if critic_params:
                effect = critic_params['effect']
                source = 'critic'
            else:
                # Fall back to outlet
                outlet_params = get_outlet_effect_as_of(outlet_effects, outlet, review_date)
                if outlet_params:
                    effect = outlet_params['effect']
                    source = 'outlet'
            
            adjusted_review_scores.append(review_score - effect)
            adjustment_sources[source] += 1
        
        critic_adjusted_score = np.mean(adjusted_review_scores)
        result['critic_outlet_adjusted_score'] = critic_adjusted_score
        result['critic_outlet_adjustment'] = critic_adjusted_score - raw_score
        result['pct_critic_adjusted'] = adjustment_sources['critic'] / len(movie_reviews)
        result['pct_outlet_adjusted'] = adjustment_sources['outlet'] / len(movie_reviews)
    else:
        critic_adjusted_score = raw_score
        result['critic_outlet_adjusted_score'] = raw_score
        result['critic_outlet_adjustment'] = 0
        result['pct_critic_adjusted'] = 0
        result['pct_outlet_adjusted'] = 0
    
    # Step 2: Bayesian shrinkage
    shrink_params = get_shrinkage_params_as_of(shrinkage_params, movie_date)
    
    if shrink_params and result['n_reviews'] > 0:
        sigma_sq = shrink_params.get('sigma_sq', 200)
        tau_sq = shrink_params.get('tau_sq', 300)
        grand_mean = shrink_params.get('grand_mean', 60)
        n = result['n_reviews']
        
        if not np.isnan(sigma_sq) and not np.isnan(tau_sq) and tau_sq > 0:
            # Shrinkage factor
            B = (sigma_sq / n) / (sigma_sq / n + tau_sq)
            
            # Apply shrinkage
            shrunk_score = B * grand_mean + (1 - B) * critic_adjusted_score
            
            result['shrinkage_factor'] = B
            result['shrunk_score'] = shrunk_score
            result['shrinkage_adjustment'] = shrunk_score - critic_adjusted_score
            result['grand_mean_at_time'] = grand_mean
        else:
            result['shrunk_score'] = critic_adjusted_score
            result['shrinkage_factor'] = 0
            result['shrinkage_adjustment'] = 0
    else:
        result['shrunk_score'] = critic_adjusted_score
        result['shrinkage_factor'] = 0
        result['shrinkage_adjustment'] = 0
    
    result['adjusted_score'] = result['shrunk_score']
    result['total_adjustment'] = result['adjusted_score'] - raw_score
    
    return result


def process_all_movies(
    movies_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    critic_effects: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    shrinkage_params: pd.DataFrame,
    output_path: str
) -> pd.DataFrame:
    """Process all movies and compute adjusted scores."""
    
    # Filter to movies with metascores
    movies_with_scores = movies_df.dropna(subset=['metascore'])
    
    print(f"\nProcessing {len(movies_with_scores):,} movies with metascores...")
    
    results = []
    
    for _, movie in tqdm(movies_with_scores.iterrows(), total=len(movies_with_scores),
                         desc="Computing adjusted scores"):
        movie_reviews = reviews_df[reviews_df['movie_slug'] == movie['movie_slug']]
        
        result = compute_adjusted_score(
            movie,
            movie_reviews,
            critic_effects,
            outlet_effects,
            shrinkage_params,
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
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
