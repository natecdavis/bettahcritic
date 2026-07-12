"""
Hierarchical Critic Effect Estimator

Estimates critic-level effects with shrinkage toward outlet effects for sparse critics.

The model:
  review_score = movie_quality + outlet_effect + critic_deviation + noise
  
Where:
  - outlet_effect: the outlet's systematic bias
  - critic_deviation: how much this critic deviates from their outlet
  
For critics with many reviews, we estimate critic_deviation directly.
For critics with few reviews, we shrink toward zero (i.e., toward the outlet effect).
For reviews with no critic, we use the outlet effect alone.

This captures the finding that within-outlet critic variance is ~4x between-outlet variance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import argparse
from tqdm import tqdm


class HierarchicalEffectEstimator:
    """
    Estimates critic effects hierarchically:
    - First compute outlet effects
    - Then compute critic deviations from their outlet
    - Shrink sparse critics toward their outlet
    """
    
    def __init__(self, halflife_days: float = 500):
        self.halflife = halflife_days
        self.decay = np.log(2) / halflife_days
    
    def get_weights(self, dates: pd.Series, as_of_date: pd.Timestamp) -> np.ndarray:
        """Compute exponential weights for observations before as_of_date."""
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


def compute_outlet_effects(reviews_df: pd.DataFrame, movies_df: pd.DataFrame,
                           halflife_days: float = 500) -> pd.DataFrame:
    """
    Compute time-varying outlet effects using EWA.
    """
    estimator = HierarchicalEffectEstimator(halflife_days)
    
    # Merge reviews with movie info
    reviews = reviews_df.merge(
        movies_df[['movie_slug', 'release_date', 'metascore']],
        on='movie_slug',
        how='left'
    )
    
    reviews['date'] = pd.to_datetime(reviews['release_date'], errors='coerce')
    reviews = reviews.dropna(subset=['date', 'score', 'outlet', 'metascore'])
    reviews['deviation'] = reviews['score'] - reviews['metascore']
    
    # Compute at quarterly intervals
    date_range = pd.date_range(
        start=reviews['date'].min(),
        end=reviews['date'].max(),
        freq='Q'
    )
    
    # Weighted means per (outlet, quarter) via integer codes + bincount:
    # one O(n_reviews) pass per quarter instead of one per outlet per quarter
    outlet_codes, outlet_index = pd.factorize(reviews['outlet'])
    n_outlets = len(outlet_index)
    deviations = reviews['deviation'].to_numpy(dtype=float)
    dates = reviews['date']

    results = []

    print(f"Computing outlet effects at {len(date_range)} time points...")

    for as_of_date in tqdm(date_range, desc="Outlet effects"):
        weights = estimator.get_weights(dates, as_of_date)

        w_sum = np.bincount(outlet_codes, weights=weights, minlength=n_outlets)
        wd_sum = np.bincount(outlet_codes, weights=weights * deviations, minlength=n_outlets)

        with np.errstate(invalid='ignore'):
            effects = np.divide(wd_sum, w_sum,
                                out=np.full(n_outlets, np.nan), where=w_sum > 0)

        for idx in np.nonzero((w_sum >= 10) & ~np.isnan(effects))[0]:
            results.append({
                'outlet': outlet_index[idx],
                'date': as_of_date,
                'effect': effects[idx],
                'effective_n': w_sum[idx],
            })

    return pd.DataFrame(results)


def compute_critic_effects_hierarchical(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    halflife_days: float = 500,
    shrinkage_n: float = 20.0  # Effective sample size for full weight on critic effect
) -> pd.DataFrame:
    """
    Compute time-varying critic effects with hierarchical shrinkage.
    
    For each critic:
    1. Get their outlet's effect
    2. Compute their deviation from that outlet
    3. Shrink the deviation toward zero based on sample size
    
    Final critic effect = outlet_effect + shrunk_critic_deviation
    
    Args:
        shrinkage_n: Number of reviews at which critic gets 50% weight on their own effect.
                     With n reviews, critic weight = n / (n + shrinkage_n)
    """
    estimator = HierarchicalEffectEstimator(halflife_days)
    
    # Merge reviews with movie info
    reviews = reviews_df.merge(
        movies_df[['movie_slug', 'release_date', 'metascore']],
        on='movie_slug',
        how='left'
    )
    
    reviews['date'] = pd.to_datetime(reviews['release_date'], errors='coerce')
    reviews = reviews.dropna(subset=['date', 'score', 'metascore'])
    reviews['deviation'] = reviews['score'] - reviews['metascore']
    
    # Get critic-outlet mapping (use most common outlet for each critic)
    critic_outlet = reviews.dropna(subset=['critic_slug', 'outlet']).groupby('critic_slug')['outlet'].agg(
        lambda x: x.value_counts().index[0]
    ).to_dict()
    
    # Compute at quarterly intervals
    date_range = pd.date_range(
        start=reviews['date'].min(),
        end=reviews['date'].max(),
        freq='Q'
    )
    
    # Filter to reviews with critic
    critic_reviews = reviews.dropna(subset=['critic_slug'])

    # Weighted means per (critic, quarter) via integer codes + bincount
    critic_codes, critic_index = pd.factorize(critic_reviews['critic_slug'])
    n_critics = len(critic_index)
    deviations = critic_reviews['deviation'].to_numpy(dtype=float)
    dates = critic_reviews['date']

    # Outlet effect as of each quarter, as a (quarter, outlet) matrix:
    # reindex onto the union of time grids, forward-fill, select our quarters
    outlet_pivot = outlet_effects.pivot(index='date', columns='outlet', values='effect')
    union_index = outlet_pivot.index.union(date_range)
    outlet_asof = outlet_pivot.reindex(union_index).ffill().reindex(date_range)
    outlet_asof_values = outlet_asof.to_numpy()

    # Column index of each critic's outlet in the as-of matrix (-1 = no outlet)
    outlet_col = {name: j for j, name in enumerate(outlet_asof.columns)}
    critic_outlet_names = np.array(
        [critic_outlet.get(c) for c in critic_index], dtype=object
    )
    critic_outlet_cols = np.array(
        [outlet_col.get(o, -1) for o in critic_outlet_names], dtype=int
    )

    results = []

    print(f"Computing critic effects for {n_critics} critics at {len(date_range)} time points...")

    for qi, as_of_date in enumerate(tqdm(date_range, desc="Critic effects")):
        weights = estimator.get_weights(dates, as_of_date)

        w_sum = np.bincount(critic_codes, weights=weights, minlength=n_critics)
        wd_sum = np.bincount(critic_codes, weights=weights * deviations, minlength=n_critics)

        with np.errstate(invalid='ignore'):
            raw_effects = np.divide(wd_sum, w_sum,
                                    out=np.full(n_critics, np.nan), where=w_sum > 0)

        # Outlet effect per critic as of this quarter (0 when unknown, as before)
        oe = np.zeros(n_critics)
        has_outlet = critic_outlet_cols >= 0
        oe[has_outlet] = outlet_asof_values[qi, critic_outlet_cols[has_outlet]]
        oe = np.nan_to_num(oe, nan=0.0)

        # Shrink each critic's deviation from their outlet toward zero
        critic_deviation = raw_effects - oe
        critic_weight = w_sum / (w_sum + shrinkage_n)
        shrunk_deviation = critic_weight * critic_deviation
        final_effect = oe + shrunk_deviation

        for idx in np.nonzero((w_sum >= 1) & ~np.isnan(raw_effects))[0]:
            results.append({
                'critic': critic_index[idx],
                'outlet': critic_outlet_names[idx],
                'date': as_of_date,
                'raw_effect': raw_effects[idx],
                'outlet_effect': oe[idx],
                'critic_deviation': critic_deviation[idx],
                'shrunk_deviation': shrunk_deviation[idx],
                'final_effect': final_effect[idx],
                'effective_n': w_sum[idx],
                'shrinkage_weight': critic_weight[idx],
            })

    return pd.DataFrame(results)


def evaluate_hierarchical_model(
    reviews_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    critic_effects: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    sample_size: int = 10000
) -> dict:
    """
    Evaluate the hierarchical model's performance.
    
    Compare variance explained by:
    1. Outlet only
    2. Critic only (no shrinkage)
    3. Hierarchical (critic with shrinkage to outlet)
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Merge reviews with movie info
    reviews = reviews_df.merge(
        movies_df[['movie_slug', 'release_date', 'metascore']],
        on='movie_slug',
        how='left'
    )
    reviews['date'] = pd.to_datetime(reviews['release_date'], errors='coerce')
    reviews = reviews.dropna(subset=['date', 'score', 'metascore'])
    reviews['deviation'] = reviews['score'] - reviews['metascore']
    
    # Sample for speed
    if len(reviews) > sample_size:
        reviews = reviews.sample(sample_size, random_state=42)
    
    print(f"\nEvaluating on {len(reviews):,} reviews...")

    # As-of lookup of effects per review via merge_asof (last effect <= review date)
    reviews = reviews.sort_values('date', kind='stable')
    reviews['_outlet_key'] = reviews['outlet'].fillna('')
    reviews['_critic_key'] = reviews['critic_slug'].fillna('')

    oe = outlet_effects.sort_values('date', kind='stable')
    oe = oe[['outlet', 'date', 'effect']].rename(
        columns={'outlet': '_outlet_key', 'effect': '_outlet_eff'})
    reviews = pd.merge_asof(reviews, oe, on='date', by='_outlet_key', direction='backward')

    ce = critic_effects.sort_values('date', kind='stable')
    ce = ce[['critic', 'date', 'raw_effect', 'final_effect']].rename(
        columns={'critic': '_critic_key', 'raw_effect': '_raw_eff', 'final_effect': '_final_eff'})
    reviews = pd.merge_asof(reviews, ce, on='date', by='_critic_key', direction='backward')

    reviews['outlet_adj'] = reviews['_outlet_eff'].fillna(0)
    has_critic = reviews['_raw_eff'].notna()
    reviews['critic_raw_adj'] = np.where(has_critic, reviews['_raw_eff'], reviews['outlet_adj'])
    reviews['hierarchical_adj'] = np.where(has_critic, reviews['_final_eff'], reviews['outlet_adj'])
    reviews['effect_source'] = np.where(has_critic, 'critic', 'outlet')
    reviews = reviews.drop(columns=['_outlet_key', '_critic_key', '_outlet_eff', '_raw_eff', '_final_eff'])

    # Compute residual variance for each approach
    baseline_var = reviews['deviation'].var()
    
    outlet_residuals = reviews['deviation'] - reviews['outlet_adj']
    outlet_residual_var = outlet_residuals.var()
    outlet_explained = 1 - outlet_residual_var / baseline_var
    
    critic_raw_residuals = reviews['deviation'] - reviews['critic_raw_adj']
    critic_raw_residual_var = critic_raw_residuals.var()
    critic_raw_explained = 1 - critic_raw_residual_var / baseline_var
    
    hierarchical_residuals = reviews['deviation'] - reviews['hierarchical_adj']
    hierarchical_residual_var = hierarchical_residuals.var()
    hierarchical_explained = 1 - hierarchical_residual_var / baseline_var
    
    print(f"\nBaseline variance: {baseline_var:.1f}")
    print(f"\nVariance explained:")
    print(f"  Outlet only:                  {outlet_explained*100:.2f}%")
    print(f"  Critic raw (no shrinkage):    {critic_raw_explained*100:.2f}%")
    print(f"  Hierarchical (with shrinkage): {hierarchical_explained*100:.2f}%")
    
    print(f"\nImprovement over outlet:")
    print(f"  Critic raw:    +{(critic_raw_explained - outlet_explained)*100:.2f}%")
    print(f"  Hierarchical:  +{(hierarchical_explained - outlet_explained)*100:.2f}%")
    
    # Effect source breakdown
    source_counts = reviews['effect_source'].value_counts()
    print(f"\nEffect source breakdown:")
    for source, count in source_counts.items():
        print(f"  {source}: {count:,} ({count/len(reviews)*100:.1f}%)")
    
    return {
        'baseline_var': baseline_var,
        'outlet_explained': outlet_explained,
        'critic_raw_explained': critic_raw_explained,
        'hierarchical_explained': hierarchical_explained,
    }


def save_effects_for_adjustment(
    critic_effects: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    output_dir: str
):
    """Save effects in format ready for score adjustment."""
    
    # Save outlet effects
    outlet_path = os.path.join(output_dir, 'outlet_effects_hierarchical.csv')
    outlet_effects.to_csv(outlet_path, index=False)
    print(f"Saved outlet effects to: {outlet_path}")
    
    # Save critic effects
    critic_path = os.path.join(output_dir, 'critic_effects_hierarchical.csv')
    critic_effects.to_csv(critic_path, index=False)
    print(f"Saved critic effects to: {critic_path}")
    
    # Create a simple lookup for latest effects
    latest_outlet = outlet_effects.sort_values('date').groupby('outlet').last().reset_index()
    latest_outlet = latest_outlet[['outlet', 'effect', 'effective_n']]
    latest_outlet.to_csv(os.path.join(output_dir, 'outlet_effects_latest.csv'), index=False)
    
    latest_critic = critic_effects.sort_values('date').groupby('critic').last().reset_index()
    latest_critic = latest_critic[['critic', 'outlet', 'final_effect', 'outlet_effect', 
                                   'shrunk_deviation', 'effective_n', 'shrinkage_weight']]
    latest_critic.to_csv(os.path.join(output_dir, 'critic_effects_latest.csv'), index=False)
    
    print(f"\nSaved latest effects for quick lookup")


def main():
    parser = argparse.ArgumentParser(
        description='Compute hierarchical critic effects with shrinkage to outlet'
    )
    
    parser.add_argument('--input-dir', type=str, default='./metacritic_data',
                        help='Directory with movies.csv and reviews.csv')
    parser.add_argument('--output-dir', type=str, default='./hierarchical_effects',
                        help='Directory for output files')
    parser.add_argument('--halflife', type=float, default=500,
                        help='Halflife in days for EWA (default: 500)')
    parser.add_argument('--shrinkage-n', type=float, default=20,
                        help='Reviews needed for 50%% weight on critic effect (default: 20)')
    parser.add_argument('--eval-sample', type=int, default=20000,
                        help='Sample size for evaluation (default: 20000)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    movies_df = pd.read_csv(os.path.join(args.input_dir, 'movies.csv'))
    reviews_df = pd.read_csv(os.path.join(args.input_dir, 'reviews.csv'))
    
    print(f"Loaded {len(movies_df):,} movies and {len(reviews_df):,} reviews")
    
    # Step 1: Compute outlet effects
    print(f"\n--- Step 1: Outlet Effects (halflife={args.halflife} days) ---")
    outlet_effects = compute_outlet_effects(reviews_df, movies_df, args.halflife)
    print(f"Computed {len(outlet_effects):,} outlet-quarter observations")
    
    # Step 2: Compute hierarchical critic effects
    print(f"\n--- Step 2: Critic Effects (shrinkage_n={args.shrinkage_n}) ---")
    critic_effects = compute_critic_effects_hierarchical(
        reviews_df, movies_df, outlet_effects,
        halflife_days=args.halflife,
        shrinkage_n=args.shrinkage_n
    )
    print(f"Computed {len(critic_effects):,} critic-quarter observations")
    
    # Step 3: Evaluate
    print(f"\n--- Step 3: Evaluation ---")
    eval_results = evaluate_hierarchical_model(
        reviews_df, movies_df, critic_effects, outlet_effects,
        sample_size=args.eval_sample
    )
    
    # Step 4: Save
    print(f"\n--- Step 4: Saving Results ---")
    save_effects_for_adjustment(critic_effects, outlet_effects, args.output_dir)
    
    # Save metadata
    metadata = {
        'halflife_days': args.halflife,
        'shrinkage_n': args.shrinkage_n,
        'n_movies': len(movies_df),
        'n_reviews': len(reviews_df),
        'n_outlets': len(outlet_effects['outlet'].unique()),
        'n_critics': len(critic_effects['critic'].unique()),
        'outlet_variance_explained': eval_results['outlet_explained'],
        'hierarchical_variance_explained': eval_results['hierarchical_explained'],
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Hierarchical model: critic_effect = outlet_effect + shrunk(critic_deviation)

Shrinkage formula: weight = n / (n + {args.shrinkage_n})
  - Critic with 10 reviews: {10/(10+args.shrinkage_n)*100:.0f}% weight on their effect
  - Critic with 20 reviews: {20/(20+args.shrinkage_n)*100:.0f}% weight on their effect
  - Critic with 50 reviews: {50/(50+args.shrinkage_n)*100:.0f}% weight on their effect
  - Critic with 100 reviews: {100/(100+args.shrinkage_n)*100:.0f}% weight on their effect

Results:
  - Outlet-only variance explained: {eval_results['outlet_explained']*100:.2f}%
  - Hierarchical variance explained: {eval_results['hierarchical_explained']*100:.2f}%
  - Improvement: +{(eval_results['hierarchical_explained'] - eval_results['outlet_explained'])*100:.2f}%

Files saved to: {args.output_dir}/
    """)
    
    print("Done!")


if __name__ == "__main__":
    main()
