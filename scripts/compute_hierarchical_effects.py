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
    
    outlets = reviews['outlet'].unique()
    results = []
    
    print(f"Computing outlet effects at {len(date_range)} time points...")
    
    for as_of_date in tqdm(date_range, desc="Outlet effects"):
        weights = estimator.get_weights(reviews['date'], as_of_date)
        
        for outlet in outlets:
            mask = reviews['outlet'] == outlet
            outlet_weights = weights[mask]
            outlet_deviations = reviews.loc[mask, 'deviation'].values
            
            effective_n = outlet_weights.sum()
            if effective_n < 10:
                continue
            
            effect = estimator.weighted_mean(outlet_deviations, outlet_weights)
            
            if not np.isnan(effect):
                results.append({
                    'outlet': outlet,
                    'date': as_of_date,
                    'effect': effect,
                    'effective_n': effective_n,
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
    critics = critic_reviews['critic_slug'].unique()
    
    results = []
    
    print(f"Computing critic effects for {len(critics)} critics at {len(date_range)} time points...")
    
    for as_of_date in tqdm(date_range, desc="Critic effects"):
        weights = estimator.get_weights(critic_reviews['date'], as_of_date)
        
        # Get outlet effects as of this date
        outlet_effects_now = {}
        for outlet in outlet_effects['outlet'].unique():
            outlet_data = outlet_effects[
                (outlet_effects['outlet'] == outlet) & 
                (outlet_effects['date'] <= as_of_date)
            ]
            if len(outlet_data) > 0:
                outlet_effects_now[outlet] = outlet_data.iloc[-1]['effect']
        
        for critic in critics:
            mask = critic_reviews['critic_slug'] == critic
            critic_weights = weights[mask]
            critic_deviations = critic_reviews.loc[mask, 'deviation'].values
            
            effective_n = critic_weights.sum()
            if effective_n < 1:  # Need at least some data
                continue
            
            # Raw critic effect (deviation from metascore)
            raw_critic_effect = estimator.weighted_mean(critic_deviations, critic_weights)
            
            if np.isnan(raw_critic_effect):
                continue
            
            # Get outlet effect for this critic
            critic_outlet_name = critic_outlet.get(critic)
            outlet_effect = outlet_effects_now.get(critic_outlet_name, 0)
            
            # Critic's deviation from their outlet
            critic_deviation = raw_critic_effect - outlet_effect
            
            # Shrinkage: weight on critic's own deviation vs zero
            # More reviews = more weight on critic's deviation
            critic_weight = effective_n / (effective_n + shrinkage_n)
            shrunk_deviation = critic_weight * critic_deviation
            
            # Final effect = outlet + shrunk deviation
            final_effect = outlet_effect + shrunk_deviation
            
            results.append({
                'critic': critic,
                'outlet': critic_outlet_name,
                'date': as_of_date,
                'raw_effect': raw_critic_effect,
                'outlet_effect': outlet_effect,
                'critic_deviation': critic_deviation,
                'shrunk_deviation': shrunk_deviation,
                'final_effect': final_effect,
                'effective_n': effective_n,
                'shrinkage_weight': critic_weight,
            })
    
    return pd.DataFrame(results)


def get_effect_for_review(
    review_row: pd.Series,
    critic_effects: pd.DataFrame,
    outlet_effects: pd.DataFrame,
    as_of_date: pd.Timestamp
) -> tuple[float, str]:
    """
    Get the appropriate effect for a single review.
    
    Priority:
    1. If critic is known and has effect estimate, use critic effect
    2. Else if outlet has effect estimate, use outlet effect
    3. Else return 0
    
    Returns (effect, source) where source is 'critic', 'outlet', or 'none'
    """
    critic = review_row.get('critic_slug')
    outlet = review_row.get('outlet')
    
    # Try critic effect first
    if pd.notna(critic):
        critic_data = critic_effects[
            (critic_effects['critic'] == critic) &
            (critic_effects['date'] <= as_of_date)
        ]
        if len(critic_data) > 0:
            return critic_data.iloc[-1]['final_effect'], 'critic'
    
    # Fall back to outlet effect
    if pd.notna(outlet):
        outlet_data = outlet_effects[
            (outlet_effects['outlet'] == outlet) &
            (outlet_effects['date'] <= as_of_date)
        ]
        if len(outlet_data) > 0:
            return outlet_data.iloc[-1]['effect'], 'outlet'
    
    return 0.0, 'none'


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
    
    # Get effects for each review
    outlet_adj = []
    critic_raw_adj = []
    hierarchical_adj = []
    sources = []
    
    # DEBUG: Check a few rows
    debug_count = 0
    debug_diffs = []
    
    for _, row in tqdm(reviews.iterrows(), total=len(reviews), desc="Evaluating"):
        as_of_date = row['date']
        
        # Outlet effect
        outlet = row.get('outlet')
        outlet_data = outlet_effects[
            (outlet_effects['outlet'] == outlet) &
            (outlet_effects['date'] <= as_of_date)
        ]
        outlet_effect = outlet_data.iloc[-1]['effect'] if len(outlet_data) > 0 else 0
        outlet_adj.append(outlet_effect)
        
        # Raw critic effect (no shrinkage)
        critic = row.get('critic_slug')
        if pd.notna(critic):
            critic_data = critic_effects[
                (critic_effects['critic'] == critic) &
                (critic_effects['date'] <= as_of_date)
            ]
            if len(critic_data) > 0:
                raw_eff = critic_data.iloc[-1]['raw_effect']
                hier_eff = critic_data.iloc[-1]['final_effect']
                critic_raw_adj.append(raw_eff)
                hierarchical_adj.append(hier_eff)
                sources.append('critic')
                
                # DEBUG
                if debug_count < 5 and abs(raw_eff - outlet_effect) > 0.01:
                    print(f"\n  DEBUG: critic={critic}, outlet_eff={outlet_effect:.2f}, raw_eff={raw_eff:.2f}, hier_eff={hier_eff:.2f}")
                    debug_count += 1
                debug_diffs.append(raw_eff - outlet_effect)
            else:
                critic_raw_adj.append(outlet_effect)
                hierarchical_adj.append(outlet_effect)
                sources.append('outlet')
        else:
            critic_raw_adj.append(outlet_effect)
            hierarchical_adj.append(outlet_effect)
            sources.append('outlet')
    
    reviews['outlet_adj'] = outlet_adj
    reviews['critic_raw_adj'] = critic_raw_adj
    reviews['hierarchical_adj'] = hierarchical_adj
    reviews['effect_source'] = sources
    
    # DEBUG: Print summary of differences
    debug_diffs_arr = np.array([d for d in debug_diffs if not np.isnan(d)])
    print(f"\n  DEBUG: critic_raw - outlet differences:")
    print(f"    Count: {len(debug_diffs_arr)}")
    print(f"    Mean diff: {debug_diffs_arr.mean():.4f}")
    print(f"    Std diff: {debug_diffs_arr.std():.4f}")
    print(f"    Min/Max: {debug_diffs_arr.min():.4f} / {debug_diffs_arr.max():.4f}")
    print(f"    Exactly zero: {(debug_diffs_arr == 0).sum()}")
    
    # Also check the actual adjustment arrays
    outlet_arr = np.array(outlet_adj)
    critic_arr = np.array(critic_raw_adj)
    adj_diff = critic_arr - outlet_arr
    print(f"\n  DEBUG: actual adjustment array differences:")
    print(f"    Mean: {adj_diff.mean():.4f}")
    print(f"    Std: {adj_diff.std():.4f}")
    print(f"    Exactly zero: {(adj_diff == 0).sum()} / {len(adj_diff)}")
    
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
