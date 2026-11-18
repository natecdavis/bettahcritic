"""
Cross-category diagnostics: which modelling findings from movies carry over
to TV, games and music, and is there a file-drawer effect?

Run after each category's effects + adjusted scores exist (daily_update.py).
For each category, writes <data_root>/diagnostics.json (embedded in the
webapp metadata) and prints a comparison table. Settings that change scores
(config/categories.json coverage_prior) are NOT flipped automatically; the
recommendation is printed so the decision stays reviewable.

File-drawer tests
-----------------
A file-drawer effect here means unfavourable reviews go unpublished (outlets
skip, or spike, coverage of things they dislike), so sparsely reviewed items
look better than they are. Three tests:

  A. Coverage gradient. Within release year, regress each item's bias-corrected
     review mean on log(n_reviews), with heteroskedasticity-robust (HC1) SEs.
     A positive slope means low-coverage items score lower than the
     unconditional prior assumes, which is the quantity the shrinkage prior
     needs to know.

  B. Selective coverage (mechanism). For each outlet with enough reviews, over
     the window it was active: logistic regression of "did this outlet review
     the item" on the item's consensus excluding that outlet, controlling for
     log(number of other reviews) and release year. A positive coefficient
     means that, holding prominence fixed, the outlet skips worse items.

  C. Out-of-sample value of a coverage-aware prior. For items with >= 8
     reviews, estimate each from 3 random reviews, shrinking toward either the
     unconditional prior or a log(n)-conditional prior, and score both against
     the mean of the item's other reviews. Repeated over several draws.

Verdicts:
  file_drawer      B shows outlets selecting on quality: pooled odds ratio per
                   +10 points >= 1.10 AND the share of outlets significantly
                   positive exceeds the share significantly negative by >= 25
                   points. (A alone is not enough: in movies, coverage tracks
                   prestige, and outlets show no net selection.)
  coverage_prior   file_drawer AND A significant (|t| > 4) with a p10->p90 gap
                   >= 2 points AND C cuts held-out error by >= 1%.

Usage:
    python scripts/analyze_categories.py                 # all categories
    python scripts/analyze_categories.py --categories tv games
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

from pipeline_io import load_items_and_reviews

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'categories.json')

MIN_OUTLET_REVIEWS = 200
MAX_OUTLETS = 80
SUBSAMPLE_K = 3
SUBSAMPLE_MIN_N = 8
SUBSAMPLE_DRAWS = 5


def ols_slope_hc1(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """No-intercept OLS slope (inputs already demeaned) and its HC1 robust SE."""
    sxx = np.sum(x * x)
    b = np.sum(x * y) / sxx
    e = y - b * x
    n = len(x)
    var = np.sum(x * x * e * e) / sxx ** 2 * n / (n - 1)
    return float(b), float(np.sqrt(var))


def logit_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 50):
    """Logistic regression by Newton/IRLS. Returns (coefs, standard errors)."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (p * (1 - p))[:, None])
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def load_category(cfg: dict):
    root = cfg['data_root']
    items, reviews = load_items_and_reviews(os.path.join(root, 'metacritic_data'), cfg['items_file'],
                                            cfg.get('exclude_release_types'))
    scores = pd.read_csv(os.path.join(root, 'adjusted_scores', 'adjusted_scores.csv'))
    scores = scores[scores['raw_score'].notna()]   # unscored TV seasons are for show aggregates only
    with open(os.path.join(root, 'hierarchical_effects', 'metadata.json')) as f:
        effects_meta = json.load(f)
    outlet_latest = pd.read_csv(os.path.join(root, 'hierarchical_effects', 'outlet_effects_latest.csv'))
    critic_latest = pd.read_csv(os.path.join(root, 'hierarchical_effects', 'critic_effects_latest.csv'))
    items['release'] = pd.to_datetime(items['release_date'], errors='coerce')
    items = items.dropna(subset=['metascore', 'release'])
    reviews = reviews[reviews['movie_slug'].isin(set(items['movie_slug']))]
    reviews = reviews.dropna(subset=['score'])
    return items, reviews, scores, effects_meta, outlet_latest, critic_latest


def review_adjusted_scores(reviews, outlet_latest, critic_latest):
    """Per-review bias-corrected scores using the latest effects (an
    approximation of the pipeline's as-of lookup, fine for diagnostics)."""
    ce = critic_latest.set_index('critic')['final_effect']
    oe = outlet_latest.set_index('outlet')['effect']
    eff = reviews['critic_slug'].map(ce)
    eff = eff.fillna(reviews['outlet'].map(oe)).fillna(0.0)
    return reviews['score'] - eff


# ------------------------------------------------------------------ test A

def coverage_gradient(scores: pd.DataFrame) -> dict:
    df = scores.dropna(subset=['critic_outlet_adjusted_score', 'release_date']).copy()
    df = df[df['n_reviews'] >= 1]
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df.dropna(subset=['year'])
    df['log_n'] = np.log(df['n_reviews'])
    y = df['critic_outlet_adjusted_score'] - df.groupby('year')['critic_outlet_adjusted_score'].transform('mean')
    x = df['log_n'] - df.groupby('year')['log_n'].transform('mean')
    slope, se = ols_slope_hc1(x.to_numpy(), y.to_numpy())
    p10, p50, p90 = np.percentile(df['n_reviews'], [10, 50, 90])
    gap = slope * (np.log(p90) - np.log(p10))

    # Same gradient on the raw metascore, for comparison
    yr = df['raw_score'] - df.groupby('year')['raw_score'].transform('mean')
    raw_slope, _ = ols_slope_hc1(x.to_numpy(), yr.to_numpy())

    # Binned profile (year-demeaned) for charts
    edges = [0, 4, 7, 11, 19, 34, 10_000]
    labels = ['1-4', '5-7', '8-11', '12-19', '20-34', '35+']
    df['bin'] = pd.cut(df['n_reviews'], edges, labels=labels)
    df['y_dm'] = y
    prof = df.groupby('bin', observed=False)['y_dm'].agg(['mean', 'count', 'sem'])
    return {
        'n_items': int(len(df)),
        'slope_per_log_n': slope,
        'slope_per_doubling': slope * np.log(2),
        'se_per_doubling': se * np.log(2),
        't_stat': slope / se if se > 0 else float('nan'),
        'n_reviews_p10_p50_p90': [float(p10), float(p50), float(p90)],
        'p10_to_p90_gap_points': gap,
        'raw_slope_per_doubling': raw_slope * np.log(2),
        'profile': [
            {'bin': str(b), 'mean_vs_year': None if pd.isna(r['mean']) else float(r['mean']),
             'sem': None if pd.isna(r['sem']) else float(r['sem']), 'count': int(r['count'])}
            for b, r in prof.iterrows()
        ],
    }


# ------------------------------------------------------------------ test B

def selective_coverage(items: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    rv = reviews.dropna(subset=['outlet'])
    # One review per (item, outlet); an outlet's duplicate reviews are averaged
    rv = rv.groupby(['movie_slug', 'outlet'], as_index=False)['score'].mean()
    totals = rv.groupby('movie_slug')['score'].agg(['sum', 'count'])
    it = items.set_index('movie_slug').loc[lambda d: d.index.isin(totals.index), ['release']]
    it = it.join(totals)
    it['year'] = it['release'].dt.year + it['release'].dt.dayofyear / 366

    counts = rv['outlet'].value_counts()
    outlets = counts[counts >= MIN_OUTLET_REVIEWS].index[:MAX_OUTLETS]
    by_outlet = {o: g.set_index('movie_slug')['score'] for o, g in rv[rv['outlet'].isin(outlets)].groupby('outlet')}

    rows = []
    for o in outlets:
        own = by_outlet[o]
        dates = it.loc[own.index.intersection(it.index), 'release']
        lo, hi = dates.quantile(0.05), dates.quantile(0.95)
        cand = it[(it['release'] >= lo) & (it['release'] <= hi)].copy()
        own_score = own.reindex(cand.index)
        reviewed = own_score.notna()
        cand['n_other'] = cand['count'] - reviewed.astype(int)
        cand['q_other'] = (cand['sum'] - own_score.fillna(0)) / cand['n_other']
        cand['reviewed'] = reviewed.astype(int)
        cand = cand[cand['n_other'] >= 3]
        if cand['reviewed'].sum() < 50 or (1 - cand['reviewed']).sum() < 50:
            continue
        ln = np.log(cand['n_other'])
        X = np.column_stack([
            np.ones(len(cand)),
            (cand['q_other'] - cand['q_other'].mean()) / 10.0,   # per 10 points
            ln - ln.mean(),
            cand['year'] - cand['year'].mean(),
        ])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coefs, ses = logit_irls(X, cand['reviewed'].to_numpy(dtype=float))
            b, se = float(coefs[1]), float(ses[1])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(b) or not np.isfinite(se):
            continue
        rows.append({'outlet': o, 'n_reviews': int(reviewed.sum()), 'n_candidates': int(len(cand)),
                     'coef_per_10pts': b, 'se': se, 'odds_ratio_per_10pts': float(np.exp(b))})

    if not rows:
        return {'n_outlets': 0}
    res = pd.DataFrame(rows)
    w = 1 / res['se'] ** 2
    pooled = float(np.sum(w * res['coef_per_10pts']) / np.sum(w))
    pooled_se = float(np.sqrt(1 / np.sum(w)))
    sig_pos = ((res['coef_per_10pts'] / res['se']) > 2.58).mean()
    sig_neg = ((res['coef_per_10pts'] / res['se']) < -2.58).mean()
    return {
        'n_outlets': int(len(res)),
        'median_odds_ratio_per_10pts': float(np.exp(res['coef_per_10pts'].median())),
        'pooled_odds_ratio_per_10pts': float(np.exp(pooled)),
        'pooled_coef_se': pooled_se,
        'share_outlets_sig_positive': float(sig_pos),
        'share_outlets_sig_negative': float(sig_neg),
        'outlets': res.sort_values('n_reviews', ascending=False).head(15).to_dict('records'),
    }


# ------------------------------------------------------------------ test C

def subsample_validation(items, reviews, outlet_latest, critic_latest, seed=0) -> dict:
    rv = reviews[['movie_slug', 'score', 'outlet', 'critic_slug']].copy()
    rv['adj'] = review_adjusted_scores(rv, outlet_latest, critic_latest)
    n = rv.groupby('movie_slug').size()
    item_mean = rv.groupby('movie_slug')['adj'].mean()
    within_var = rv.groupby('movie_slug')['adj'].var()
    sigma_sq = float(within_var.mean())

    # Prior fits on all items (two parameters on thousands of items: in-sample
    # leakage is negligible); same functional forms the pipeline uses
    log_n = np.log(n)
    gm = float(item_mean.mean())
    tau_sq_uncond = float(item_mean.var())
    x = log_n - log_n.mean()
    slope = float(np.sum(x * (item_mean - gm)) / np.sum(x ** 2))
    resid = item_mean - gm - slope * x
    tau_sq_cov = float(resid.var())

    eligible = n[n >= SUBSAMPLE_MIN_N].index
    rv_e = rv[rv['movie_slug'].isin(eligible)]
    rng = np.random.default_rng(seed)
    k = SUBSAMPLE_K
    errs_u, errs_c, errs_raw, sparse_flags = [], [], [], []
    for _ in range(SUBSAMPLE_DRAWS):
        r = rv_e.assign(_u=rng.random(len(rv_e))).sort_values(['movie_slug', '_u'])
        r['_rank'] = r.groupby('movie_slug').cumcount()
        obs = r[r['_rank'] < k].groupby('movie_slug')['adj'].mean()
        held = r[r['_rank'] >= k].groupby('movie_slug')['adj'].mean()
        xi = (np.log(n.loc[obs.index]) - log_n.mean())
        B_u = (sigma_sq / k) / (sigma_sq / k + tau_sq_uncond)
        B_c = (sigma_sq / k) / (sigma_sq / k + tau_sq_cov)
        est_u = B_u * gm + (1 - B_u) * obs
        est_c = B_c * (gm + slope * xi) + (1 - B_c) * obs
        errs_u.append((est_u - held) ** 2)
        errs_c.append((est_c - held) ** 2)
        errs_raw.append((obs - held) ** 2)
        sparse_flags.append(n.loc[obs.index] <= 15)
    eu, ec, er = pd.concat(errs_u), pd.concat(errs_c), pd.concat(errs_raw)
    sparse = pd.concat(sparse_flags)
    out = {
        'n_items': int(len(eligible)),
        'k_observed': k,
        'mse_no_shrinkage': float(er.mean()),
        'mse_unconditional_prior': float(eu.mean()),
        'mse_coverage_prior': float(ec.mean()),
        'improvement_pct': float(100 * (1 - ec.mean() / eu.mean())),
        'shrinkage_improvement_pct': float(100 * (1 - eu.mean() / er.mean())),
    }
    if sparse.any():
        out['improvement_pct_items_8_to_15_reviews'] = float(
            100 * (1 - ec[sparse.values].mean() / eu[sparse.values].mean()))
    # Paired bootstrap-free SE of the MSE difference
    d = (ec - eu).to_numpy()
    out['mse_diff_t'] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d) / SUBSAMPLE_DRAWS)))
    return out


# ------------------------------------------------------- other findings

def model_summary(items, reviews, scores, effects_meta, outlet_latest, critic_latest, cfg) -> dict:
    n = reviews.groupby('movie_slug').size()
    rel = items.set_index('movie_slug')['release']
    rdate = pd.to_datetime(reviews['date'], errors='coerce')
    lag_days = (rdate - reviews['movie_slug'].map(rel)).dt.days
    sp_path = os.path.join(cfg['data_root'], 'adjusted_scores', 'shrinkage_params_hl2000.csv')
    sp = pd.read_csv(sp_path) if os.path.exists(sp_path) else None

    oe = outlet_latest.dropna(subset=['effect'])
    big = oe[oe['effective_n'] >= 20]
    ce = critic_latest.dropna(subset=['final_effect'])
    ce = ce[ce['effective_n'] >= 20]

    yr = pd.to_datetime(scores['release_date'], errors='coerce').dt.year
    by_decade = scores.assign(decade=(yr // 10) * 10).groupby('decade')['raw_score'].agg(['mean', 'count'])
    by_decade = by_decade[by_decade['count'] >= 30]

    out = {
        'n_items': int(len(items)),
        'n_reviews': int(len(reviews)),
        'median_reviews_per_item': float(n.median()),
        'share_reviews_named_critic': float(reviews['critic_slug'].notna().mean()),
        'n_outlets': int(reviews['outlet'].nunique()),
        'outlet_variance_explained': effects_meta.get('outlet_variance_explained'),
        'hierarchical_variance_explained': effects_meta.get('hierarchical_variance_explained'),
        'sd_outlet_effect': float(big['effect'].std()) if len(big) else None,
        'sd_critic_effect': float(ce['final_effect'].std()) if len(ce) else None,
        'mean_abs_bias_adjustment': float(scores['critic_outlet_adjustment'].abs().mean()),
        'mean_abs_shrinkage_adjustment': float(scores['shrinkage_adjustment'].abs().mean()),
        'mean_abs_total_adjustment': float(scores['total_adjustment'].abs().mean()),
        'median_divisiveness': float(scores['polarization_std'].median()),
        'share_reviews_dated_1y_after_release': float((lag_days > 365).mean()),
        'mean_metascore_by_decade': {str(int(k)): round(float(v), 1) for k, v in by_decade['mean'].items()},
    }
    if sp is not None and len(sp):
        last = sp.iloc[-1]
        med_n = float(n.median())
        B = (last['sigma_sq'] / med_n) / (last['sigma_sq'] / med_n + last['tau_sq'])
        out.update({
            'grand_mean_latest': float(last['grand_mean']),
            'tau_sq_latest': float(last['tau_sq']),
            'sigma_sq_latest': float(last['sigma_sq']),
            'shrinkage_weight_at_median_n': float(B),
        })
    return out


def analyze(category: str, cfg: dict) -> dict:
    items, reviews, scores, effects_meta, outlet_latest, critic_latest = load_category(cfg)
    print(f"\n=== {category}: {len(items):,} items, {len(reviews):,} reviews")
    result = {
        'category': category,
        'generated': pd.Timestamp.now().isoformat(timespec='seconds'),
        'summary': model_summary(items, reviews, scores, effects_meta, outlet_latest, critic_latest, cfg),
        'coverage_gradient': coverage_gradient(scores),
        'selective_coverage': selective_coverage(items, reviews),
        'subsample_validation': subsample_validation(items, reviews, outlet_latest, critic_latest),
    }
    a, b, c = result['coverage_gradient'], result['selective_coverage'], result['subsample_validation']
    file_drawer = bool(
        b.get('n_outlets', 0) >= 5
        and b['pooled_odds_ratio_per_10pts'] >= 1.10
        and b['share_outlets_sig_positive'] - b['share_outlets_sig_negative'] >= 0.25)
    gradient = bool(a['t_stat'] > 4 and a['p10_to_p90_gap_points'] >= 2)
    result['verdicts'] = {
        'file_drawer': file_drawer,
        'coverage_gradient': gradient,
        'coverage_prior_helps_out_of_sample': bool(c['improvement_pct'] >= 1.0),
    }
    result['recommend_coverage_prior'] = bool(file_drawer and gradient and c['improvement_pct'] >= 1.0)
    result['currently_enabled'] = bool(cfg.get('coverage_prior'))

    print(f"  A coverage gradient: {a['slope_per_doubling']:+.2f} pts per doubling of reviews "
          f"(t={a['t_stat']:.1f}); p10->p90 n {a['n_reviews_p10_p50_p90'][0]:.0f}->"
          f"{a['n_reviews_p10_p50_p90'][2]:.0f} gap {a['p10_to_p90_gap_points']:+.1f} pts")
    if b.get('n_outlets'):
        print(f"  B selective coverage: {b['n_outlets']} outlets; pooled OR per +10 pts "
              f"{b['pooled_odds_ratio_per_10pts']:.2f}, median {b['median_odds_ratio_per_10pts']:.2f}; "
              f"{b['share_outlets_sig_positive']:.0%} sig. positive, "
              f"{b['share_outlets_sig_negative']:.0%} sig. negative")
    print(f"  C subsample (k={c['k_observed']}): coverage prior changes MSE by "
          f"{-c['improvement_pct']:+.1f}% vs unconditional (t={c['mse_diff_t']:.1f}); "
          f"shrinkage itself {-c['shrinkage_improvement_pct']:+.1f}% vs none")
    print(f"  -> file drawer: {file_drawer}; recommend coverage_prior={result['recommend_coverage_prior']} "
          f"(config has {result['currently_enabled']})")
    return result


def main():
    p = argparse.ArgumentParser(description='Cross-category diagnostics / file-drawer tests')
    p.add_argument('--categories', nargs='*', default=None)
    p.add_argument('--combined-output', default=None,
                   help='Also write all results to this JSON file')
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        config = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    cats = args.categories or list(config)

    results = {}
    for cat in cats:
        cfg = config[cat]
        if not os.path.exists(os.path.join(cfg['data_root'], 'adjusted_scores', 'adjusted_scores.csv')):
            print(f"\n=== {cat}: no adjusted scores yet, skipping")
            continue
        res = analyze(cat, cfg)
        results[cat] = res
        with open(os.path.join(cfg['data_root'], 'diagnostics.json'), 'w') as f:
            json.dump(res, f, indent=2, default=float)

    if args.combined_output:
        with open(args.combined_output, 'w') as f:
            json.dump(results, f, indent=2, default=float)


if __name__ == '__main__':
    main()
