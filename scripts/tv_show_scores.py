"""
Aggregate TV season scores into one score per show.

Model (one level above the season model used everywhere else):

    show quality      mu_k    ~ N(grand_mean, tau_b^2)
    season quality    theta_s ~ N(mu_k, tau_w^2)
    bias-corrected season review mean
                      ybar_s  ~ N(theta_s, sigma^2 / n_s)

The show score is the posterior mean of the *average* of its seasons'
qualities, (1/m) * sum(theta_s): every season counts equally (a show is its
run of seasons), but each season's quality is estimated by borrowing from its
sibling seasons first and the overall TV average second. A one-season show
gets exactly its season score from the main pipeline; a thinly reviewed
season of a long-running show is pulled toward that show's other seasons.

Variance components:
  sigma^2, grand_mean, tau^2  walk-forward params from the season pipeline
                              (shrinkage_params_hl2000.csv), as of each season
  tau_w^2  within-show season variance, method of moments over multi-season
           shows (spread of season means around their show mean, minus the
           review-sampling noise sigma^2/n)
  tau_b^2  between-show variance = tau^2 - tau_w^2

Seasons Metacritic hasn't scored (fewer than 4 reviews) are included when
they have any reviews; with 1-3 reviews they lean mostly on the show's other
seasons. Shows with no scored season at all are left out.

Also reported: how many seasons Metacritic lists for the show, reviewed or
not (listed_seasons from the scraper); a Metacritic-style raw show score
(review-weighted mean of season metascores, which reproduces Metacritic's own
show score); and the best season.

Usage:
    python scripts/tv_show_scores.py --scores-dir data/tv/adjusted_scores --webapp-dir public/data
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from metacritic_adjusted_scores_v2 import merge_asof_with_backfill

MIN_TAU_B_SQ = 1.0


def estimate_tau_w_sq(seasons: pd.DataFrame) -> float:
    m = seasons.groupby('show')['ybar'].transform('size')
    multi = seasons[m > 1]
    k = m[m > 1]
    dev = multi['ybar'] - multi.groupby('show')['ybar'].transform('mean')
    raw = float((dev ** 2 * k / (k - 1)).mean())
    noise = float((multi['sigma_sq'] / multi['n']).mean())
    return max(raw - noise, 0.0)


def show_scores(seasons: pd.DataFrame, tau_w_sq: float) -> pd.DataFrame:
    rows = []
    for show, g in seasons.groupby('show', sort=False):
        g = g.sort_values('season_number', kind='stable')
        y = g['ybar'].to_numpy()
        v = (g['sigma_sq'] / g['n']).to_numpy()          # sampling variance per season
        first = g.iloc[0]                                 # prior as of the show's premiere
        gm = first['grand_mean']
        tau_b_sq = max(first['tau_sq'] - tau_w_sq, MIN_TAU_B_SQ)

        # Posterior for the show mean mu_k
        w = 1.0 / (tau_w_sq + v)
        prec = 1.0 / tau_b_sq + w.sum()
        mu = (gm / tau_b_sq + (w * y).sum()) / prec
        # Each season shrinks toward mu by its own reliability; average them.
        # (When tau_w_sq == 0 every season collapses onto mu.)
        k = tau_w_sq / (tau_w_sq + v)
        theta = mu + k * (y - mu)
        score = float(theta.mean())

        # Metacritic's show score only covers seasons it scored; so does
        # "best season" (a 2-review season is too thin to headline)
        scored = g['raw_score'].notna()
        best = g.loc[g.loc[scored, 'adjusted_score'].idxmax()]
        raw = np.average(g.loc[scored, 'raw_score'], weights=g.loc[scored, 'n'])
        dates = pd.to_datetime(g['release_date'], errors='coerce')
        rows.append({
            'slug': show,
            'title': first['title'],
            'genre': first['genre'],
            'first_year': int(dates.min().year) if dates.notna().any() else None,
            'last_year': int(dates.max().year) if dates.notna().any() else None,
            'last_release_date': dates.max().strftime('%Y-%m-%d') if dates.notna().any() else '',
            'n_seasons': int(len(g)),
            'n_listed_seasons': int(max(first['listed_seasons'], len(g)))
                                if pd.notna(first.get('listed_seasons')) else int(len(g)),
            'n_unscored_seasons': int((~scored).sum()),
            'n_reviews': int(g['n'].sum()),
            'adjusted_score': round(score, 1),
            'raw_score': round(float(raw), 1),
            'total_adjustment': round(score - float(raw), 2),
            'best_season': best['subtitle'],
            'best_season_score': round(float(best['adjusted_score']), 1),
            'season_range': [round(float(g['adjusted_score'].min()), 1),
                             round(float(g['adjusted_score'].max()), 1)],
            'url': f"/tv/{show}/",
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description='Aggregate TV season scores per show')
    p.add_argument('--scores-dir', default='data/tv/adjusted_scores')
    p.add_argument('--webapp-dir', default='public/data')
    p.add_argument('--items-path', default='data/tv/metacritic_data/items.csv',
                   help='Scraped TV items, for the number of seasons Metacritic lists per show')
    args = p.parse_args()

    s = pd.read_csv(os.path.join(args.scores_dir, 'adjusted_scores.csv'))
    s = s[(s['n_reviews'] > 0) & s['critic_outlet_adjusted_score'].notna() & s['adjusted_score'].notna()].copy()
    s['show'] = s['movie_slug'].str.split('/').str[0]
    # Shows Metacritic never scored (no season with 4+ reviews) are left out
    s = s[s.groupby('show')['raw_score'].transform(lambda r: r.notna().any())]
    if os.path.exists(args.items_path):
        listed = pd.read_csv(args.items_path, usecols=lambda c: c in ('item_slug', 'listed_seasons'),
                             low_memory=False)
        if 'listed_seasons' in listed.columns:
            listed = listed.groupby(listed['item_slug'].str.split('/').str[0])['listed_seasons'].max()
            s['listed_seasons'] = s['show'].map(listed)
    if 'listed_seasons' not in s.columns:
        s['listed_seasons'] = np.nan
    s['season_number'] = pd.to_numeric(s['subtitle'].str.extract(r'(\d+)')[0], errors='coerce')
    s['ybar'] = s['critic_outlet_adjusted_score']
    s['n'] = s['n_reviews']

    sp = pd.read_csv(os.path.join(args.scores_dir, 'shrinkage_params_hl2000.csv'))
    sp['_date'] = pd.to_datetime(sp['date'])
    sp = sp[['_date', 'grand_mean', 'tau_sq', 'sigma_sq']]
    s['_date'] = pd.to_datetime(s['release_date'], errors='coerce')
    s = s.dropna(subset=['_date']).sort_values('_date', kind='stable')
    s = merge_asof_with_backfill(s, sp, on='_date')

    # Estimated from Metacritic-scored seasons (4+ reviews) only: with 1-3
    # reviews a season mean is mostly sampling noise, and subtracting that
    # large noise term makes the moment estimate unstable
    tau_w_sq = estimate_tau_w_sq(s[s['raw_score'].notna()])
    shows = show_scores(s, tau_w_sq).sort_values('adjusted_score', ascending=False)

    out = os.path.join(args.webapp_dir, 'tv_shows_all.json')
    shows = shows.astype(object).where(shows.notna(), None)
    with open(out, 'w') as f:
        json.dump({
            'updated': datetime.now().isoformat(),
            'count': len(shows),
            'model': {
                'tau_w_sq': tau_w_sq,
                'tau_sq_latest': float(sp['tau_sq'].iloc[-1]),
                'sigma_sq_latest': float(sp['sigma_sq'].iloc[-1]),
            },
            'items': shows.to_dict('records'),
        }, f, separators=(',', ':'))
    print(f"tau_w^2 (within-show season variance) = {tau_w_sq:.1f}")
    print(f"Saved {len(shows):,} shows to {out}")
    for r in shows.head(10).itertuples():
        print(f"  {r.adjusted_score:5.1f}  {r.title[:40]:40s} {r.n_seasons} seasons (raw {r.raw_score})")


if __name__ == '__main__':
    main()
