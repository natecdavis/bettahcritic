#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Starting update: $(date)"

# Step 0: Scrape new movies and reviews
# Walks the listing back --months-back months, adding new movies and
# refreshing metadata (metascores, dates) for known ones. --max-pages
# must be large enough to cover that window (24 movies per page).
python scripts/scrape_new_movies.py \
  --input-dir ./data/metacritic_data \
  --max-pages 25 \
  --months-back 3 \
  --delay 0.5

# Step 1-4: Run the rest of the pipeline
python scripts/daily_update.py \
  --data-dir ./data/metacritic_data \
  --effects-dir ./data/hierarchical_effects \
  --scores-dir ./data/adjusted_scores \
  --webapp-dir ./public/data \
  --scripts-dir ./scripts \
  --skip-fetch

# Commit and push
git add public/data/
if git diff --staged --quiet; then
  echo "No changes to commit"
else
  git commit -m "Update data $(date +'%Y-%m-%d')"
  git push
fi

echo "Done: $(date)"