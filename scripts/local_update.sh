#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Starting update: $(date)"

# ---- Movies -------------------------------------------------------------
# Walks the listing back --months-back months, adding new movies and
# refreshing metadata (metascores, dates) for known ones. --max-pages
# must be large enough to cover that window (24 movies per page).
python scripts/scrape_new_movies.py \
  --input-dir ./data/metacritic_data \
  --max-pages 25 \
  --months-back 3 \
  --delay 0.5

python scripts/daily_update.py \
  --category movies \
  --webapp-dir ./public/data \
  --scripts-dir ./scripts \
  --skip-fetch

# ---- TV, games, music ---------------------------------------------------
# Paths and model settings come from config/categories.json. A failure in
# one category is logged and skipped so it can't block the others or the
# movies update from being pushed.
for category in tv games music; do
  echo ""
  echo "==== $category ===="
  if python scripts/scrape_category.py --category "$category" --mode incremental --months-back 3 \
     && python scripts/daily_update.py --category "$category" --webapp-dir ./public/data \
          --scripts-dir ./scripts --skip-fetch; then
    echo "$category updated"
  else
    echo "WARNING: $category update failed; its previous data stays live"
  fi
done

# Commit and push
git add public/data/
if git diff --staged --quiet; then
  echo "No changes to commit"
else
  git commit -m "Update data $(date +'%Y-%m-%d')"
  git push
fi

echo "Done: $(date)"
