# Adjusted Metacritic Scores

A webapp that displays Metacritic movie scores adjusted for critic bias and sample size.

**Live Demo:** [your-app.vercel.app](https://your-app.vercel.app) *(update after deployment)*

## Features

- **In Theaters**: Movies from the last 60 days, ranked by adjusted score
- **All Movies**: Browse 20,000+ movies with search, genre filter, and year range
- **Bias Adjustment**: Accounts for individual critic tendencies (some rate higher/lower than average)
- **Bayesian Shrinkage**: Films with few reviews are pulled toward the overall average
- **Daily Updates**: Automatic data refresh via GitHub Actions

## Methodology

Raw Metacritic scores can be biased by *which* critics review a film. This project:

1. **Estimates critic effects**: Using exponentially-weighted averages, we calculate each critic's typical deviation from the metascore
2. **Applies hierarchical shrinkage**: Critics with few reviews are shrunk toward their outlet's average
3. **Adjusts review scores**: Each review is adjusted by subtracting the critic's estimated bias
4. **Re-averages**: Adjusted scores are computed from bias-corrected reviews
5. **Applies Bayesian shrinkage**: Films with few reviews are pulled toward the time-varying grand mean

This results in scores that better reflect the *quality signal* in reviews rather than *who happened to review* the film.

## Tech Stack

- **Frontend**: React + Vite + Tailwind CSS
- **Data Pipeline**: Python (pandas, numpy, scipy)
- **Hosting**: Vercel (free tier)
- **Automation**: GitHub Actions

---

## Deployment Guide

### Prerequisites

- GitHub account
- Vercel account (free at [vercel.com](https://vercel.com))
- Node.js 18+ installed locally
- Python 3.9+ installed locally

### Step 1: Set Up the Repository

1. Create a new GitHub repository (e.g., `metacritic-adjusted`)

2. Clone it locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/metacritic-adjusted.git
   cd metacritic-adjusted
   ```

3. Copy the webapp files into your repo (all files from this project)

4. Create the data directory structure:
   ```bash
   mkdir -p data/metacritic_data
   mkdir -p data/hierarchical_effects  
   mkdir -p data/adjusted_scores
   mkdir -p public/data
   ```

5. Copy your existing data files:
   ```bash
   cp /path/to/metacritic_data/*.csv data/metacritic_data/
   cp /path/to/hierarchical_effects/*.csv data/hierarchical_effects/
   cp /path/to/adjusted_scores_v2/*.csv data/adjusted_scores/
   cp /path/to/webapp_data/*.json public/data/
   ```

### Step 2: Test Locally

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the dev server:
   ```bash
   npm run dev
   ```

3. Open http://localhost:5173 in your browser

### Step 3: Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub

2. Click "Add New Project"

3. Import your `metacritic-adjusted` repository

4. Configure the project:
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

5. Click "Deploy"

6. Wait for the build to complete (usually 1-2 minutes)

7. Your app is now live at `https://your-project-name.vercel.app`!

### Step 4: Set Up Daily Updates

The GitHub Actions workflow (`.github/workflows/daily-update.yml`) will:
- Run at 6 AM UTC every day
- Re-compute adjusted scores from your data
- Commit the updated JSON files
- Trigger a Vercel re-deployment

**To enable:**

1. Make sure your Python scripts are in the `scripts/` directory:
   ```bash
   cp /path/to/compute_hierarchical_effects.py scripts/
   cp /path/to/metacritic_adjusted_scores_v2.py scripts/
   cp /path/to/csv_to_json.py scripts/
   cp /path/to/fix_dates.py scripts/
   ```

2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

3. The workflow will run automatically on schedule, or you can trigger it manually:
   - Go to your repo on GitHub
   - Click "Actions" tab
   - Select "Daily Data Update"
   - Click "Run workflow"

### Step 5: Custom Domain (Optional)

1. In Vercel, go to your project settings
2. Click "Domains"
3. Add your custom domain
4. Follow the DNS configuration instructions

---

## Project Structure

```
metacritic-adjusted/
├── .github/
│   └── workflows/
│       └── daily-update.yml    # GitHub Actions workflow
├── data/
│   ├── metacritic_data/        # Raw movies.csv and reviews.csv
│   ├── hierarchical_effects/   # Computed critic/outlet effects
│   └── adjusted_scores/        # Final adjusted_scores.csv
├── public/
│   └── data/                   # JSON files served by the webapp
│       ├── movies_all.json
│       ├── movies_recent.json
│       └── metadata.json
├── scripts/
│   ├── daily_update.py         # Main update orchestration
│   ├── compute_hierarchical_effects.py
│   ├── metacritic_adjusted_scores_v2.py
│   ├── csv_to_json.py
│   └── fix_dates.py
├── src/
│   ├── App.jsx                 # Main React component
│   ├── main.jsx                # Entry point
│   └── index.css               # Tailwind styles
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

---

## Updating Your Data

### Manual Update

```bash
# 1. Update your raw data (movies.csv, reviews.csv)
# 2. Run the pipeline
python scripts/daily_update.py \
  --data-dir ./data/metacritic_data \
  --effects-dir ./data/hierarchical_effects \
  --scores-dir ./data/adjusted_scores \
  --webapp-dir ./public/data \
  --scripts-dir ./scripts

# 3. Test locally
npm run dev

# 4. Commit and push
git add .
git commit -m "Update data"
git push
```

### Adding New Movies

To add newly scraped movies:

1. Append new rows to `data/metacritic_data/movies.csv`
2. Append new reviews to `data/metacritic_data/reviews.csv`
3. Run the update pipeline (effects will be recomputed)

---

## Troubleshooting

**Build fails on Vercel:**
- Check that all dependencies are in `package.json`
- Ensure `public/data/` contains the JSON files

**Data not loading:**
- Check browser console for errors
- Verify JSON files are valid: `python -m json.tool public/data/movies_all.json`

**GitHub Actions failing:**
- Check the Actions tab for error logs
- Ensure Python scripts have no syntax errors
- Verify data file paths are correct

---

## License

MIT License - feel free to use this for your own projects!

## Credits

- Data sourced from [Metacritic](https://www.metacritic.com)
- Methodology inspired by Bayesian hierarchical models and James-Stein estimation
