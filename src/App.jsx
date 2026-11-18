import React, { useState, useMemo, useEffect } from 'react';
import {
  Search, TrendingUp, TrendingDown, Calendar, Film, Tv, Gamepad2, Music, Tag,
  ChevronDown, ChevronUp, Star, BarChart3, ChevronLeft, ChevronRight, Loader, ExternalLink,
} from 'lucide-react';

// Per-category presentation. `recentDays` is how far back the "recent" tab
// looks; `forwardDays` admits items reviewed ahead of release.
const CATEGORIES = {
  movies: {
    label: 'Movies', icon: Film, noun: 'movies', recentTab: 'In Theaters', allTab: 'All Movies',
    recentDays: 60, forwardDays: 14, metacriticPath: (m) => `/movie/${m.slug}/`,
  },
  tv: {
    label: 'TV', icon: Tv, noun: 'seasons', recentTab: 'New Seasons', allTab: 'All Seasons', showsTab: 'Shows',
    recentDays: 90, forwardDays: 14,
  },
  games: {
    label: 'Games', icon: Gamepad2, noun: 'games', recentTab: 'New Releases', allTab: 'All Games',
    recentDays: 90, forwardDays: 14,
  },
  music: {
    label: 'Music', icon: Music, noun: 'albums', recentTab: 'New Releases', allTab: 'All Albums',
    recentDays: 60, forwardDays: 7,
  },
};
const CATEGORY_KEYS = Object.keys(CATEGORIES);

// Metacritic color scheme: Green 61+, Yellow 40-60, Red 39-
const getScoreColor = (score) => {
  if (score >= 60.5) return 'bg-green-600';
  if (score >= 39.5) return 'bg-yellow-500';
  return 'bg-red-600';
};

const getScoreTextColor = (score) => {
  if (score >= 60.5) return 'text-green-600';
  if (score >= 39.5) return 'text-yellow-600';
  return 'text-red-600';
};

const metacriticUrl = (item, category) => {
  const path = item.url || CATEGORIES[category].metacriticPath?.(item);
  return path ? `https://www.metacritic.com${path}` : null;
};

const ScoreBadge = ({ score, size = 'md' }) => {
  const sizeClasses = { sm: 'w-10 h-10 text-sm', md: 'w-12 h-12 text-base', lg: 'w-14 h-14 text-lg' };
  return (
    <div className={`${sizeClasses[size]} ${getScoreColor(score)} rounded-lg flex items-center justify-center text-white font-bold shadow`}>
      {Math.round(score)}
    </div>
  );
};

const AdjustmentBadge = ({ adjustment }) => {
  const isPositive = adjustment > 0;
  return (
    <div className={`flex items-center gap-1 text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
      {isPositive ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
      <span>{isPositive ? '+' : ''}{adjustment.toFixed(1)}</span>
    </div>
  );
};

const Title = ({ item, category, className }) => {
  const href = metacriticUrl(item, category);
  if (!href) return <span className={className}>{item.title}</span>;
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={`${className} hover:text-blue-700 hover:underline`}>
      {item.title}
    </a>
  );
};

// Pagination component
const Pagination = ({ currentPage, totalPages, onPageChange, totalItems, itemsPerPage, noun }) => {
  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);

  const getVisiblePages = () => {
    const pages = [];
    const showPages = 5;
    let start = Math.max(1, currentPage - Math.floor(showPages / 2));
    let end = Math.min(totalPages, start + showPages - 1);
    if (end - start + 1 < showPages) start = Math.max(1, end - showPages + 1);
    for (let i = start; i <= end; i++) pages.push(i);
    return pages;
  };

  if (totalPages <= 1) return null;
  const visible = getVisiblePages();

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mt-6 px-2">
      <div className="text-sm text-gray-500">
        Showing {startItem.toLocaleString()}-{endItem.toLocaleString()} of {totalItems.toLocaleString()} {noun}
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft size={20} />
        </button>

        {visible[0] > 1 && (
          <>
            <button onClick={() => onPageChange(1)} className="px-3 py-1 rounded-lg hover:bg-gray-100">1</button>
            {visible[0] > 2 && <span className="px-2 text-gray-400">...</span>}
          </>
        )}

        {visible.map(page => (
          <button
            key={page}
            onClick={() => onPageChange(page)}
            className={`px-3 py-1 rounded-lg ${currentPage === page ? 'bg-blue-600 text-white' : 'hover:bg-gray-100'}`}
          >
            {page}
          </button>
        ))}

        {visible[visible.length - 1] < totalPages && (
          <>
            {visible[visible.length - 1] < totalPages - 1 && <span className="px-2 text-gray-400">...</span>}
            <button onClick={() => onPageChange(totalPages)} className="px-3 py-1 rounded-lg hover:bg-gray-100">{totalPages}</button>
          </>
        )}

        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronRight size={20} />
        </button>
      </div>
    </div>
  );
};

const ItemCard = ({ item, rank, category }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow p-4">
      <div className="flex gap-4 items-start">
        <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-gray-600 font-semibold text-sm flex-shrink-0">
          {rank}
        </div>
        <div className="flex-grow min-w-0">
          <h3 className="font-semibold text-gray-900">
            <Title item={item} category={category} />
            {item.subtitle && <span className="font-normal text-gray-500"> · {item.subtitle}</span>}
          </h3>
          <div className="flex flex-wrap items-center gap-2 mt-1 text-sm text-gray-500">
            <span className="flex items-center gap-1"><Calendar size={14} />{item.release_date}</span>
            <span className="flex items-center gap-1"><Tag size={14} />{item.genre || 'Unknown'}</span>
            <span className="flex items-center gap-1"><BarChart3 size={14} />{item.n_reviews} reviews</span>
          </div>
        </div>
        <ScoreBadge score={item.adjusted_score} />
      </div>

      <button onClick={() => setExpanded(!expanded)} className="mt-3 text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        {expanded ? 'Less' : 'Details'}
      </button>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-100 grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-500 block">Raw Score</span>
            <span className={`font-semibold ${getScoreTextColor(item.raw_score)}`}>{item.raw_score}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Adjusted</span>
            <span className={`font-semibold ${getScoreTextColor(item.adjusted_score)}`}>{item.adjusted_score.toFixed(1)}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Adjustment</span>
            <AdjustmentBadge adjustment={item.total_adjustment} />
          </div>
          <div>
            <span className="text-gray-500 block">Divisiveness</span>
            <span className="font-semibold text-gray-700">
              {item.polarization != null ? item.polarization.toFixed(1) : '—'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

const ItemTableRow = ({ item, rank, category }) => (
  <tr className="hover:bg-gray-50 border-b border-gray-100">
    <td className="py-3 px-4 text-center text-gray-500 font-medium">{rank}</td>
    <td className="py-3 px-4">
      <Title item={item} category={category} className="font-medium text-gray-900" />
      <div className="text-sm text-gray-500">
        {[item.subtitle, item.year, item.genre || 'Unknown'].filter(Boolean).join(' · ')}
      </div>
    </td>
    <td className="py-3 px-4 text-center">
      <span className={`inline-flex items-center justify-center w-10 h-10 rounded-lg text-white font-bold text-sm ${getScoreColor(item.adjusted_score)}`}>
        {Math.round(item.adjusted_score)}
      </span>
    </td>
    <td className="py-3 px-4 text-center text-gray-600">{item.raw_score}</td>
    <td className="py-3 px-4 text-center"><AdjustmentBadge adjustment={item.total_adjustment} /></td>
    <td className="py-3 px-4 text-center text-gray-500">{item.n_reviews}</td>
    <td className="py-3 px-4 text-center text-gray-500">
      {item.polarization != null ? item.polarization.toFixed(1) : '—'}
    </td>
  </tr>
);

const ShowTableRow = ({ show, rank }) => {
  const years = show.first_year === show.last_year ? show.first_year : `${show.first_year}–${show.last_year}`;
  return (
    <tr className="hover:bg-gray-50 border-b border-gray-100">
      <td className="py-3 px-4 text-center text-gray-500 font-medium">{rank}</td>
      <td className="py-3 px-4">
        <Title item={show} category="tv" className="font-medium text-gray-900" />
        <div className="text-sm text-gray-500">
          <span title={show.n_unscored_seasons ? `${show.n_unscored_seasons} with too few reviews for a Metacritic score` : undefined}>
            {`${show.n_seasons} season${show.n_seasons === 1 ? '' : 's'}`}
          </span>
          {[years, show.genre || 'Unknown'].filter(Boolean).map(t => ` · ${t}`).join('')}
        </div>
      </td>
      <td className="py-3 px-4 text-center">
        <span className={`inline-flex items-center justify-center w-10 h-10 rounded-lg text-white font-bold text-sm ${getScoreColor(show.adjusted_score)}`}>
          {Math.round(show.adjusted_score)}
        </span>
      </td>
      <td className="py-3 px-4 text-center text-gray-600">{Math.round(show.raw_score)}</td>
      <td className="py-3 px-4 text-center"><AdjustmentBadge adjustment={show.total_adjustment} /></td>
      <td className="py-3 px-4 text-center text-gray-500">{show.n_reviews}</td>
      <td className="py-3 px-4 text-center text-sm text-gray-500 whitespace-nowrap">
        {show.n_seasons > 1
          ? <span title={`Seasons range ${Math.round(show.season_range[0])}–${Math.round(show.season_range[1])}`}>{show.best_season.replace('Season ', 'S')} · {Math.round(show.best_season_score)}</span>
          : '—'}
      </td>
    </tr>
  );
};

// Per-show TV aggregates, loaded the first time the Shows tab opens
const useShowData = (enabled) => {
  const [state, setState] = useState(null);
  useEffect(() => {
    if (!enabled || (state && !state.error)) return;
    let cancelled = false;
    fetch('/data/tv_shows_all.json')
      .then(res => { if (!res.ok) throw new Error('Failed to load TV shows'); return res.json(); })
      .then(data => { if (!cancelled) setState({ items: data.items || [], error: null }); })
      .catch(err => { if (!cancelled) setState({ items: [], error: err.message }); });
    return () => { cancelled = true; };
  }, [enabled, state]);
  return { shows: state?.items || [], showsLoading: enabled && !state, showsError: state?.error || null };
};

// Loads one category's data on first visit and caches it for the session
const useCategoryData = (category) => {
  const [cache, setCache] = useState({});
  const entry = cache[category];

  // Forget failed loads of other categories, so switching back retries them
  useEffect(() => {
    setCache(c => {
      const stale = Object.keys(c).filter(k => k !== category && c[k].error);
      if (!stale.length) return c;
      const next = { ...c };
      stale.forEach(k => delete next[k]);
      return next;
    });
  }, [category]);

  useEffect(() => {
    if (entry) return;
    let cancelled = false;
    const load = async () => {
      try {
        const [allRes, metaRes] = await Promise.all([
          fetch(`/data/${category}_all.json`),
          fetch(`/data/${category}_metadata.json`),
        ]);
        if (!allRes.ok || !metaRes.ok) throw new Error(`Failed to load ${CATEGORIES[category].label} data`);
        const [allData, metaData] = await Promise.all([allRes.json(), metaRes.json()]);
        if (!cancelled) {
          setCache(c => ({ ...c, [category]: { items: allData.items || [], metadata: metaData, error: null } }));
        }
      } catch (err) {
        console.error('Error loading data:', err);
        if (!cancelled) {
          setCache(c => ({ ...c, [category]: { items: [], metadata: null, error: err.message } }));
        }
      }
    };
    load();
    return () => { cancelled = true; };
  }, [category, entry]);

  return {
    items: entry?.items || [],
    metadata: entry?.metadata || null,
    loading: !entry,
    error: entry?.error || null,
  };
};

// Items released in the last N days OR releasing in the next few days
// (reviews often come out before release)
const filterRecent = (items, daysBack, daysForward) => {
  const cutoffPast = new Date();
  cutoffPast.setDate(cutoffPast.getDate() - daysBack);
  const cutoffFuture = new Date();
  cutoffFuture.setDate(cutoffFuture.getDate() + daysForward);
  return items.filter(item => {
    if (!item.release_date) return false;
    const releaseDate = new Date(item.release_date);
    return releaseDate >= cutoffPast && releaseDate <= cutoffFuture;
  });
};

const categoryFromHash = () => {
  const key = window.location.hash.replace('#', '');
  return CATEGORY_KEYS.includes(key) ? key : 'movies';
};

// Methodology notes that depend on what the pipeline found for this category
const Methodology = ({ category, metadata }) => {
  const cfg = CATEGORIES[category];
  const model = metadata?.model || {};
  const diag = model.diagnostics || {};
  const named = model.share_reviews_named_critic ?? diag.summary?.share_reviews_named_critic;
  const outletOnly = named != null && named < 0.2;
  const fileDrawer = diag.verdicts?.file_drawer;
  const typicalDiv = diag.summary?.median_divisiveness;
  const sel = diag.selective_coverage;

  return (
    <div className="mt-8 bg-blue-50 rounded-xl p-5 border border-blue-100">
      <h3 className="font-semibold text-blue-900 mb-2">How Adjusted Scores Work ({cfg.label})</h3>
      <p className="text-sm text-blue-800">
        Raw Metacritic scores can be biased by which critics happen to review something. Adjusted scores
        account for {outletOnly ? 'each publication’s' : 'individual critic and publication'} tendencies
        and apply Bayesian shrinkage for {cfg.noun} with few reviews.
      </p>
      <ul className="mt-2 text-sm text-blue-800 space-y-1">
        <li>• <strong>Negative adjustment:</strong> Reviewed by generous {outletOnly ? 'publications' : 'critics'}</li>
        <li>• <strong>Positive adjustment:</strong> Reviewed by harsh {outletOnly ? 'publications' : 'critics'}</li>
        {outletOnly && (
          <li>• <strong>Publication-level only:</strong> Metacritic doesn’t name the individual critic for most {cfg.label.toLowerCase()} reviews, so bias is estimated per publication</li>
        )}
        {model.excluded_release_types?.length > 0 && (
          <li>• <strong>New releases only:</strong> Reissues, deluxe editions, box sets, compilations and archival releases are left out, both from the list and from estimating publication tendencies</li>
        )}
        <li>• <strong>Shrinkage:</strong> {cfg.noun.charAt(0).toUpperCase() + cfg.noun.slice(1)} with few reviews are pulled toward the
          {model.coverage_prior ? ' average for similarly covered titles' : ' average'}</li>
        {fileDrawer != null && (
          <li>• <strong>File-drawer check:</strong>{' '}
            {fileDrawer
              ? <>publications tend to skip {cfg.noun} they would rate poorly{sel?.pooled_odds_ratio_per_10pts ? ` (each +10 points of consensus raises the odds of coverage ${sel.pooled_odds_ratio_per_10pts.toFixed(2)}×)` : ''}, so thinly reviewed {cfg.noun} are shrunk toward a lower prior</>
              : <>no evidence that publications skip {cfg.noun} they dislike, so all {cfg.noun} share one prior</>}
          </li>
        )}
        {category === 'tv' && (
          <li>• <strong>Shows tab:</strong> A show’s score averages its seasons, each counting equally. A season with few reviews is pulled toward the show’s other seasons before the overall average, since seasons of the same show tend to score alike. Seasons with too few reviews for Metacritic to score (under 4) still count this way. “Raw” is Metacritic’s show score: its scored seasons weighted by review count. “Best” is the top Metacritic-scored season</li>
        )}
        <li>• <strong>Divisiveness (Div.):</strong> How much critics disagreed — the spread (standard deviation) of the adjusted review scores{typicalDiv ? `. ~${Math.round(typicalDiv)} is typical for ${cfg.label.toLowerCase()}` : ''}</li>
      </ul>
    </div>
  );
};

export default function App() {
  const [category, setCategory] = useState(categoryFromHash);
  const { items, metadata, loading, error } = useCategoryData(category);
  const cfg = CATEGORIES[category];

  const [activeTab, setActiveTab] = useState('recent');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('all');
  const [yearRange, setYearRange] = useState({ min: '', max: '' });
  const [sortBy, setSortBy] = useState('adjusted_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);

  const itemsPerPage = 25;

  const showsMode = category === 'tv' && activeTab === 'shows';
  const { shows, showsLoading, showsError } = useShowData(showsMode);
  const noun = showsMode ? 'shows' : cfg.noun;

  // Keep the category in the URL so links can point at a section
  useEffect(() => {
    const onHash = () => setCategory(categoryFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const switchCategory = (key) => {
    if (key === category) return;
    window.location.hash = key === 'movies' ? '' : key;
    setCategory(key);
    if (activeTab === 'shows') setActiveTab('all');
    setSelectedGenre('all');
    setSearchQuery('');
    setYearRange({ min: '', max: '' });
  };

  const genres = useMemo(() => {
    if (metadata?.genres) return ['all', ...metadata.genres];
    const genreSet = new Set(items.map(m => m.genre).filter(Boolean));
    return ['all', ...Array.from(genreSet).sort()];
  }, [metadata, items]);

  const recentItems = useMemo(
    () => filterRecent(items, cfg.recentDays, cfg.forwardDays),
    [items, cfg],
  );

  const filteredItems = useMemo(() => {
    let list = showsMode ? [...shows] : activeTab === 'recent' ? [...recentItems] : [...items];

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      list = list.filter(m =>
        m.title?.toLowerCase().includes(query) || m.subtitle?.toLowerCase().includes(query));
    }

    if (selectedGenre !== 'all') {
      list = list.filter(m => m.genre === selectedGenre);
    }

    if (activeTab === 'all') {
      if (yearRange.min) list = list.filter(m => m.year >= parseInt(yearRange.min));
      if (yearRange.max) list = list.filter(m => m.year <= parseInt(yearRange.max));
    } else if (showsMode) {
      // A show matches if any of its run falls in the range
      if (yearRange.min) list = list.filter(m => m.last_year >= parseInt(yearRange.min));
      if (yearRange.max) list = list.filter(m => m.first_year <= parseInt(yearRange.max));
    }

    list.sort((a, b) => {
      if (sortBy === 'title') {
        return sortOrder === 'asc'
          ? (a.title || '').localeCompare(b.title || '')
          : (b.title || '').localeCompare(a.title || '');
      }
      const aVal = a[sortBy] ?? 0;
      const bVal = b[sortBy] ?? 0;
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return list;
  }, [showsMode, shows, activeTab, recentItems, items, searchQuery, selectedGenre, yearRange, sortBy, sortOrder]);

  const totalPages = Math.ceil(filteredItems.length / itemsPerPage);
  const paginatedItems = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredItems.slice(start, start + itemsPerPage);
  }, [filteredItems, currentPage, itemsPerPage]);

  useEffect(() => {
    setCurrentPage(1);
  }, [category, activeTab, searchQuery, selectedGenre, yearRange, sortBy, sortOrder]);

  const toggleSort = (field) => {
    if (sortBy === field) setSortOrder(o => o === 'asc' ? 'desc' : 'asc');
    else { setSortBy(field); setSortOrder('desc'); }
  };

  const SortBtn = ({ field, children }) => (
    <button onClick={() => toggleSort(field)} className={`flex items-center gap-1 ${sortBy === field ? 'text-blue-600 font-semibold' : ''}`}>
      {children}
      {sortBy === field && (sortOrder === 'desc' ? <ChevronDown size={14} /> : <ChevronUp size={14} />)}
    </button>
  );

  const EmptyIcon = cfg.icon;

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                <Star className="text-yellow-500" size={24} />
                Adjusted Metacritic Scores
              </h1>
              <p className="text-sm text-gray-500">Critic scores adjusted for bias & sample size</p>
            </div>
            <div className="text-right text-xs text-gray-400">
              {metadata?.updated && (
                <div>Updated: {new Date(metadata.updated).toLocaleDateString()}</div>
              )}
              {!loading && <div>{items.length.toLocaleString()} {cfg.noun}</div>}
            </div>
          </div>

          <nav className="flex gap-1 mt-4 border-b border-gray-200 -mb-px overflow-x-auto" aria-label="Category">
            {CATEGORY_KEYS.map(key => {
              const { label, icon: Icon } = CATEGORIES[key];
              const active = key === category;
              return (
                <button
                  key={key}
                  onClick={() => switchCategory(key)}
                  aria-current={active ? 'page' : undefined}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap ${active ? 'border-blue-600 text-blue-700' : 'border-transparent text-gray-500 hover:text-gray-800'}`}
                >
                  <Icon size={16} />{label}
                </button>
              );
            })}
          </nav>

          <div className="flex gap-2 mt-3">
            <button
              onClick={() => setActiveTab('recent')}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${activeTab === 'recent' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              {cfg.recentTab}
            </button>
            <button
              onClick={() => setActiveTab('all')}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${activeTab === 'all' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              {cfg.allTab}
            </button>
            {cfg.showsTab && (
              <button
                onClick={() => { setActiveTab('shows'); if (sortBy === 'polarization') setSortBy('adjusted_score'); }}
                className={`px-4 py-2 rounded-lg font-medium text-sm ${activeTab === 'shows' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
              >
                {cfg.showsTab}
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-6">
        {(error || (showsMode && showsError)) && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 text-red-700">
            <p className="font-medium">Error loading data</p>
            <p className="text-sm">{error || showsError}</p>
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-6">
          <div className="flex flex-wrap gap-3 items-center">
            <div className="relative flex-grow max-w-xs">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
              <input
                type="text"
                placeholder={category === 'music' ? 'Search albums or artists...' : `Search ${noun}...`}
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <select
              value={selectedGenre}
              onChange={e => setSelectedGenre(e.target.value)}
              className="border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {genres.map(g => <option key={g} value={g}>{g === 'all' ? 'All Genres' : g}</option>)}
            </select>

            {activeTab !== 'recent' && (
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  placeholder="From"
                  value={yearRange.min}
                  onChange={e => setYearRange(p => ({ ...p, min: e.target.value }))}
                  className="w-20 border border-gray-200 rounded-lg px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-gray-400">-</span>
                <input
                  type="number"
                  placeholder="To"
                  value={yearRange.max}
                  onChange={e => setYearRange(p => ({ ...p, max: e.target.value }))}
                  className="w-20 border border-gray-200 rounded-lg px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}
          </div>
        </div>

        {loading || showsLoading ? (
          <div className="text-center py-16">
            <Loader className="animate-spin mx-auto mb-4 text-blue-600" size={40} />
            <p className="text-gray-600">Loading {noun}...</p>
          </div>
        ) : (
          <>
            <div className="mb-4 text-sm text-gray-500">
              {activeTab === 'recent'
                ? `${filteredItems.length} ${cfg.noun} (last ${cfg.recentDays} days + upcoming ${cfg.forwardDays} days)`
                : `${filteredItems.length.toLocaleString()} ${noun}`}
            </div>

            {showsMode ? (
              <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-200 text-xs font-semibold text-gray-500 uppercase">
                      <tr>
                        <th className="py-3 px-4 w-12">#</th>
                        <th className="py-3 px-4 text-left"><SortBtn field="title">Show</SortBtn></th>
                        <th className="py-3 px-4 text-center w-20"><SortBtn field="adjusted_score">Adj.</SortBtn></th>
                        <th className="py-3 px-4 text-center w-16" title="Metacritic’s show score: season scores weighted by review count"><SortBtn field="raw_score">Raw</SortBtn></th>
                        <th className="py-3 px-4 text-center w-20"><SortBtn field="total_adjustment">+/-</SortBtn></th>
                        <th className="py-3 px-4 text-center w-16"><SortBtn field="n_reviews">N</SortBtn></th>
                        <th className="py-3 px-4 text-center w-24" title="Best season by adjusted score"><SortBtn field="best_season_score">Best</SortBtn></th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedItems.map((m, i) => (
                        <ShowTableRow key={m.slug} show={m} rank={(currentPage - 1) * itemsPerPage + i + 1} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : activeTab === 'recent' ? (
              <div className="space-y-3">
                {paginatedItems.map((m, i) => (
                  <ItemCard
                    key={m.slug}
                    item={m}
                    category={category}
                    rank={(currentPage - 1) * itemsPerPage + i + 1}
                  />
                ))}
              </div>
            ) : (
              <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50 border-b border-gray-200 text-xs font-semibold text-gray-500 uppercase">
                      <tr>
                        <th className="py-3 px-4 w-12">#</th>
                        <th className="py-3 px-4 text-left"><SortBtn field="title">Title</SortBtn></th>
                        <th className="py-3 px-4 text-center w-20"><SortBtn field="adjusted_score">Adj.</SortBtn></th>
                        <th className="py-3 px-4 text-center w-16"><SortBtn field="raw_score">Raw</SortBtn></th>
                        <th className="py-3 px-4 text-center w-20"><SortBtn field="total_adjustment">+/-</SortBtn></th>
                        <th className="py-3 px-4 text-center w-16"><SortBtn field="n_reviews">N</SortBtn></th>
                        <th className="py-3 px-4 text-center w-20" title="Polarization: how much critics disagreed (std of adjusted review scores)">
                          <SortBtn field="polarization">Div.</SortBtn>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedItems.map((m, i) => (
                        <ItemTableRow
                          key={m.slug}
                          item={m}
                          category={category}
                          rank={(currentPage - 1) * itemsPerPage + i + 1}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {filteredItems.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                <EmptyIcon size={48} className="mx-auto mb-4 opacity-50" />
                <p>No {noun} found.</p>
              </div>
            )}

            <Pagination
              currentPage={currentPage}
              totalPages={totalPages}
              onPageChange={setCurrentPage}
              totalItems={filteredItems.length}
              itemsPerPage={itemsPerPage}
              noun={noun}
            />
          </>
        )}

        <Methodology category={category} metadata={metadata} />
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-5xl mx-auto px-4 py-6 text-center text-sm text-gray-500">
          <p>Data sourced from Metacritic. Adjusted scores are for informational purposes.</p>
          <p className="mt-1">Methodology: Hierarchical critic effects with Bayesian shrinkage.</p>
        </div>
      </footer>
    </div>
  );
}
