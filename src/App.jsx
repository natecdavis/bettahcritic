import React, { useState, useMemo, useEffect } from 'react';
import { Search, TrendingUp, TrendingDown, Calendar, Film, ChevronDown, ChevronUp, Star, BarChart3, ChevronLeft, ChevronRight, Loader } from 'lucide-react';

// Metacritic color scheme: Green 61+, Yellow 40-60, Red 39-
const getScoreColor = (score) => {
  if (score >= 61) return 'bg-green-600';
  if (score >= 40) return 'bg-yellow-500';
  return 'bg-red-600';
};

const getScoreTextColor = (score) => {
  if (score >= 61) return 'text-green-600';
  if (score >= 40) return 'text-yellow-600';
  return 'text-red-600';
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

// Pagination component
const Pagination = ({ currentPage, totalPages, onPageChange, totalItems, itemsPerPage }) => {
  const startItem = (currentPage - 1) * itemsPerPage + 1;
  const endItem = Math.min(currentPage * itemsPerPage, totalItems);
  
  const getVisiblePages = () => {
    const pages = [];
    const showPages = 5;
    let start = Math.max(1, currentPage - Math.floor(showPages / 2));
    let end = Math.min(totalPages, start + showPages - 1);
    
    if (end - start + 1 < showPages) {
      start = Math.max(1, end - showPages + 1);
    }
    
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  };

  if (totalPages <= 1) return null;

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mt-6 px-2">
      <div className="text-sm text-gray-500">
        Showing {startItem.toLocaleString()}-{endItem.toLocaleString()} of {totalItems.toLocaleString()} movies
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="p-2 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft size={20} />
        </button>
        
        {getVisiblePages()[0] > 1 && (
          <>
            <button onClick={() => onPageChange(1)} className="px-3 py-1 rounded-lg hover:bg-gray-100">1</button>
            {getVisiblePages()[0] > 2 && <span className="px-2 text-gray-400">...</span>}
          </>
        )}
        
        {getVisiblePages().map(page => (
          <button
            key={page}
            onClick={() => onPageChange(page)}
            className={`px-3 py-1 rounded-lg ${currentPage === page ? 'bg-blue-600 text-white' : 'hover:bg-gray-100'}`}
          >
            {page}
          </button>
        ))}
        
        {getVisiblePages()[getVisiblePages().length - 1] < totalPages && (
          <>
            {getVisiblePages()[getVisiblePages().length - 1] < totalPages - 1 && <span className="px-2 text-gray-400">...</span>}
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

const MovieCard = ({ movie, rank }) => {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow p-4">
      <div className="flex gap-4 items-start">
        <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-gray-600 font-semibold text-sm flex-shrink-0">
          {rank}
        </div>
        <div className="flex-grow min-w-0">
          <h3 className="font-semibold text-gray-900">{movie.title}</h3>
          <div className="flex flex-wrap items-center gap-2 mt-1 text-sm text-gray-500">
            <span className="flex items-center gap-1"><Calendar size={14} />{movie.release_date}</span>
            <span className="flex items-center gap-1"><Film size={14} />{movie.genre || 'Unknown'}</span>
            <span className="flex items-center gap-1"><BarChart3 size={14} />{movie.n_reviews} reviews</span>
          </div>
        </div>
        <ScoreBadge score={movie.adjusted_score} />
      </div>
      
      <button onClick={() => setExpanded(!expanded)} className="mt-3 text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">
        {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        {expanded ? 'Less' : 'Details'}
      </button>
      
      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-100 grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500 block">Raw Score</span>
            <span className={`font-semibold ${getScoreTextColor(movie.raw_score)}`}>{movie.raw_score}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Adjusted</span>
            <span className={`font-semibold ${getScoreTextColor(movie.adjusted_score)}`}>{movie.adjusted_score.toFixed(1)}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Adjustment</span>
            <AdjustmentBadge adjustment={movie.total_adjustment} />
          </div>
        </div>
      )}
    </div>
  );
};

const MovieTableRow = ({ movie, rank }) => (
  <tr className="hover:bg-gray-50 border-b border-gray-100">
    <td className="py-3 px-4 text-center text-gray-500 font-medium">{rank}</td>
    <td className="py-3 px-4">
      <div className="font-medium text-gray-900">{movie.title}</div>
      <div className="text-sm text-gray-500">{movie.year} · {movie.genre || 'Unknown'}</div>
    </td>
    <td className="py-3 px-4 text-center">
      <span className={`inline-flex items-center justify-center w-10 h-10 rounded-lg text-white font-bold text-sm ${getScoreColor(movie.adjusted_score)}`}>
        {Math.round(movie.adjusted_score)}
      </span>
    </td>
    <td className="py-3 px-4 text-center text-gray-600">{movie.raw_score}</td>
    <td className="py-3 px-4 text-center"><AdjustmentBadge adjustment={movie.total_adjustment} /></td>
    <td className="py-3 px-4 text-center text-gray-500">{movie.n_reviews}</td>
  </tr>
);

// Data loading hook
const useMovieData = () => {
  const [allMovies, setAllMovies] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Load all movies and metadata
        const [allRes, metaRes] = await Promise.all([
          fetch('/data/movies_all.json'),
          fetch('/data/metadata.json'),
        ]);
        
        if (!allRes.ok || !metaRes.ok) {
          throw new Error('Failed to load movie data');
        }
        
        const allData = await allRes.json();
        const metaData = await metaRes.json();
        
        setAllMovies(allData.movies || []);
        setMetadata(metaData);
        setError(null);
      } catch (err) {
        console.error('Error loading data:', err);
        setError(err.message);
        setAllMovies([]);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, []);

  return { allMovies, metadata, loading, error };
};

// Filter movies released in last N days (for "in theaters" approximation)
const filterRecentMovies = (movies, days = 60) => {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);
  const today = new Date();
  
  return movies.filter(movie => {
    if (!movie.release_date) return false;
    const releaseDate = new Date(movie.release_date);
    return releaseDate >= cutoffDate && releaseDate <= today;
  });
};

export default function App() {
  const { allMovies, metadata, loading, error } = useMovieData();
  
  const [activeTab, setActiveTab] = useState('recent');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('all');
  const [yearRange, setYearRange] = useState({ min: '', max: '' });
  const [sortBy, setSortBy] = useState('adjusted_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  
  const itemsPerPage = 25;
  
  // Get unique genres from metadata or compute from data
  const genres = useMemo(() => {
    if (metadata?.genres) {
      return ['all', ...metadata.genres];
    }
    const genreSet = new Set(allMovies.map(m => m.genre).filter(Boolean));
    return ['all', ...Array.from(genreSet).sort()];
  }, [metadata, allMovies]);
  
  // Get recent movies (last 60 days)
  const recentMovies = useMemo(() => {
    return filterRecentMovies(allMovies, 60);
  }, [allMovies]);
  
  // Filter and sort movies
  const filteredMovies = useMemo(() => {
    let movies = activeTab === 'recent' ? [...recentMovies] : [...allMovies];
    
    // Apply search
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      movies = movies.filter(m => m.title?.toLowerCase().includes(query));
    }
    
    // Apply genre filter
    if (selectedGenre !== 'all') {
      movies = movies.filter(m => m.genre === selectedGenre);
    }
    
    // Apply year range (only for all movies tab)
    if (activeTab === 'all') {
      if (yearRange.min) movies = movies.filter(m => m.year >= parseInt(yearRange.min));
      if (yearRange.max) movies = movies.filter(m => m.year <= parseInt(yearRange.max));
    }
    
    // Sort
    movies.sort((a, b) => {
      if (sortBy === 'title') {
        return sortOrder === 'asc' 
          ? (a.title || '').localeCompare(b.title || '')
          : (b.title || '').localeCompare(a.title || '');
      }
      const aVal = a[sortBy] ?? 0;
      const bVal = b[sortBy] ?? 0;
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });
    
    return movies;
  }, [activeTab, recentMovies, allMovies, searchQuery, selectedGenre, yearRange, sortBy, sortOrder]);
  
  // Pagination
  const totalPages = Math.ceil(filteredMovies.length / itemsPerPage);
  const paginatedMovies = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return filteredMovies.slice(start, start + itemsPerPage);
  }, [filteredMovies, currentPage, itemsPerPage]);
  
  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [activeTab, searchQuery, selectedGenre, yearRange, sortBy, sortOrder]);
  
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader className="animate-spin mx-auto mb-4 text-blue-600" size={48} />
          <p className="text-gray-600">Loading movies...</p>
        </div>
      </div>
    );
  }

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
              <div>{allMovies.length.toLocaleString()} movies</div>
            </div>
          </div>
          
          <div className="flex gap-2 mt-4">
            <button 
              onClick={() => setActiveTab('recent')}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${activeTab === 'recent' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              In Theaters
            </button>
            <button 
              onClick={() => setActiveTab('all')}
              className={`px-4 py-2 rounded-lg font-medium text-sm ${activeTab === 'all' ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
            >
              All Movies
            </button>
          </div>
        </div>
      </header>
      
      <main className="max-w-5xl mx-auto px-4 py-6">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6 text-red-700">
            <p className="font-medium">Error loading data</p>
            <p className="text-sm">{error}</p>
          </div>
        )}
        
        {/* Filters */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-6">
          <div className="flex flex-wrap gap-3 items-center">
            <div className="relative flex-grow max-w-xs">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
              <input 
                type="text" 
                placeholder="Search movies..." 
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
            
            {activeTab === 'all' && (
              <div className="flex items-center gap-2">
                <input 
                  type="number" 
                  placeholder="From" 
                  value={yearRange.min} 
                  onChange={e => setYearRange(p => ({...p, min: e.target.value}))}
                  className="w-20 border border-gray-200 rounded-lg px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" 
                />
                <span className="text-gray-400">-</span>
                <input 
                  type="number" 
                  placeholder="To" 
                  value={yearRange.max} 
                  onChange={e => setYearRange(p => ({...p, max: e.target.value}))}
                  className="w-20 border border-gray-200 rounded-lg px-2 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500" 
                />
              </div>
            )}
          </div>
        </div>
        
        {/* Results info */}
        <div className="mb-4 text-sm text-gray-500">
          {activeTab === 'recent' 
            ? `${filteredMovies.length} movies from the last 60 days`
            : `${filteredMovies.length.toLocaleString()} movies`
          }
        </div>
        
        {/* Content */}
        {activeTab === 'recent' ? (
          <div className="space-y-3">
            {paginatedMovies.map((m, i) => (
              <MovieCard 
                key={m.movie_slug} 
                movie={m} 
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
                  </tr>
                </thead>
                <tbody>
                  {paginatedMovies.map((m, i) => (
                    <MovieTableRow 
                      key={m.movie_slug} 
                      movie={m} 
                      rank={(currentPage - 1) * itemsPerPage + i + 1} 
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        {filteredMovies.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <Film size={48} className="mx-auto mb-4 opacity-50" />
            <p>No movies found.</p>
          </div>
        )}
        
        {/* Pagination */}
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={setCurrentPage}
          totalItems={filteredMovies.length}
          itemsPerPage={itemsPerPage}
        />
        
        {/* Info box */}
        <div className="mt-8 bg-blue-50 rounded-xl p-5 border border-blue-100">
          <h3 className="font-semibold text-blue-900 mb-2">How Adjusted Scores Work</h3>
          <p className="text-sm text-blue-800">
            Raw Metacritic scores can be biased by which critics review a film. Adjusted scores account for 
            individual critic tendencies and apply Bayesian shrinkage for films with few reviews.
          </p>
          <ul className="mt-2 text-sm text-blue-800 space-y-1">
            <li>• <strong>Negative adjustment:</strong> Reviewed by generous critics</li>
            <li>• <strong>Positive adjustment:</strong> Reviewed by harsh critics</li>
            <li>• <strong>Shrinkage:</strong> Films with few reviews are pulled toward the average</li>
          </ul>
        </div>
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
