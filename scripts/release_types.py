"""
Classify music releases that aren't new albums: reissues/deluxe editions/box
sets, compilations, and archival releases of old recordings.

Metacritic has no field for this, so it's inferred from the title and the
listing description (e.g. "The Pixies 1989 album is reissued with demos...").
Patterns are deliberately specific: "the deluxe edition includes two bonus
tracks" describes a new album, and a band celebrating its "30th anniversary"
with a new record is still a new record. Albums with no description can only
be caught by title, so config/music_release_overrides.csv lists manual
corrections in both directions.

classify() returns '' for a regular release, else 'reissue', 'compilation' or
'archival'. Live albums, soundtracks, EPs and mixtapes count as regular unless
they are archival (recorded long before release).
"""

import os
import re

import pandas as pd

OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                              'config', 'music_release_overrides.csv')

TITLE_REISSUE = re.compile(
    r"deluxe|box ?set|boxed set|reissue|remaster|anniversary|expanded edition|legacy edition"
    r"|collector'?s edition|special edition|immersion edition|definitive edition"
    r"|taylor'?s version|\boknotok\b", re.I)
TITLE_COMPILATION = re.compile(
    r"greatest hits|^(?:the )?(?:very )?best of\b|b-sides|\brarities\b"
    r"|^the singles\b|^singles\b|\bsingles collection\b|\banthology\b(?! of)", re.I)
TITLE_ARCHIVAL = re.compile(
    r"bootleg series|peel sessions|^the complete\b|complete (?:sessions|recordings|studio)"
    r"|\blost (?:album|tapes|recordings)\b|\bunreleased\b|\bthe \w+ sessions\b", re.I)
# A live/session recording named for an old year ("Live at the Academy of
# Music 1971", released 2013). Bare year titles like "1989" are left alone.
TITLE_DATED_RECORDING = re.compile(
    r"\b(?:live|sessions?|concert|recordings|tapes|demos|lost)\b.*\b((?:19|20)\d\d)\b"
    r"|\b((?:19|20)\d\d)\b.*\b(?:live|sessions?|concert|recordings|tapes|demos)\b", re.I)
TITLE_YEAR_RANGE = re.compile(r"\b((?:19|20)\d\d)\s*[-–]\s*((?:19|20)?\d\d)\b")

DESC_REISSUE = re.compile(
    r"\bis reissued|\breissue[ds]?\b|\bre-?issue of|\bre-?released\b|\bre-?release of"
    r"|\bremastered\b|\bremaster of|anniversary (?:edition|reissue|re-?release|box)"
    r"|\bbox(?:ed)? set\b|\bexpanded (?:edition|reissue|version)\b"
    r"|\b(?:cd|compact disc) debut\b", re.I)
# Describes the release itself as a new album; overrides compilation/archival
# cues like "featuring previously unreleased material" or "two songs from its
# 2014 compilation"
DESC_NEW_ALBUM = re.compile(
    r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth"
    r"|tenth|debut|latest|new) (?:full-length |solo |studio |proper )*(?:album|release|effort|lp)\b",
    re.I)
DESC_COMPILATION = re.compile(
    r"(?:^|\bis an? |\bthis |\bthe |\blatest |\bsecond |\bthird )(?:[\w-]+ ){0,2}compilation\b(?! of (?:his|her|their|its) )"
    r"|\bgreatest[- ]hits\b|\bbest[- ]of (?:collection|compilation|album|set)\b|career[- ]spanning"
    r"|\bcollects (?:the |some |rare |previously |together )|\bcollection of (?:singles|hits|b-sides|rare|previously)"
    r"|\bb-sides\b|\brarities\b", re.I)
DESC_ARCHIVAL = re.compile(
    r"never[- ]released|\bbootleg(?:ged)?\b|\barchival\b|\barchive release\b|\bshelved\b|\blost album\b"
    r"|\bdemos\b|\bouttakes\b|\balternate (?:takes|versions|mixes)\b", re.I)
DESC_OLD_RECORDING = re.compile(
    r"\b(?:recorded|culled from|recordings? from|performances? (?:in|at|from)|concert (?:in|at|from)"
    r"|sessions? (?:in|from)|tapes? from)\b[^.]{0,80}?\b((?:19|20)\d\d)\b", re.I)

YEARS_OLD_FOR_ARCHIVAL = 10


def _year_range_is_retrospective(title: str, release_year) -> bool:
    m = TITLE_YEAR_RANGE.search(title)
    if not m or release_year is None:
        return False
    end = m.group(2)
    end_year = int(end) if len(end) == 4 else int(m.group(1)[:2] + end)
    return end_year <= release_year - 3


def classify(title: str, description: str, release_year=None) -> str:
    title = ' '.join(title.split()) if isinstance(title, str) else ''
    description = description if isinstance(description, str) else ''
    if pd.isna(release_year):
        release_year = None

    if TITLE_REISSUE.search(title) or DESC_REISSUE.search(description):
        return 'reissue'
    if TITLE_COMPILATION.search(title):
        return 'compilation'
    if TITLE_ARCHIVAL.search(title) or _year_range_is_retrospective(title, release_year):
        return 'archival'
    dated = TITLE_DATED_RECORDING.search(title)
    if dated and release_year is not None:
        year = int(dated.group(1) or dated.group(2))
        if year <= release_year - YEARS_OLD_FOR_ARCHIVAL:
            return 'archival'
    if DESC_NEW_ALBUM.search(description):
        return ''
    if DESC_COMPILATION.search(description):
        return 'compilation'
    if DESC_ARCHIVAL.search(description):
        return 'archival'
    m = DESC_OLD_RECORDING.search(description)
    if m and release_year is not None and int(m.group(1)) <= release_year - YEARS_OLD_FOR_ARCHIVAL:
        return 'archival'
    return ''


def load_overrides() -> dict:
    """item_slug -> release type ('' forces "regular")."""
    if not os.path.exists(OVERRIDES_PATH):
        return {}
    df = pd.read_csv(OVERRIDES_PATH, comment='#', dtype=str, keep_default_na=False)
    return dict(zip(df['item_slug'].str.strip(), df['release_type'].str.strip()))


def classify_items(items: pd.DataFrame, slug_col: str = 'item_slug') -> pd.Series:
    desc = items['description'] if 'description' in items.columns else pd.Series('', index=items.index)
    years = pd.to_datetime(items['release_date'], errors='coerce').dt.year
    types = pd.Series([classify(t, d, y) for t, d, y in zip(items['title'], desc, years)],
                      index=items.index)
    overrides = load_overrides()
    if overrides:
        forced = items[slug_col].map(overrides)
        types = forced.where(forced.notna(), types)
    return types
