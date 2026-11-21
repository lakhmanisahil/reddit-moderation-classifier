from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
REAL_DATA = DATA_RAW / "reddit_posts.csv"            # your real dataset (see scripts/prepare_real_data.py)
SYNTHETIC_DATA = DATA_RAW / "reddit_posts_synthetic.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# ── Reproducibility ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2          # 80 % train / 20 % test

# ── Columns ──────────────────────────────────────────────────────────────
TEXT_COLS = ["title", "selftext"]       # joined into one "text" column
TARGET = "is_deleted"                   # what the classical models predict
NUMERIC_COLS = [                        # used in the EDA correlation analysis
    "score", "upvote_ratio", "num_comments", "num_reports",
    "author_karma", "account_age_days", "text_length",
    "is_flagged", "is_deleted",
]

# ── TF-IDF ───────────────────────────────────────────────────────────────
TFIDF_PARAMS = dict(
    max_features=20_000,   # keep the 20k most useful words/phrases
    ngram_range=(1, 2),    # single words AND two-word phrases ("free trial")
    min_df=2,              # ignore words that appear in only one post (typos, names)
    max_df=0.9,            # ignore words in >90 % of posts (no signal)
    sublinear_tf=True,     # 1 + log(tf): a word said 10x isn't 10x more important
)

# ── Gemini ───────────────────────────────────────────────────────────────
# Gemini 2.0 Flash was shut down on 1 June 2026 (calls to it now fail), so the default is
# Google's listed replacement. Override with GEMINI_MODEL in your .env. Current list:
# https://ai.google.dev/gemini-api/docs/models
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_SAMPLE_SIZE = 150     # test posts per evaluation run: ~45 deleted posts → a usable F1, still cheap
MODERATION_LABELS = ["SAFE", "SPAM", "TOXIC", "LOW_EFFORT", "MISINFORMATION"]


def ensure_dirs() -> None:
    for p in (DATA_RAW, ARTIFACTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
