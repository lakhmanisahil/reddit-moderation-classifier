"""Load the dataset (real if you have it, synthetic otherwise) and clean it."""
import pandas as pd

from src.config import REAL_DATA, SYNTHETIC_DATA, TEXT_COLS

REQUIRED = ["title", "selftext", "is_deleted"]   # the minimum the models need


def load_raw(path=None) -> pd.DataFrame:
    """Read the CSV. Uses the real dataset automatically when it exists."""
    if path is None:
        path = REAL_DATA if REAL_DATA.exists() else SYNTHETIC_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"No data at {path}. Run `python scripts/make_synthetic_data.py` "
            "or put a real dataset at data/raw/reddit_posts.csv"
        )
    print(f"Loading {path.name}")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning. Each rule is explained in docs/STEP_BY_STEP.md (Step 4)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    n0 = len(df)
    df = df.copy()

    # 1. Empty bodies are normal on Reddit (title-only / link posts) → empty string, not NaN
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    # 2. Real Reddit dumps show "[removed]" / "[deleted]" instead of the text of removed posts.
    #    Those rows would teach the model "the word 'removed' means deleted" = data leakage.
    placeholder = df["selftext"].str.strip().isin(["[removed]", "[deleted]"])
    df.loc[placeholder, "selftext"] = ""

    # 3. The same post scraped twice — keep the first copy.
    #    Use the post id when we have one; two different users CAN both post "help" with no body.
    key = ["post_id"] if "post_id" in df else [c for c in ("title", "selftext", "author", "subreddit") if c in df]
    df = df.drop_duplicates(subset=key)

    # 4. Posts with no text at all can't be classified from text
    df = df[(df["title"].str.strip() + df["selftext"].str.strip()) != ""]

    # 5. Target must be 0/1
    df["is_deleted"] = df["is_deleted"].astype(int)

    # 6. Helpful derived columns
    df["text"] = (df["title"] + " " + df["selftext"]).str.strip()
    df["text_length"] = df["text"].str.split().str.len()
    if "created_utc" in df:
        df["created_at"] = pd.to_datetime(df["created_utc"], unit="s")

    print(f"Kept {len(df):,} of {n0:,} rows | deleted rate {df['is_deleted'].mean():.1%}")
    return df.reset_index(drop=True)


def load_clean_data(path=None) -> pd.DataFrame:
    return clean(load_raw(path))
