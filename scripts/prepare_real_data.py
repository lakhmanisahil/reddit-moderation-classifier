"""
Convert a REAL Reddit moderation dataset into this project's format.

Recommended dataset: **HumanMOD** on Hugging Face (SamW/HumanMOD) — 8,511 posts that
moderators removed and 8,511 that they didn't. Its posts file has the columns
Reddit_ID, Subreddits, Titles, Body, Author, URLs, Labels (1 = moderated/removed).

1. Download the posts file from https://huggingface.co/datasets/SamW/HumanMOD
   (Files and versions tab) into data/raw/
2. Run:
       python scripts/prepare_real_data.py data/raw/<the file you downloaded>

It writes data/raw/reddit_posts.csv, which every other script then uses automatically.

Got a different dataset (Kaggle, Pushshift dump, your own scrape)? Add its column names to
COLUMN_MAP below. The minimum you need is a title, a body and a removed/not-removed label.
HumanMOD has no score/report columns, so the EDA correlation step will use what exists.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REAL_DATA  # noqa: E402

# their column name  →  our column name
COLUMN_MAP = {
    "Reddit_ID": "post_id", "id": "post_id",
    "Subreddits": "subreddit", "subreddit": "subreddit",
    "Titles": "title", "title": "title",
    "Body": "selftext", "selftext": "selftext", "body": "selftext",
    "Author": "author", "author": "author",
    "Labels": "is_deleted", "removed": "is_deleted", "is_removed": "is_deleted",
    "score": "score", "num_comments": "num_comments", "upvote_ratio": "upvote_ratio",
    "created_utc": "created_utc", "num_reports": "num_reports",
}


def read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main(path: str) -> None:
    df = read_any(Path(path))
    print(f"Read {len(df):,} rows with columns: {list(df.columns)}")
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    for col in ("title", "selftext", "is_deleted"):
        if col not in df.columns:
            sys.exit(f"Couldn't find a '{col}' column. Add your dataset's name for it to COLUMN_MAP.")
    df["is_deleted"] = df["is_deleted"].astype(int)
    if "num_reports" in df.columns:
        df["is_flagged"] = (df["num_reports"] > 0).astype(int)
    df.to_csv(REAL_DATA, index=False)
    print(f"Saved {len(df):,} rows to {REAL_DATA} | removed rate {df['is_deleted'].mean():.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/prepare_real_data.py <path-to-downloaded-file>")
    main(sys.argv[1])
