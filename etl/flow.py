# etl/flow.py
from pathlib import Path
import os
import time
from typing import List, Optional

import pandas as pd
from prefect import flow, task
from dotenv import load_dotenv
import praw

from etl.clean_reddit import clean_reddit_df, save_clean, CLEAN_DIR, OUT_CLEAN

# ---------- Paths ----------
BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

RAW_REDDIT = RAW_DIR / "reddit_comments.csv"
OUT_REDDIT = PROC_DIR / "reddit_scored.csv"

# ---------- Config ----------
load_dotenv(BASE / ".env")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "feedback-pulse")

SUBREDDITS_DEFAULT = ["technology", "apple", "android"]
COMMENTS_PER_SUBREDDIT = 200

# ---------- Tasks ----------
@task
def make_reddit_client() -> praw.Reddit:
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        raise RuntimeError("Missing Reddit credentials in .env")
    r = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    _ = r.read_only  # sanity check
    return r

@task
def fetch_reddit_comments(
    reddit: praw.Reddit,
    subreddits: Optional[List[str]] = None,
    per_sub: int = COMMENTS_PER_SUBREDDIT,
) -> pd.DataFrame:
    subreddits = subreddits or SUBREDDITS_DEFAULT
    rows = []
    for sub in subreddits:
        sr = reddit.subreddit(sub)
        for c in sr.comments(limit=per_sub):
            try:
                rows.append(
                    {
                        "source": "reddit",
                        "subreddit": sub,
                        "comment_id": c.id,
                        "link_id": getattr(c, "link_id", None),
                        "author": str(getattr(c, "author", "")),
                        "body": c.body,
                        "score": int(getattr(c, "score", 0)),
                        "created_utc": float(getattr(c, "created_utc", 0.0)),
                        "permalink": f"https://www.reddit.com{getattr(c, 'permalink', '')}",
                    }
                )
            except Exception:
                continue
        time.sleep(0.4)  # be polite to API
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No comments fetched. Try different subreddits or increase per_sub.")
    return df

@task
def save_raw(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)

@task
def clean_reddit(df: pd.DataFrame) -> pd.DataFrame:
    return clean_reddit_df(df)

@task
def save_clean_task(df: pd.DataFrame, path: Path) -> str:
    return save_clean(df, path)

@task
def score_sentiment_reddit(df: pd.DataFrame) -> pd.DataFrame:
    import torch
    from transformers import pipeline

    if df.empty:
        return df

    # Prefer cleaned text if available
    text_col = "text_clean" if "text_clean" in df.columns else "body"
    texts = (
        df[text_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .tolist()
    )

    device_arg = 0 if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else -1

    clf = pipeline(
        task="sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device_arg,
        truncation=True,
        max_length=512,
    )

    preds = clf(texts, batch_size=16, truncation=True, max_length=512)

    out = df.copy()
    out["sentiment_label"] = [p["label"] for p in preds]
    out["sentiment_score"] = [float(p["score"]) for p in preds]
    return out

@task
def save_processed(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)

# ---------- Flow ----------
@flow(name="feedback-pulse-reddit")
def run_reddit_flow(
    subreddits: Optional[List[str]] = None,
    per_sub: int = COMMENTS_PER_SUBREDDIT,
):
    reddit = make_reddit_client()
    df_raw = fetch_reddit_comments(reddit, subreddits=subreddits, per_sub=per_sub)
    raw_path = save_raw(df_raw, RAW_REDDIT)
    print(f"Saved raw Reddit comments → {raw_path}")

    df_clean = clean_reddit(df_raw)
    clean_path = save_clean_task(df_clean, OUT_CLEAN)
    print(f"Saved CLEAN Reddit comments → {clean_path}")

    df_scored = score_sentiment_reddit(df_clean)
    scored_path = save_processed(df_scored, OUT_REDDIT)
    print(f"Saved scored Reddit comments → {scored_path}")

if __name__ == "__main__":
    run_reddit_flow()
