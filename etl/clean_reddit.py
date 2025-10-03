# etl/clean_reddit.py
from pathlib import Path
import re
import pandas as pd

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUT_CLEAN = CLEAN_DIR / "reddit_clean.csv"

# --- Helpers ---
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HTML_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
CODEBLOCK_RE = re.compile(r"`{1,3}.*?`{1,3}", re.DOTALL)

def _basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text

    # Remove code blocks / inline code first
    t = CODEBLOCK_RE.sub(" ", t)

    # Replace markdown links with just the visible text
    t = MD_LINK_RE.sub(r"\1", t)

    # Strip URLs and HTML
    t = URL_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)

    # Remove typical Reddit noise
    t = t.replace("&amp;", "&")
    t = t.replace("&lt;", "<").replace("&gt;", ">")

    # Collapse whitespace
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def clean_reddit_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a text_clean column and light filtering."""
    if df is None or df.empty:
        return df

    out = df.copy()
    # Prefer the 'body' column; create text_clean
    out["text_clean"] = (
        out.get("body", "")
        .fillna("")
        .map(_basic_clean)
    )

    # Drop rows that are clearly non-content
    bad_vals = {"", "[deleted]", "[removed]"}
    out = out[~out["text_clean"].isin(bad_vals)]

    # Drop very short remnants
    out = out[out["text_clean"].str.len() >= 5]

    # Keep a tidy set of columns (add any others you need)
    keep = [c for c in ["source", "subreddit", "comment_id", "author",
                        "score", "created_utc", "permalink", "body", "text_clean"]
            if c in out.columns]
    out = out[keep]
    return out.reset_index(drop=True)

def save_clean(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)
