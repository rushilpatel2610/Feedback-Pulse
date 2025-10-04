# dashboard/app.py
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feedback Pulse – Reddit", layout="wide")

@st.cache_data(show_spinner=False)
def load_csv_safe(path: Path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load {path.name}: {e}")
        return None

def main():
    st.title("🔎 Feedback Pulse – Reddit Sentiment")

    base = Path(__file__).resolve().parents[1]
    scored_path = base / "data" / "processed" / "reddit_scored.csv"
    clean_path  = base / "data" / "clean" / "reddit_clean.csv"  # not strictly needed to render

    df = load_csv_safe(scored_path)
    clean_df = load_csv_safe(clean_path)  # optional, might be None

    if df is None:
        st.warning("No processed data found. Run the ETL locally with `python -m etl.flow`, "
                   "then commit & push the resulting CSVs if you want them visible on Cloud.")
        # Show tiny demo so the page isn’t blank
        demo = pd.DataFrame({
            "subreddit": ["technology", "apple", "android"],
            "author": ["demo1", "demo2", "demo3"],
            "sentiment_label": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
            "sentiment_score": [0.91, 0.51, 0.98],
            "text_clean": ["Great!", "meh", "ugh"],
            "permalink": ["", "", ""],
        })
        st.dataframe(demo, use_container_width=True, height=240)
        return

    # Basic hygiene
    if "created_utc" in df.columns:
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

    # Controls
    left, right = st.columns(2)
    with left:
        subs = sorted(df["subreddit"].dropna().unique().tolist()) if "subreddit" in df else []
        chosen_subs = st.multiselect("Subreddits", subs, default=subs)
    with right:
        labels = sorted(df["sentiment_label"].dropna().unique().tolist()) if "sentiment_label" in df else []
        chosen_labels = st.multiselect("Sentiment labels", labels, default=labels)

    # Filter
    f = df.copy()
    if chosen_subs:
        f = f[f["subreddit"].isin(chosen_subs)]
    if chosen_labels:
        f = f[f["sentiment_label"].isin(chosen_labels)]

    st.markdown(f"**Rows:** {len(f)}")
    cols_to_show = [c for c in ["subreddit","author","sentiment_label","sentiment_score","text_clean","permalink"] if c in f.columns]
    st.dataframe(
        f[cols_to_show].sort_values("sentiment_score", ascending=False),
        use_container_width=True,
        height=350
    )

    c1, c2 = st.columns(2)
    with c1:
        if "sentiment_label" in f:
            st.subheader("Counts by sentiment")
            fig = px.histogram(f, x="sentiment_label")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if {"subreddit", "sentiment_score"}.issubset(f.columns):
            st.subheader("Avg sentiment score by subreddit")
            agg = f.groupby("subreddit")["sentiment_score"].mean().reset_index().sort_values("sentiment_score", ascending=False)
            fig2 = px.bar(agg, x="subreddit", y="sentiment_score")
            st.plotly_chart(fig2, use_container_width=True)

    if "created_dt" in f.columns:
        st.subheader("Volume over time")
        ts = f.set_index("created_dt").resample("1H").size().rename("count").reset_index()
        fig3 = px.line(ts, x="created_dt", y="count")
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
