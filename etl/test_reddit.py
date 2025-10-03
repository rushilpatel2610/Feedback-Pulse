import os
import praw
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Create Reddit client
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

print("✅ Reddit connection OK?", reddit.read_only)

# Fetch a few comments from r/technology
print("\n--- Latest comments from r/technology ---")
for c in reddit.subreddit("technology").comments(limit=5):
    print("-", c.body[:120].replace("\n", " "), "...")
