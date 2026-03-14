# Databricks notebook source
# MAGIC %md
# MAGIC # Steam Data Ingestion

# COMMAND ----------

import requests
import pandas as pd
import time
import re
import os
from urllib.parse import urlparse

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Validate and parse the Steam URL

# COMMAND ----------

def parse_app_id_from_url(url: str) -> str:
    """
    Validates a Steam store URL and extracts the App ID from it.

    Handles formats like:
      - https://store.steampowered.com/app/1091500/Cyberpunk_2077/
      - https://store.steampowered.com/app/1091500
      - store.steampowered.com/app/1091500/Cyberpunk_2077

    Args:
        url: Full or partial Steam store URL

    Returns:
        App ID as a string

    Raises:
        ValueError: If the URL is empty, not a Steam URL, or has no App ID
    """

    if not url or not url.strip():
        raise ValueError("No URL entered. Please provide a Steam store URL.")

    url = url.strip()

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.lower()
    if "steampowered.com" not in domain:
        raise ValueError(
            f"'{domain}' is not a Steam URL.\n"
            f"Expected: https://store.steampowered.com/app/APP_ID/Game_Name/"
        )

    if "/app/" not in parsed.path:
        raise ValueError(
            f"This Steam URL doesn't point to a game page.\n"
            f"Make sure your URL contains '/app/' — e.g. store.steampowered.com/app/1091500/"
        )

    match = re.search(r'/app/(\d+)', parsed.path)

    if not match:
        raise ValueError(
            f"Found '/app/' in the URL but could not extract a numeric App ID.\n"
            f"Expected format: store.steampowered.com/app/1091500/Game_Name/"
        )

    return match.group(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fetch the game name from the Steam store API

# COMMAND ----------

def get_game_name(app_id: str) -> str:
    """
    Fetches the official game name from the Steam store API.

    Args:
        app_id: Steam App ID as a string

    Returns:
        Game name as a string, or "Unknown Game" if it cannot be retrieved
    """

    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Response is a dict keyed by app_id
        # e.g. { "1091500": { "success": True, "data": { "name": "Cyberpunk 2077", ... } } }
        app_data = data.get(app_id, {})

        if not app_data.get("success"):
            print(f"  Warning: Steam returned success=false for App ID {app_id}. Using 'Unknown Game'.")
            return "Unknown Game"

        game_name = app_data["data"]["name"]
        print(f"  Game name: {game_name}")
        return game_name

    except Exception as e:
        print(f"  Warning: Could not retrieve game name ({e}). Using 'Unknown Game'.")
        return "Unknown Game"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fetch reviews from the Steam API

# COMMAND ----------

def get_steam_reviews(app_id: str, game_name: str, max_reviews: int = 5000, language: str = "english") -> pd.DataFrame:
    """
    Fetch Steam reviews for a given app_id using cursor-based pagination.

    Args:
        app_id: Steam App ID
        game_name: Game name to attach to every row for record keeping
        max_reviews: Maximum number of reviews to fetch
        language: Language filter (default: English)

    Returns:
        Tuple of (DataFrame of raw reviews, query_summary dict)
    """

    BASE_URL = f"https://store.steampowered.com/appreviews/{app_id}"

    all_reviews = []
    query_summary = {}
    cursor = "*"

    params = {
        "json":          1,
        "language":      language,
        "filter":        "recent",
        "review_type":   "all",
        "purchase_type": "all",
        "num_per_page":  100,
        "cursor":        cursor,
    }

    print(f"Fetching reviews for App ID: {app_id}")

    while len(all_reviews) < max_reviews:
        params["cursor"] = cursor

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 429:
            print("Rate limited — waiting 10 seconds...")
            time.sleep(10)
            continue

        response.raise_for_status()
        data = response.json()

        # Capture query_summary from the first page only
        if cursor == "*":
            query_summary = data.get("query_summary", {})

        if data.get("success") != 1:
            print("API returned success=0, stopping.")
            break

        reviews_batch = data.get("reviews", [])

        if not reviews_batch:
            print("No more reviews available.")
            break

        all_reviews.extend(reviews_batch)
        print(f"  Fetched {len(all_reviews)} reviews so far...")

        new_cursor = data.get("cursor")
        if not new_cursor or new_cursor == cursor:
            print("Cursor unchanged — end of results.")
            break

        cursor = new_cursor
        time.sleep(0.5)

    print(f"Done. Total reviews fetched: {len(all_reviews)}")
    return parse_reviews(all_reviews[:max_reviews], game_name), query_summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Flatten nested JSON into a DataFrame

# COMMAND ----------

def parse_reviews(raw_reviews: list, game_name: str) -> pd.DataFrame:
    """
    Flatten the nested JSON response into a clean DataFrame.

    Args:
        raw_reviews: List of raw review dicts from the Steam API
        game_name: Game name to attach to every row

    Returns:
        Cleaned DataFrame with one row per review
    """

    parsed = []

    for r in raw_reviews:
        author = r.get("author", {})

        parsed.append({
            # --- Record keeping ---
            "game_name":                    game_name,

            # --- Core fields ---
            "recommendationid":             r.get("recommendationid"),
            "review":                       r.get("review"),
            "voted_up":                     r.get("voted_up"),
            "timestamp_created":            r.get("timestamp_created"),
            "language":                     r.get("language"),
            "playtime_forever_mins":        author.get("playtime_forever", 0),

            # --- Enrich fields ---
            "votes_up":                     r.get("votes_up", 0),
            "timestamp_updated":            r.get("timestamp_updated"),
            "playtime_last_two_weeks":      author.get("playtime_last_two_weeks", 0),
            "author_num_reviews":           author.get("num_reviews", 0),

            # --- New review level fields ---
            "weighted_vote_score":          float(r.get("weighted_vote_score", 0)),
            "comment_count":                r.get("comment_count", 0),
            "steam_purchase":               r.get("steam_purchase"),
            "received_for_free":            r.get("received_for_free"),
            "written_during_early_access":  r.get("written_during_early_access"),
            "playtime_at_review_mins":      author.get("playtime_at_review", 0),
            "developer_response":           r.get("developer_response"),
            "timestamp_dev_responded":      r.get("timestamp_dev_responded"),
        })

    return pd.DataFrame(parsed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entry point

# COMMAND ----------

# ── Workflow parameter — Steam URL ───────────────────────────────
dbutils.widgets.text("steam_url", "", "Steam Store URL")
raw_url = dbutils.widgets.get("steam_url")

if not raw_url:
    raise ValueError("No Steam URL provided. Set the steam_url widget or job parameter.")

try:
    app_id = parse_app_id_from_url(raw_url)
    print(f"✓ App ID found: {app_id}")
except ValueError as e:
    raise ValueError(f"Invalid Steam URL: {e}")

print("Looking up game name...")
game_name = get_game_name(app_id)

df, query_summary = get_steam_reviews(app_id=app_id, game_name=game_name, max_reviews=500000)

df_summary = pd.DataFrame([{
    "game_name":         game_name,
    "app_id":            app_id,
    "total_positive":    query_summary.get("total_positive"),
    "total_negative":    query_summary.get("total_negative"),
    "total_reviews":     query_summary.get("total_reviews"),
    "review_score_desc": query_summary.get("review_score_desc"),
    "ingestion_date":    pd.Timestamp.now().strftime("%Y-%m-%d"),
}])

os.makedirs("/Volumes/main/steam_analytics/raw_data/bronze", exist_ok=True)
df_summary.to_csv("/Volumes/main/steam_analytics/raw_data/bronze/game_summary.csv", index=False)
df.to_csv("/Volumes/main/steam_analytics/raw_data/bronze/steam_reviews.csv", index=False)

print(f"✓ Reviews saved: {len(df)} rows")
print(f"✓ Summary saved: {df_summary.iloc[0]['review_score_desc']} — {df_summary.iloc[0]['total_reviews']:,} total reviews")