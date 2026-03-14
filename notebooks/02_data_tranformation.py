# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 — Silver Layer Transformations

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Loading Bronze Table

# COMMAND ----------

df_bronze = spark.read.csv(
    "/Volumes/main/steam_analytics/raw_data/bronze/steam_reviews.csv",
    header=True,
    inferSchema=True
)

print(f"  Rows loaded: {df_bronze.count()}")
print(f"  Columns: {df_bronze.columns}")
display(df_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deduplicate by recommendationid

# COMMAND ----------

# Each review has a unique recommendationid from Steam.
# If the ingestion notebook is run multiple times, duplicates can sneak in.
# This keeps only the first occurrence of each review.

print("Deduplicating...")

df_deduped = df_bronze.dropDuplicates(["recommendationid"])

dupes_removed = df_bronze.count() - df_deduped.count()
print(f"  Duplicate rows removed: {dupes_removed}")

# ── Step 2b: Drop rows with invalid recommendationid or timestamp ─
# Some rows come through with null or non-numeric values in
# recommendationid and timestamp_created. These are malformed
# records — possibly header rows repeated mid-file or API
# edge cases. Both fields are essential to the pipeline so
# any row missing either is dropped here.

print("Dropping rows with invalid recommendationid or timestamp_created...")

df_valid = df_deduped.filter(
    F.col("recommendationid").isNotNull() &
    F.col("timestamp_created").isNotNull() &
    F.col("recommendationid").rlike("^[0-9]+$") &
    F.col("timestamp_created").rlike("^[0-9]+$")
)

dropped_invalid = df_deduped.count() - df_valid.count()
print(f"  Invalid rows dropped: {dropped_invalid}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Filter to English reviews only

# COMMAND ----------

# Sentiment models perform best on English text. Non-English
# reviews are dropped here rather than carried forward.

print("Filtering to English reviews...")

df_english = df_valid.filter(F.col("language") == "english")

dropped_language = df_deduped.count() - df_english.count()
print(f"  Non-English rows dropped: {dropped_language}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Filter out reviews with no text

# COMMAND ----------

# Some Steam reviews contain only a thumbs up/down with no written
# text. These have no value for sentiment analysis or keyword
# extraction so they are removed here.

print("Filtering out empty reviews...")

df_has_text = df_english.filter(
    F.col("review").isNotNull() &
    (F.trim(F.col("review")) != "")
)

dropped_empty = df_english.count() - df_has_text.count()
print(f"  Empty review rows dropped: {dropped_empty}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Filter out low-quality reviews

# COMMAND ----------

# Reviews shorter than 10 characters (e.g. "ok", "bad", "lol")
# carry very little signal for sentiment or keyword analysis.
# Reviews from authors with 500+ reviews are likely bots or
# serial reviewers who may skew results.

print("Filtering out low-quality reviews...")

df_quality = df_has_text.filter(
    (F.length(F.col("review")) >= 10) &
    (F.col("author_num_reviews") < 500)
)

dropped_quality = df_has_text.count() - df_quality.count()
print(f"  Low-quality rows dropped: {dropped_quality}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Convert timestamps to readable dates

# COMMAND ----------

# Steam returns timestamps as Unix epoch integers (seconds since Jan 1 1970).
# We convert them to proper date columns so they can be used on the time-series trend chart in the dashboard.

df_dates = df_quality \
    .withColumn(
        "review_date",
        F.to_date(F.from_unixtime(F.col("timestamp_created")))
    ) \
    .withColumn(
        "review_updated_date",
        F.to_date(F.from_unixtime(F.col("timestamp_updated")))
    ) \
    .withColumn(
        "review_year",
        F.year(F.col("review_date"))
    ) \
    .withColumn(
        "review_month",
        F.month(F.col("review_date"))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Convert playtime and assign buckets

# COMMAND ----------

# playtime_at_review_mins captures how much the reviewer had played at the exact moment they wrote the review
# more precise than lifetime playtime for understanding sentiment by engagement level.

# Buckets reflect meaningful gamer engagement levels:
#   < 2h   → tried it briefly, likely a refund-window review
#   2–10h  → played enough to form an opinion
#   10–50h → engaged player
#   50h+   → hardcore / long-term player

df_playtime = df_dates \
    .withColumn(
        "playtime_at_review_hours",
        F.round(F.col("playtime_at_review_mins") / 60, 1)
    ) \
    .withColumn(
        "playtime_bucket",
        F.when(F.col("playtime_at_review_hours") < 2,  "< 2h")
         .when(F.col("playtime_at_review_hours") < 10, "2–10h")
         .when(F.col("playtime_at_review_hours") < 50, "10–50h")
         .otherwise("50h+")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Cast and clean new fields

# COMMAND ----------

# weighted_vote_score comes in as a string from the API
# (e.g. "0.823451") — cast it to float for use in Gold aggregations.
# steam_purchase, received_for_free, and written_during_early_access
# come in as strings ("True"/"False") from the CSV — cast to boolean.

print("Casting new fields...")

df_cast = df_playtime \
    .withColumn(
        "weighted_vote_score",
        F.col("weighted_vote_score").cast("float")
    ) \
    .withColumn(
        "steam_purchase",
        F.col("steam_purchase").cast("boolean")
    ) \
    .withColumn(
        "received_for_free",
        F.col("received_for_free").cast("boolean")
    ) \
    .withColumn(
        "written_during_early_access",
        F.col("written_during_early_access").cast("boolean")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Flag purchase type

# COMMAND ----------

# Combine steam_purchase and received_for_free into a single
# readable purchase_type label for easier dashboard filtering.
#   steam_purchase=True,  received_for_free=False → "Steam"
#   steam_purchase=False, received_for_free=False → "Key"
#   received_for_free=True                        → "Free copy"

df_purchase = df_cast.withColumn(
    "purchase_type",
    F.when(F.col("received_for_free") == True, "Free copy")
     .when(F.col("steam_purchase") == True,    "Steam")
     .otherwise("Key")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Compute review length

# COMMAND ----------

# Character count of the review text. Useful as a proxy for review depth
# Longer reviews tend to be more detailed and informative.
# Can be used alongside weighted_vote_score in Gold aggregations.

df_length = df_purchase.withColumn(
    "review_length",
    F.length(F.col("review"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Clean up columns

# COMMAND ----------

# Drop raw timestamp integers and raw minutes columns now that
# we have clean derived versions. Drop language since we have
# already filtered to English only. Drop steam_purchase and
# received_for_free since they are now captured in purchase_type.

df_silver = df_length.drop(
    "timestamp_created",
    "timestamp_updated",
    "playtime_forever_mins",
    "playtime_at_review_mins",
    "language",
    "steam_purchase",
    "received_for_free"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Preview and validate

# COMMAND ----------

print("\nSilver layer preview:")
display(df_silver.limit(5))

print(f"\nFinal row count:    {df_silver.count()}")
print(f"Final column count: {len(df_silver.columns)}")
print(f"\nSchema:")
df_silver.printSchema()

# Spot check — playtime bucket distribution
print("\nPlaytime bucket distribution:")
df_silver.groupBy("playtime_bucket") \
    .count() \
    .orderBy("playtime_bucket") \
    .show()

# Spot check — purchase type distribution
print("Purchase type distribution:")
df_silver.groupBy("purchase_type") \
    .count() \
    .orderBy("purchase_type") \
    .show()

# Spot check — early access split
print("Early access split:")
df_silver.groupBy("written_during_early_access") \
    .count() \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Write to Silver layer

# COMMAND ----------

# Write as Delta format
# overwrite mode is intentional
# cleanly from Bronze on each pipeline run.

print("Writing Silver layer...")

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .save("/Volumes/main/steam_analytics/raw_data/silver/steam_reviews")

print("Silver layer written successfully.")