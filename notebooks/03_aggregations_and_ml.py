# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3 — Gold Layer: Sentiment Scoring & Aggregations

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load the Silver table

# COMMAND ----------

df_silver = spark.read.format("delta").load(
    "/Volumes/main/steam_analytics/raw_data/silver/steam_reviews"
)

print(f"  Rows loaded: {df_silver.count()}")

# Convert to Pandas — scikit-learn runs on the driver node and works with Pandas DataFrames, not Spark DataFrames.
df = df_silver.toPandas()

print(f"  Converted to Pandas: {df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Prepare features and labels

# COMMAND ----------

# X is the review text — our input feature.
# y is voted_up cast to integer (True → 1, False → 0) — our label.
# We drop any rows where either is null to avoid training errors.

print("Preparing features and labels...")

df_model = df[["review", "voted_up"]].dropna()
df_model["voted_up"] = df_model["voted_up"].map(
    {"True": 1, "False": 0, True: 1, False: 0}
).astype(int)

X = df_model["review"]
y = df_model["voted_up"]

print(f"  Training samples: {len(X):,}")
print(f"  Positive reviews: {y.sum():,} ({y.mean():.1%})")
print(f"  Negative reviews: {(1 - y).sum():,} ({(1 - y.mean()):.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train / test split

# COMMAND ----------

# We split the data 80/20 — 80% to train the model, 20% to test it on reviews it has never seen.
# random_state=42 ensures the split is reproducible every run.

print("Splitting into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Train size: {len(X_train):,}")
print(f"  Test size:  {len(X_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Build the model pipeline

# COMMAND ----------

print("Building model pipeline...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Train the model

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Evaluate on test set

# COMMAND ----------

# predict_proba() gives us the continuous score, the probability that a review is positive, ranging from 0.0 to 1.0.
# We report both accuracy and a full classification report showing precision, recall, and F1 score for each class.

print("Evaluating model on test set...")

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy     = accuracy_score(y_test, y_pred)

print(f"\n  Test accuracy: {accuracy:.1%}")
print(f"\n  Classification report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Score all reviews 

# COMMAND ----------

# sentiment_score  → probability of being positive (0.0 to 1.0)
# sentiment_label  → "POSITIVE" if score >= 0.5, else "NEGATIVE"
# sentiment_signed → rescales 0–1 to -1–+1 so dashboard averages
#                    are centred on zero and intuitively readable:
#                    +1.0 = strongly positive
#                     0.0 = neutral
#                    -1.0 = strongly negative

print("Scoring all reviews...")

all_scores = model.predict_proba(df_model["review"])[:, 1]

df_model = df_model.copy()
df_model["sentiment_score"]  = all_scores
df_model["sentiment_label"]  = np.where(all_scores >= 0.5, "POSITIVE", "NEGATIVE")
df_model["sentiment_signed"] = (all_scores * 2) - 1

# Merge sentiment scores back onto the full Silver DataFrame
df_scored = df.join(
    df_model[["sentiment_score", "sentiment_label", "sentiment_signed"]],
    how="left"
)

print(f"  Scored reviews: {df_scored['sentiment_score'].notna().sum():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Sentiment trend over time (by month)

# COMMAND ----------

print("Aggregating sentiment trend over time...")

df_scored["voted_up_int"] = df_scored["voted_up"].map(
    {"True": 1, "False": 0, True: 1, False: 0}
).astype(int)

df_trend = df_scored.groupby(
    ["game_name", "review_year", "review_month"],
    as_index=False
).agg(
    avg_sentiment  = ("sentiment_signed", "mean"),
    review_count   = ("sentiment_signed", "count"),
    positive_count = ("voted_up_int",     "sum")
)

df_trend["review_period"] = (
    df_trend["review_year"].astype(str) + "-" +
    df_trend["review_month"].astype(str).str.zfill(2)
)

df_trend = df_trend.sort_values(["review_year", "review_month"])
print(f"  Trend periods: {len(df_trend)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Sentiment by playtime bucket

# COMMAND ----------

print("Aggregating sentiment by playtime bucket...")

df_playtime = df_scored.groupby(
    ["game_name", "playtime_bucket"],
    as_index=False
).agg(
    avg_sentiment = ("sentiment_signed", "mean"),
    review_count  = ("sentiment_signed", "count")
)

print(f"  Playtime buckets: {len(df_playtime)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Sentiment by purchase type

# COMMAND ----------

print("Aggregating sentiment by purchase type...")

df_purchase = df_scored.groupby(
    ["game_name", "purchase_type"],
    as_index=False
).agg(
    avg_sentiment = ("sentiment_signed", "mean"),
    review_count  = ("sentiment_signed", "count")
)

print(f"  Purchase types: {len(df_purchase)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Sentiment by early access vs full release

# COMMAND ----------

print("Aggregating sentiment by early access flag...")

df_early = df_scored.groupby(
    ["game_name", "written_during_early_access"],
    as_index=False
).agg(
    avg_sentiment = ("sentiment_signed", "mean"),
    review_count  = ("sentiment_signed", "count")
)

df_early["release_stage"] = df_early["written_during_early_access"].map(
    {True: "Early Access", False: "Full Release"}
)

df_early = df_early.drop(columns=["written_during_early_access"])
print(f"  Release stages: {len(df_early)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: TF-IDF keyword extraction

# COMMAND ----------

# Extracts the most distinctive words for positive and negative reviews using the feature weights learned by the model.
# Words with the highest positive coefficients are the strongest predictors of a positive review, and vice versa.

print("Extracting keywords from model weights...")

feature_names = model.named_steps["tfidf"].get_feature_names_out()
coefficients  = model.named_steps["classifier"].coef_[0]

df_weights = pd.DataFrame({
    "word":        feature_names,
    "coefficient": coefficients
})

df_keywords_pos = df_weights.nlargest(20, "coefficient").copy()
df_keywords_pos["sentiment"] = "POSITIVE"
df_keywords_pos["game_name"] = df_scored["game_name"].iloc[0]

df_keywords_neg = df_weights.nsmallest(20, "coefficient").copy()
df_keywords_neg["sentiment"] = "NEGATIVE"
df_keywords_neg["game_name"] = df_scored["game_name"].iloc[0]
df_keywords_neg["coefficient"] = df_keywords_neg["coefficient"].abs()

df_keywords_pd = pd.concat(
    [df_keywords_pos, df_keywords_neg],
    ignore_index=True
).rename(columns={"coefficient": "importance_score"})

print(f"\n  Top 5 positive keywords: {df_keywords_pos['word'].head().tolist()}")
print(f"  Top 5 negative keywords: {df_keywords_neg['word'].head().tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Preview all aggregations

# COMMAND ----------

print("\nSentiment trend preview:")
print(df_trend.head())

print("\nSentiment by playtime bucket:")
print(df_playtime)

print("\nSentiment by purchase type:")
print(df_purchase)

print("\nSentiment by early access vs full release:")
print(df_early)

print(f"\nModel test accuracy: {accuracy:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14: Convert back to Spark and write Gold tables

# COMMAND ----------

# ── Step 14: Convert back to Spark and write Gold tables ─────────

print("\nWriting Gold tables...")

BASE_PATH = "/Volumes/main/steam_analytics/raw_data/gold"

spark.createDataFrame(df_scored).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BASE_PATH}/reviews_scored")

spark.createDataFrame(df_trend).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BASE_PATH}/sentiment_trend")

spark.createDataFrame(df_playtime).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BASE_PATH}/sentiment_by_playtime")

spark.createDataFrame(df_purchase).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BASE_PATH}/sentiment_by_purchase_type")

spark.createDataFrame(df_early).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BASE_PATH}/sentiment_by_early_access")

spark.createDataFrame(df_keywords_pd).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BASE_PATH}/keywords")

print("All Gold tables written successfully.")
print(f"\nGold tables written to {BASE_PATH}:")

# Note: This specific game has a heavily positive review distribution (97.6% positive).
# The model performs well on the majority class but negative precision is limited
# by the small number of negative reviews available for training.
# This is expected and reflects genuine player sentiment rather than a model flaw.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 15: Export aggregations as CSVs for dashboard

# COMMAND ----------

import os

EXPORT_PATH = "/Volumes/main/steam_analytics/raw_data/dashboard_data"
os.makedirs(EXPORT_PATH, exist_ok=True)

df_trend.to_csv(        f"{EXPORT_PATH}/sentiment_trend.csv",           index=False)
df_playtime.to_csv(     f"{EXPORT_PATH}/sentiment_by_playtime.csv",     index=False)
df_purchase.to_csv(     f"{EXPORT_PATH}/sentiment_by_purchase_type.csv",index=False)
df_early.to_csv(        f"{EXPORT_PATH}/sentiment_by_early_access.csv", index=False)
df_keywords_pd.to_csv(  f"{EXPORT_PATH}/keywords.csv",                  index=False)

# Export game summary from Bronze
df_summary_export = pd.read_csv(
    "/Volumes/main/steam_analytics/raw_data/bronze/game_summary.csv"
)
df_summary_export.to_csv(f"{EXPORT_PATH}/game_summary.csv", index=False)

print("Dashboard CSVs exported successfully.")
print(f"Exported to: {EXPORT_PATH}")