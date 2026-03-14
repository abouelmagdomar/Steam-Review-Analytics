# Steam Review Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-Unity%20Catalog-red?logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange?logo=apachespark&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LR-blue?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-ff4b4b?logo=streamlit&logoColor=white)

An automated Steam review analytics pipeline that ingests thousands of player reviews via the Steam Web API, runs sentiment analysis using TF-IDF and Logistic Regression, and delivers an interactive dashboard of insights including sentiment trends over time, sentiment by player playtime, sentiment by purchase type, and top positive and negative keywords.

Demonstrated on **Assassin's Creed Shadows** (Ubisoft, 2025) with 8,528 reviews analysed.

**[View Live Dashboard →](https://your-username-steam-review-analytics.streamlit.app)**

---

## Overview

Game studios generate thousands of player reviews across their catalogues but manually reading and categorising them at scale is not feasible. This tool automates that process end to end — from raw API ingestion through to a client-ready analytics dashboard and can be repointed at any Steam title by changing a single URL.

The pipeline is built on the Databricks Medallion Architecture (Bronze → Silver → Gold), giving it the structure and governance expected in a production data engineering environment.

---

## Pipeline Architecture
```
Steam Web API
      ↓
Bronze Layer  — Raw review ingestion, cursor-based pagination, Delta write
      ↓
Silver Layer  — Deduplication, validation, timestamp conversion,
                playtime bucketing, purchase type classification
      ↓
Gold Layer    — TF-IDF vectorisation, Logistic Regression sentiment scoring,
                aggregations by time, playtime, purchase type, early access,
                keyword extraction from model coefficients
      ↓
Streamlit Dashboard — Interactive visualisation of Gold aggregations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Python, Requests, Steam Web API |
| Data processing | PySpark, Pandas, Databricks |
| Storage | Delta Lake, Unity Catalog |
| Sentiment model | scikit-learn — TF-IDF + Logistic Regression |
| Orchestration | Databricks Workflows |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

---

## Dashboard Panels

- **Headline metrics** — total reviews, positive/negative split, average sentiment score, Steam verdict
- **Sentiment trend over time** — monthly average signed sentiment score across the game's review history
- **Sentiment by playtime** — how sentiment differs across player engagement levels (< 2h, 2–10h, 10–50h, 50h+)
- **Sentiment by purchase type** — Steam purchase vs key activation vs free copy
- **Early access vs full release** — sentiment split across release stages
- **Top keywords** — most distinctive positive and negative keywords learned by the model

---

## Sentiment Model

Reviews are scored using a scikit-learn pipeline of TF-IDF vectorisation followed by Logistic Regression trained on Steam's native `voted_up` field as the ground truth label. This produces a continuous sentiment score between 0 and 1 for each review rather than a binary positive/negative label, enabling more granular aggregations across the dashboard panels.

The model is trained fresh on each pipeline run using that game's own review data, meaning the learned vocabulary and keyword weights are specific to the title being analysed.

**Model performance on Assassin's Creed Shadows:**
- Test accuracy: 88.4%
- Training samples: 8,528 reviews
- Negative precision: 0.69 · Negative recall: 0.84
- Positive precision: 0.95 · Positive recall: 0.90
- The more balanced sentiment distribution of this title (compared to universally praised games) allows the model to learn both classes effectively, reflected in strong performance across both positive and negative classes.

---

## Repository Structure
```
steam-review-analytics/
├── notebooks/
│   ├── 01_bronze_ingestion.py       # Steam API ingestion
│   ├── 02_silver_transformations.py # Cleaning and enrichment
│   └── 03_gold_sentiment.py         # Sentiment scoring and aggregations
├── data/
│   ├── game_summary.csv
│   ├── sentiment_trend.csv
│   ├── sentiment_by_playtime.csv
│   ├── sentiment_by_purchase_type.csv
│   ├── sentiment_by_early_access.csv
│   └── keywords.csv
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## Running the Pipeline Yourself

### Prerequisites

- Databricks workspace (free tier works for experimentation)
- Unity Catalog enabled
- Python 3.11+

### Setup

**1 — Clone the repository**
```bash
git clone https://github.com/your-username/steam-review-analytics.git
cd steam-review-analytics
```

**2 — Upload the notebooks to Databricks**

In the Databricks workspace go to **Workspace → Import** and upload the three notebooks from the `notebooks/` folder in order.

**3 — Update the Unity Catalog path**

In each notebook replace the following path with your own catalog and schema:
```python
"/Volumes/main/steam_analytics/raw_data/"
```

**4 — Run the pipeline**

Either run each notebook manually in order or set up a Databricks Workflow chaining all three. When prompted enter any Steam store URL:
```
https://store.steampowered.com/app/2933620/Assassins_Creed_Shadows/
```

**5 — Run the dashboard locally**
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Key Findings — Assassin's Creed Shadows

- **Overall sentiment: +0.32** — mildly positive overall across 8,528 reviews, indicating a divided player base
- **Sentiment declined sharply post-launch:** Reviews peaked at 3,370 positive counts in March 2025 before dropping significantly through April–June, suggesting early enthusiasm faded as players spent more time with the game
- **Early drop-off signal:** Players with under 2 hours playtime score the game negatively (-0.14) — the only segment to fall below neutral — while players who pushed past 2 hours rate it positively, suggesting the opening hours are a retention risk
- **Long-term players most critical:** The 50h+ bucket scores lowest among engaged players (+0.33) compared to 10–50h (+0.37) and 2–10h (+0.38), suggesting issues that compound over extended play sessions
- **Purchase channel insight:** Free copy reviewers rate the game highest (+0.38) while Steam purchasers rate it lowest (+0.32) — players who paid full price hold the most critical view
- **Model performance:** With a more balanced sentiment distribution than universally praised titles, the model performs strongly on both classes — 88.4% accuracy, 0.84 negative recall, 0.90 positive recall

---

## Notes

This project was built as a portfolio demonstration of data engineering and NLP techniques on the Databricks platform. The pipeline is designed to be reusable across any Steam title swap the URL and re-run.