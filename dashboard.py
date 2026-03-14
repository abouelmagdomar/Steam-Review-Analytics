# dashboard.py — Steam Review Analytics Dashboard
# Live demo: https://your-username-steam-review-analytics.streamlit.app
# GitHub: https://github.com/your-username/steam-review-analytics

import streamlit as st
import plotly.express as px
import pandas as pd


# ── Page config ─────────────────────────────────────────────────

st.set_page_config(
    page_title="Steam Review Analytics",
    page_icon="🎮",
    layout="wide"
)


# ── Load data from CSV files ─────────────────────────────────────
@st.cache_data
def load_data():
    base = "data"
    return {
        "summary":  pd.read_csv(f"{base}/game_summary.csv"),
        "trend":    pd.read_csv(f"{base}/sentiment_trend.csv"),
        "playtime": pd.read_csv(f"{base}/sentiment_by_playtime.csv"),
        "purchase": pd.read_csv(f"{base}/sentiment_by_purchase_type.csv"),
        "early":    pd.read_csv(f"{base}/sentiment_by_early_access.csv"),
        "keywords": pd.read_csv(f"{base}/keywords.csv"),
    }

data        = load_data()
df_summary  = data["summary"]
df_trend    = data["trend"]
df_playtime = data["playtime"]
df_purchase = data["purchase"]
df_early    = data["early"]
df_keywords = data["keywords"]


# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.title("🎮 Steam Analytics")
st.sidebar.markdown("---")

games         = df_summary["game_name"].unique().tolist()
selected_game = st.sidebar.selectbox("Select a game", games)

st.sidebar.markdown("---")
st.sidebar.caption("Pipeline built with Databricks · scikit-learn · Streamlit")

# Filter every DataFrame down to the selected game
summary  = df_summary[df_summary["game_name"]  == selected_game].iloc[0]
trend    = df_trend[df_trend["game_name"]       == selected_game]
playtime = df_playtime[df_playtime["game_name"] == selected_game].copy()
purchase = df_purchase[df_purchase["game_name"] == selected_game]
early    = df_early[df_early["game_name"]       == selected_game]
keywords = df_keywords[df_keywords["game_name"] == selected_game]


# ── Page header ──────────────────────────────────────────────────
st.title(f"Steam Review Analytics — {selected_game}")
st.caption(
    f"Analysed {int(summary['total_reviews']):,} total reviews · "
    f"Steam verdict: **{summary['review_score_desc']}**"
)
st.markdown("---")


# ── Row 1: Headline metric cards ────────────────────────────────

total_reviews  = int(summary["total_reviews"])
total_positive = int(summary["total_positive"])
total_negative = int(summary["total_negative"])
positive_rate  = total_positive / total_reviews * 100 if total_reviews > 0 else 0
avg_sentiment  = float(trend["avg_sentiment"].mean())

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Reviews",
        value=f"{total_reviews:,}"
    )
with col2:
    st.metric(
        label="Positive Reviews",
        value=f"{total_positive:,}",
        delta=f"{positive_rate:.1f}% positive"
    )
with col3:
    st.metric(
        label="Negative Reviews",
        value=f"{total_negative:,}",
        delta=f"{100 - positive_rate:.1f}% negative",
        delta_color="inverse"
    )
with col4:
    st.metric(
        label="Avg Sentiment Score",
        value=f"{avg_sentiment:+.2f}",
        delta="signed score  −1 to +1"
    )

st.markdown("---")


# ── Row 2: Sentiment trend + Early access ───────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Sentiment trend over time")

    fig_trend = px.line(
        trend,
        x="review_period",
        y="avg_sentiment",
        markers=True,
        hover_data=["review_count"],
        labels={
            "review_period": "Month",
            "avg_sentiment": "Avg sentiment",
            "review_count":  "Review count"
        },
        color_discrete_sequence=["#7F77DD"]
    )

    fig_trend.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Neutral",
        annotation_position="bottom right"
    )
    fig_trend.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[-1, 1])
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.subheader("Early access vs full release")

    fig_early = px.bar(
        early,
        x="release_stage",
        y="avg_sentiment",
        color="release_stage",
        text=early["avg_sentiment"].apply(lambda x: f"{x:+.2f}"),
        labels={
            "release_stage": "Release stage",
            "avg_sentiment": "Avg sentiment"
        },
        color_discrete_map={
            "Early Access": "#FAC775",
            "Full Release":  "#5DCAA5"
        }
    )
    fig_early.update_traces(textposition="outside")
    fig_early.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[-1, 1])
    )
    st.plotly_chart(fig_early, use_container_width=True)

st.markdown("---")


# ── Row 3: Playtime + Purchase type ─────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment by playtime")

    bucket_order = ["< 2h", "2–10h", "10–50h", "50h+"]
    playtime["playtime_bucket"] = pd.Categorical(
        playtime["playtime_bucket"],
        categories=bucket_order,
        ordered=True
    )
    playtime = playtime.sort_values("playtime_bucket")

    fig_playtime = px.bar(
        playtime,
        x="playtime_bucket",
        y="avg_sentiment",
        color="avg_sentiment",
        text=playtime["avg_sentiment"].apply(lambda x: f"{x:+.2f}"),
        hover_data=["review_count"],
        labels={
            "playtime_bucket": "Playtime at review",
            "avg_sentiment":   "Avg sentiment",
            "review_count":    "Review count"
        },
        color_continuous_scale=["#E24B4A", "#F09595", "#9FE1CB", "#1D9E75"],
        range_color=[-1, 1]
    )
    fig_playtime.update_traces(textposition="outside")
    fig_playtime.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[-1, 1])
    )
    st.plotly_chart(fig_playtime, use_container_width=True)

with col_right:
    st.subheader("Sentiment by purchase type")

    fig_purchase = px.bar(
        purchase,
        x="purchase_type",
        y="avg_sentiment",
        color="purchase_type",
        text=purchase["avg_sentiment"].apply(lambda x: f"{x:+.2f}"),
        hover_data=["review_count"],
        labels={
            "purchase_type": "Purchase type",
            "avg_sentiment": "Avg sentiment",
            "review_count":  "Review count"
        },
        color_discrete_map={
            "Steam":     "#7F77DD",
            "Key":       "#378ADD",
            "Free copy": "#FAC775"
        }
    )
    fig_purchase.update_traces(textposition="outside")
    fig_purchase.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[-1, 1])
    )
    st.plotly_chart(fig_purchase, use_container_width=True)

st.markdown("---")


# ── Row 4: Top keywords ──────────────────────────────────────────

st.subheader("Top keywords")

col_pos, col_neg = st.columns(2)

keywords_pos = keywords[keywords["sentiment"] == "POSITIVE"] \
    .sort_values("importance_score", ascending=True).tail(15)

keywords_neg = keywords[keywords["sentiment"] == "NEGATIVE"] \
    .sort_values("importance_score", ascending=True).tail(15)

with col_pos:
    st.markdown("##### 👍 Top positive keywords")
    fig_pos = px.bar(
        keywords_pos,
        x="importance_score",
        y="word",
        orientation="h",
        labels={
            "importance_score": "Importance",
            "word":             "Keyword"
        },
        color_discrete_sequence=["#1D9E75"]
    )
    fig_pos.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_pos, use_container_width=True)

with col_neg:
    st.markdown("##### 👎 Top negative keywords")
    fig_neg = px.bar(
        keywords_neg,
        x="importance_score",
        y="word",
        orientation="h",
        labels={
            "importance_score": "Importance",
            "word":             "Keyword"
        },
        color_discrete_sequence=["#E24B4A"]
    )
    fig_neg.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_neg, use_container_width=True)