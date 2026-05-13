"""Streamlit dashboard for Reddit Idea Miner.

Run with: `streamlit run app.py`
Read-only — never writes to the DB.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from reddit_miner import dashboard
from reddit_miner.config import get_settings

SETTINGS = get_settings()
DB_PATH = str(SETTINGS.db_path)

get_cluster_runs = st.cache_data(ttl=60)(dashboard.cluster_runs)
get_clusters = st.cache_data(ttl=60)(dashboard.clusters_for_run)
get_docs = st.cache_data(ttl=60)(dashboard.docs_for_cluster)
get_mentions = st.cache_data(ttl=60)(dashboard.mentions_for_run)
get_alerts = st.cache_data(ttl=60)(dashboard.recent_alerts)


def main() -> None:
    st.set_page_config(page_title="Reddit Idea Miner", layout="wide")
    st.title("Reddit Idea Miner")
    st.caption("Unmet-demand signals from vertical subreddits.")

    if not Path(DB_PATH).exists():
        st.warning(f"No DB at `{DB_PATH}`. Run `python -m reddit_miner init-db` first.")
        st.stop()

    runs = get_cluster_runs(DB_PATH)
    if runs.empty:
        st.info(
            "No cluster runs yet. Seed subs with `add-sub` then run "
            "`python -m reddit_miner pipeline`."
        )
        st.stop()

    with st.sidebar:
        st.header("Run")
        runs_display = runs.assign(
            run_at_iso=runs["run_at"].apply(dashboard.fmt_ts),
            label=lambda df: df.apply(
                lambda r: f"#{r['id']} — {r['run_at_iso']} ({r['n_clusters']} clusters)",
                axis=1,
            ),
        )
        choice = st.selectbox("Cluster run", runs_display["label"].tolist())
        selected_run = int(runs_display.loc[runs_display["label"] == choice, "id"].iloc[0])
        include_outliers = st.checkbox("Include HDBSCAN outliers", value=False)

    tab_clusters, tab_mentions, tab_alerts = st.tabs(["Clusters", "Competitor mentions", "Alerts"])

    with tab_clusters:
        clusters = get_clusters(DB_PATH, selected_run)
        if clusters.empty:
            st.info("No clusters in this run.")
        else:
            st.subheader("All clusters in this run")
            st.dataframe(
                clusters.style.format({"avg_pain_score": "{:.2f}", "signal_score": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Cluster details")
            cluster_choice = st.selectbox(
                "Inspect cluster",
                options=clusters["id"].tolist(),
                format_func=lambda cid: (
                    f"#{cid} — {clusters.loc[clusters['id'] == cid, 'label_tfidf'].iloc[0]}"
                ),
            )
            docs = get_docs(DB_PATH, int(cluster_choice), include_outliers)
            st.write(f"**{len(docs)} top documents** (by pain score)")
            st.dataframe(docs, use_container_width=True, hide_index=True)

    with tab_mentions:
        mentions = get_mentions(DB_PATH, selected_run)
        if mentions.empty:
            st.info("No competitor mentions for clusters in this run.")
        else:
            sentiment_filter = st.multiselect(
                "Sentiment",
                options=["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"],
            )
            filtered = mentions[mentions["sentiment"].isin(sentiment_filter)]
            st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_alerts:
        alerts = get_alerts(DB_PATH)
        if alerts.empty:
            st.info("No alerts fired yet.")
        else:
            alerts = alerts.assign(triggered_at=alerts["triggered_at"].apply(dashboard.fmt_ts))
            st.dataframe(alerts, use_container_width=True, hide_index=True)


main()
