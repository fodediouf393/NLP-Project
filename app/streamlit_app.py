from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from asrs_sum.pipeline.predict import build_summarizer
from asrs_sum.utils.io import load_yaml, read_csv


TEST_DATA_PATH = ROOT / "data" / "raw" / "test.csv"
METRICS_TOPK_PATH = ROOT / "outputs" / "reports" / "test_metrics_topk.json"
METRICS_TEXTRANK_PATH = ROOT / "outputs" / "reports" / "test_metrics_textrank.json"
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main { background-color: #f6f8fb; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px; }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            padding: 1.5rem 1.75rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
        }
        .hero-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.35rem; }
        .hero-subtitle { font-size: 1rem; opacity: 0.92; }
        .section-card {
            background: white;
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            border: 1px solid #e5e7eb;
        }
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        .metric-label { color: #475569; font-size: 0.9rem; margin-bottom: 0.25rem; }
        .metric-value { color: #0f172a; font-size: 1.55rem; font-weight: 700; }
        .tag {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: #dbeafe;
            color: #1d4ed8;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .ref-box {
            background: #f8fafc;
            border-left: 5px solid #0f766e;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            margin-top: 0.4rem;
            margin-bottom: 0.5rem;
        }
        .pred-box {
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            margin-top: 0.4rem;
            margin-bottom: 0.5rem;
        }
        .pred-box-2 {
            background: #f5f3ff;
            border-left: 5px solid #7c3aed;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            margin-top: 0.4rem;
            margin-bottom: 0.5rem;
        }
        .small-note { color: #64748b; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_topk_summarizer():
    return build_summarizer(str(CONFIG_PATH), "topk")


@st.cache_resource
def load_textrank_summarizer():
    return build_summarizer(str(CONFIG_PATH), "textrank")


@st.cache_data
def load_test_dataframe() -> pd.DataFrame:
    if not TEST_DATA_PATH.exists():
        return pd.DataFrame()

    df = read_csv(TEST_DATA_PATH)

    rename_map = {}
    if "Report 1_Narrative" in df.columns and "narrative" not in df.columns:
        rename_map["Report 1_Narrative"] = "narrative"
    if "Report 1.2_Synopsis" in df.columns and "synopsis" not in df.columns:
        rename_map["Report 1.2_Synopsis"] = "synopsis"

    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ["narrative", "synopsis"]:
        if col not in df.columns:
            df[col] = ""

    df["narrative"] = df["narrative"].fillna("").astype(str)
    df["synopsis"] = df["synopsis"].fillna("").astype(str)
    df = df[(df["narrative"].str.strip() != "") & (df["synopsis"].str.strip() != "")].copy()
    df["narrative_word_count"] = df["narrative"].str.split().str.len()
    df["synopsis_word_count"] = df["synopsis"].str.split().str.len()
    return df


@st.cache_data
def load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def choose_demo_examples(df: pd.DataFrame, n: int = 3, seed: int = 42) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df[
        (df["narrative_word_count"] >= 40)
        & (df["synopsis_word_count"] >= 8)
        & (df["synopsis_word_count"] <= 60)
    ].copy()

    if len(filtered) < n:
        filtered = df.copy()

    random.seed(seed)
    indices = random.sample(list(filtered.index), k=min(n, len(filtered)))
    return filtered.loc[indices].copy()


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ASRS Extractive Summarization</div>
            <div class="hero-subtitle">
                Comparative extractive baselines: Top-k sentence selection and TextRank
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <span class="tag">NLP</span>
            <span class="tag">Extractive Summarization</span>
            <span class="tag">ASRS Aviation Reports</span>
            <span class="tag">Top-k</span>
            <span class="tag">TextRank</span>
            <span class="tag">CPU-ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(df: pd.DataFrame, metrics_topk: dict, metrics_textrank: dict) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Test examples", str(len(df)) if not df.empty else "N/A")
    with col2:
        render_metric_card("Methods", "2")
    with col3:
        render_metric_card("Deployment", "Streamlit")

    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0;">Project overview</h4>
            <p>
                This application compares two extractive summarization approaches on real ASRS aviation incident reports:
                a top-k sentence selection baseline and a TextRank graph-based baseline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics_topk and metrics_textrank:
        row = st.columns(2)
        with row[0]:
            st.markdown("### Top-k")
            render_metric_card("ROUGE-1", f"{metrics_topk.get('rouge', {}).get('rouge1_f1', 0):.3f}")
            render_metric_card("ROUGE-2", f"{metrics_topk.get('rouge', {}).get('rouge2_f1', 0):.3f}")
            render_metric_card("ROUGE-L", f"{metrics_topk.get('rouge', {}).get('rougeL_f1', 0):.3f}")
        with row[1]:
            st.markdown("### TextRank")
            render_metric_card("ROUGE-1", f"{metrics_textrank.get('rouge', {}).get('rouge1_f1', 0):.3f}")
            render_metric_card("ROUGE-2", f"{metrics_textrank.get('rouge', {}).get('rouge2_f1', 0):.3f}")
            render_metric_card("ROUGE-L", f"{metrics_textrank.get('rouge', {}).get('rougeL_f1', 0):.3f}")


def render_examples_tab(df: pd.DataFrame, topk_summarizer, textrank_summarizer) -> None:
    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0;">Real examples from the dataset</h4>
            <p class="small-note">Reference synopsis vs Top-k vs TextRank</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning("The test dataset was not found. Please make sure data/raw/test.csv exists.")
        return

    if "demo_seed" not in st.session_state:
        st.session_state["demo_seed"] = 42

    if st.button("Load 3 other examples"):
        st.session_state["demo_seed"] += 1

    examples = choose_demo_examples(df, n=3, seed=st.session_state["demo_seed"])

    for i, (_, row) in enumerate(examples.iterrows(), start=1):
        narrative = row["narrative"]
        synopsis = row["synopsis"]
        topk_pred = topk_summarizer.summarize(narrative).summary
        textrank_pred = textrank_summarizer.summarize(narrative).summary

        st.markdown(f"## Example {i}")

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Source words", str(len(narrative.split())))
        with c2:
            render_metric_card("Reference words", str(len(synopsis.split())))
        with c3:
            render_metric_card("Top-k / TextRank words", f"{len(topk_pred.split())} / {len(textrank_pred.split())}")

        with st.expander("Narrative", expanded=False):
            st.write(narrative)

        st.markdown(
            f"""
            <div class="ref-box">
                <strong>Reference synopsis</strong><br><br>
                {synopsis}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(
                f"""
                <div class="pred-box">
                    <strong>Top-k predicted summary</strong><br><br>
                    {topk_pred}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_right:
            st.markdown(
                f"""
                <div class="pred-box-2">
                    <strong>TextRank predicted summary</strong><br><br>
                    {textrank_pred}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_demo_tab(topk_summarizer, textrank_summarizer) -> None:
    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0;">Interactive summarization demo</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    method = st.selectbox("Summarization method", ["topk", "textrank"])
    summarizer = topk_summarizer if method == "topk" else textrank_summarizer

    text = st.text_area(
        "Narrative input",
        height=280,
        placeholder="Paste an ASRS narrative here...",
    )

    if st.button("Generate Summary", type="primary"):
        start = time.perf_counter()
        result = summarizer.summarize(text)
        elapsed = time.perf_counter() - start

        st.markdown(
            f"""
            <div class="pred-box">
                <strong>Generated summary ({method})</strong><br><br>
                {result.summary if result.summary else "No summary generated."}
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("Source sentences", str(result.original_sentence_count))
        with c2:
            render_metric_card("Summary words", str(result.summary_word_count))
        with c3:
            render_metric_card("Latency (s)", f"{elapsed:.4f}")

        st.markdown("### Selected sentences")
        if result.selected_sentences:
            for idx, sentence in enumerate(result.selected_sentences, start=1):
                st.markdown(f"**{idx}.** {sentence}")
        else:
            st.info("No sentence selected.")

        st.markdown("### Sentence ranking")
        if result.ranked_sentences:
            rank_df = pd.DataFrame(result.ranked_sentences)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
        else:
            st.info("No ranking details available.")


def render_single_metrics_block(title: str, metrics: dict) -> None:
    st.markdown(f"### {title}")
    if not metrics:
        st.warning(f"No metrics found for {title}.")
        return

    rouge = metrics.get("rouge", {})
    entity = metrics.get("entity_coverage", {})

    row1 = st.columns(3)
    with row1[0]:
        render_metric_card("ROUGE-1", f"{rouge.get('rouge1_f1', 0):.3f}")
    with row1[1]:
        render_metric_card("ROUGE-2", f"{rouge.get('rouge2_f1', 0):.3f}")
    with row1[2]:
        render_metric_card("ROUGE-L", f"{rouge.get('rougeL_f1', 0):.3f}")

    row2 = st.columns(3)
    with row2[0]:
        render_metric_card("Runway coverage", f"{entity.get('runway_coverage', 0):.3f}")
    with row2[1]:
        render_metric_card("Altitude coverage", f"{entity.get('altitude_coverage', 0):.3f}")
    with row2[2]:
        render_metric_card("Airport code coverage", f"{entity.get('airport_code_coverage', 0):.3f}")

    row3 = st.columns(2)
    with row3[0]:
        render_metric_card("Compression ratio mean", f"{metrics.get('compression_ratio_mean', 0):.3f}")
    with row3[1]:
        render_metric_card("Latency mean (s)", f"{metrics.get('latency_seconds_mean', 0):.4f}")


def render_metrics_tab(metrics_topk: dict, metrics_textrank: dict) -> None:
    st.markdown(
        """
        <div class="section-card">
            <h4 style="margin-top:0;">Evaluation on the test set</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        render_single_metrics_block("Top-k", metrics_topk)
    with right:
        render_single_metrics_block("TextRank", metrics_textrank)


def main() -> None:
    st.set_page_config(
        page_title="ASRS Extractive Summarizer",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    topk_summarizer = load_topk_summarizer()
    textrank_summarizer = load_textrank_summarizer()
    df = load_test_dataframe()
    metrics_topk = load_json_file(METRICS_TOPK_PATH)
    metrics_textrank = load_json_file(METRICS_TEXTRANK_PATH)

    render_header()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Real Dataset Examples",
            "Interactive Demo",
            "Evaluation Metrics",
        ]
    )

    with tab1:
        render_overview_tab(df, metrics_topk, metrics_textrank)

    with tab2:
        render_examples_tab(df, topk_summarizer, textrank_summarizer)

    with tab3:
        render_demo_tab(topk_summarizer, textrank_summarizer)

    with tab4:
        render_metrics_tab(metrics_topk, metrics_textrank)


if __name__ == "__main__":
    main()