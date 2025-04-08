import streamlit as st
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from backend import (
    scrape_reviews,
    load_text,
    split_text,
    initialize_embeddings,
    retrieve_reviews,
    analyze_feedback,
    preloaded_chunks,
    preloaded_embedding_model,
    preloaded_embeddings_list
)
from config import BASE_URL

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Qatar Airways Feedback Analysis", layout="wide")
st.title("✈️ Qatar Airways Customer Feedback Analysis")

# ------------------ Sidebar: Mode Selection ------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio("Choose data mode:", ["Use Preloaded (5 pages)", "Scrape Custom Pages"])

    if mode == "Scrape Custom Pages":
        num_pages = st.slider("Number of Pages to Scrape", 1, 10, 5)
        if st.button("🔄 Scrape & Process Reviews"):
            with st.spinner("Scraping customer reviews..."):
                scrape_reviews(BASE_URL, num_pages)
            st.success("✅ Reviews scraped successfully!")

            with st.spinner("Generating embeddings..."):
                reviews_text = load_text()
                chunks = split_text(reviews_text)
                embedding_model, embeddings_list = initialize_embeddings(chunks)

            st.session_state["chunks"] = chunks
            st.session_state["embedding_model"] = embedding_model
            st.session_state["embeddings_list"] = embeddings_list
    else:
        st.info("Using preloaded reviews and embeddings from 5 pages.")

# ------------------ Query Input Section ------------------
st.markdown("### 🔍 Search Customer Feedback")
st.markdown("_Ask a question based on real customer reviews (e.g., **'What did people say about baggage handling?'**)_")

question = st.text_input("✏️ Enter your query:")

# ------------------ Analyze Feedback Button ------------------
if st.button("🚀 Analyze Feedback"):

    # --- Case 1: Scrape Custom Pages ---
    if mode == "Scrape Custom Pages":
        if "chunks" not in st.session_state:
            st.warning("⚠️ Please scrape and process reviews first.")
        else:
            with st.spinner("🔍 Retrieving the most relevant reviews..."):
                relevant_reviews, distances = retrieve_reviews(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embedding_model"],
                    st.session_state["embeddings_list"],
                    return_scores=True
                )
            with st.spinner("🧠 Analyzing feedback using AI..."):
                response = analyze_feedback(question, relevant_reviews)

    # --- Case 2: Preloaded Embeddings ---
    else:
        with st.spinner("🔍 Retrieving the most relevant reviews..."):
            relevant_reviews, distances = retrieve_reviews(
                question,
                preloaded_chunks,
                preloaded_embedding_model,
                preloaded_embeddings_list,
                return_scores=True
            )
        with st.spinner("🧠 Analyzing feedback using AI..."):
            response = analyze_feedback(question, relevant_reviews)

    # ------------------ Display Response ------------------
    st.markdown("---")
    st.markdown("### 📌 AI-Generated Insights")
    st.markdown(f"#### 💬 Response to: _{question}_")
    st.write(response)

    # ------------------ ROUGE Score ------------------
    st.markdown("### 📊 ROUGE Score Evaluation")
    reference_text = " ".join(relevant_reviews)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, response)
    for metric, result in scores.items():
        st.markdown(f"**{metric.upper()}**: Precision = `{result.precision:.2f}`, Recall = `{result.recall:.2f}`, F1 = `{result.fmeasure:.2f}`")

    # ------------------ Review Log Viewer ------------------
    def highlight_keywords(text, keywords):
        for word in keywords:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(f"**:orange[{word}]**", text)
        return text

    keywords = [word for word in re.findall(r'\w+', question) if len(word) > 3]

    with st.expander("🗞️ View Source Review Chunks (Used by AI)", expanded=False):
        st.markdown("_These are the most relevant customer reviews retrieved and passed to the AI model:_")
        for i, chunk in enumerate(relevant_reviews, 1):
            highlighted = highlight_keywords(chunk, keywords)
            with st.container():
                st.markdown(f"**Chunk {i}:**")
                st.markdown(highlighted)
                st.markdown("---")

        log_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_reviews)])
        st.download_button("📂 Download All Review Chunks", log_text, file_name="relevant_chunks_log.txt")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("🚀 Developed by AI-Powered Insights")
