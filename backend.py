import re
import os
import faiss
import numpy as np
import groq
from selenium import webdriver
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from config import GROQ_API_KEY, BASE_URL
from dotenv import load_dotenv

# -------------------- Load environment --------------------
from config import GROQ_API_KEY as groq_api_key
print("GROQ API Key:", groq_api_key)


# -------------------- Clean Review Text Function --------------------
def clean_review_text(text):
    """Cleans the review text using regex patterns."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\d+\s+reviews', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:12345\s*){3,}', '', text)
    text = re.sub(r'\bClose\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip(" |")
    text = re.sub(r'✅\s+Trip Verified\s*\|', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------- Scrape Reviews --------------------
def scrape_reviews(base_url, num_pages=5, file_path="reviews.txt"):
    """Scrapes customer reviews and cleans them."""
    if os.path.exists(file_path):
        os.remove(file_path)

    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(30)

    for page_num in range(1, num_pages + 1):
        url = f"{base_url}/page/{page_num}" if page_num > 1 else base_url
        try:
            driver.get(url)
        except Exception as e:
            print(f"Error loading {url}: {e}")
            continue

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.find_all('div', class_='body')

        with open(file_path, 'a', encoding='utf-8') as file:
            for i, review in enumerate(reviews):
                raw_text = review.text.strip()
                cleaned_text = clean_review_text(raw_text)
                file.write(f"Review {i + 1 + (page_num - 1) * len(reviews)}:\n{cleaned_text}\n{'-' * 80}\n")

    driver.quit()


# -------------------- Load Reviews --------------------
def load_text(file_path="reviews.txt"):
    """Loads customer reviews from a file."""
    with open(file_path, encoding='utf-8') as f:
        return f.read()


# -------------------- Split Reviews --------------------
def split_text(data):
    """Splits content into individual cleaned reviews."""
    reviews = [review.strip() for review in data.split("Review")[1:]]
    return [clean_review_text(review) for review in reviews]


# -------------------- Embedding Initialization --------------------
def initialize_embeddings(docs):
    """Generates embeddings using SentenceTransformer."""
    embedding_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")
    embeddings_list = embedding_model.encode(docs, convert_to_tensor=True)
    return embedding_model, embeddings_list


# -------------------- Convert Query to Embedding --------------------
def convert_query_to_embedding(question, embedding_model):
    """Converts a question into an embedding."""
    search_vec = embedding_model.encode(question, convert_to_tensor=True)
    svec = np.array(search_vec).reshape(1, -1)
    return np.ascontiguousarray(svec, dtype="float32")


# -------------------- Retrieve Relevant Reviews --------------------
def retrieve_reviews(question, docs, embedding_model, embeddings_list, return_scores=False):
    """Retrieves the most relevant reviews using FAISS."""
    svec = convert_query_to_embedding(question, embedding_model)
    dim = embeddings_list.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_list)

    distances, indices = index.search(svec, k=5)
    matched_chunks = [docs[i] for i in indices[0]]
    if return_scores:
        return matched_chunks, distances[0]
    return matched_chunks


# -------------------- Analyze Feedback --------------------
def analyze_feedback(question, documents, model="llama3-8b-8192", temperature=0.7, max_tokens=1000):
    """Uses LLM to analyze customer feedback."""
    client = groq.Client(api_key=groq_api_key)
    batch_size = 3
    results = []

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        input_text = f"{question}\n\nDocuments:\n" + "\n".join(batch_docs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyze the customer feedback in a structured format.\n\n"
                        "Answer and be specific to the question provided and answer only based on documents provided.\n\n"
                        "**Customer Feedback Analysis – Qatar Airways**\n\n"
                        "**Summary:**\n- Provide a high-level overview.\n\n"
                        "**Key Insights:**\n- Identify positive and negative experiences.\n\n"
                        "**Specific Examples:**\n- Mention customer names and issues.\n\n"
                        "**Actionable Recommendations:**\n- Provide practical solutions.\n"
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        results.append(response.choices[0].message.content.strip())

    return "\n\n".join(results)


# -------------------- Preload and Cache Embeddings --------------------
def preload_data(file_path="reviews.txt"):
    """Loads and preprocesses reviews, returns chunks, model, and embeddings."""
    if not os.path.exists(file_path):
        scrape_reviews(BASE_URL, num_pages=5)

    reviews_text = load_text(file_path)
    chunks = split_text(reviews_text)
    embedding_model, embeddings_list = initialize_embeddings(chunks)
    return chunks, embedding_model, embeddings_list


# ✅ Preload only after all functions are defined
preloaded_chunks, preloaded_embedding_model, preloaded_embeddings_list = preload_data()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # Scrape and process reviews
    scrape_reviews(BASE_URL, num_pages=5)
    reviews_text = load_text()
    chunks = split_text(reviews_text)

    embedding_model, embeddings_list = initialize_embeddings(chunks)

    # Example query
    question = "What did Michael Schade say about Qatar Airways?"
    relevant_reviews = retrieve_reviews(question, chunks, embedding_model, embeddings_list)

    response = analyze_feedback(question, relevant_reviews)

    print("\n### LLM Response ###\n")
    print(response)
