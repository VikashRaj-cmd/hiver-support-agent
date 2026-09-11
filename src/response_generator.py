"""
response_generator.py — retrieve the most similar historical reply using TF-IDF,
optionally rewrite it with an LLM if API key is present.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

DATA = "data/processed/brand_conversations_intents.csv"
_df = None
_retrieval_vectorizer = None
_retrieval_matrix = None


def _load():
    global _df, _retrieval_vectorizer, _retrieval_matrix
    if _df is not None:
        return
    _df = pd.read_csv(DATA, dtype=str).fillna("")
    _retrieval_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    _retrieval_matrix = _retrieval_vectorizer.fit_transform(_df["customer_text"])


def retrieve(query: str, intent: str, top_k: int = 1) -> str:
    _load()
    # Filter to same intent first; fall back to all if too few
    subset = _df[_df["intent"] == intent] if (_df["intent"] == intent).sum() > 5 else _df
    idx = subset.index.tolist()
    matrix = _retrieval_matrix[idx]

    q_vec = _retrieval_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    best = sims.argsort()[-top_k:][::-1]
    return subset.iloc[best[0]]["reply_text"]


def generate_with_llm(query: str, retrieved: str, intent: str) -> str:
    """Rewrite retrieved reply with LLM. Falls back to retrieved if no key."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        return retrieved  # extractive fallback

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            f"You are an Amazon customer support agent.\n"
            f"Customer message: {query}\n"
            f"Intent: {intent}\n"
            f"Reference reply from history: {retrieved}\n\n"
            f"Write a concise, helpful reply grounded in the reference. "
            f"Do not invent policies or promises not in the reference. Max 3 sentences."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return retrieved


def get_reply(query: str, intent: str) -> dict:
    retrieved = retrieve(query, intent)
    final = generate_with_llm(query, retrieved, intent)
    return {
        "retrieved_reply": retrieved,
        "final_reply": final,
        "used_llm": final != retrieved,
    }
