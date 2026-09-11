"""
prepare_data.py — load twcs.csv, show brand counts, filter chosen brand,
build conversation pairs, save to data/processed/brand_conversations.csv
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW = "data/raw/twcs.csv"
OUT = "data/processed/brand_conversations.csv"

def clean_text(text: str) -> str:
    text = re.sub(r"@\w+", "", str(text))          # remove @mentions
    text = re.sub(r"http\S+", "", text)             # remove URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def show_brands(df: pd.DataFrame):
    # Support accounts are those that only reply (inbound == False rows authored by brand handles)
    # The dataset marks inbound=True for customer tweets, False for brand replies
    brand_col = "author_id"
    inbound_col = "inbound"

    brands = (
        df[df[inbound_col].str.strip().str.lower() == "false"][brand_col]
        .value_counts()
        .head(20)
    )
    print("\nTop 20 brands by reply count:")
    print(brands.to_string())
    print()

def build_conversations(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    df["inbound"] = df["inbound"].str.strip().str.lower()

    # All brand replies
    brand_replies = df[df["author_id"] == brand][["tweet_id", "text", "in_response_to_tweet_id"]].copy()
    brand_replies.columns = ["reply_id", "reply_text", "customer_tweet_id"]

    # All customer tweets
    customer_tweets = df[df["inbound"] == "true"][["tweet_id", "text"]].copy()
    customer_tweets.columns = ["customer_tweet_id", "customer_text"]

    # Join on tweet id
    pairs = brand_replies.merge(customer_tweets, on="customer_tweet_id", how="inner")
    pairs["customer_text"] = pairs["customer_text"].apply(clean_text)
    pairs["reply_text"] = pairs["reply_text"].apply(clean_text)

    # Drop empty rows
    pairs = pairs[pairs["customer_text"].str.len() > 5]
    pairs = pairs[pairs["reply_text"].str.len() > 5]
    pairs = pairs.drop_duplicates(subset=["customer_text"])

    return pairs.reset_index(drop=True)

def main():
    print(f"Loading {RAW} ...")
    df = load_raw()
    print(f"Total rows: {len(df):,}")

    show_brands(df)

    brand = os.getenv("TARGET_BRAND", "").strip()
    if not brand:
        print("Set TARGET_BRAND in your .env file, then re-run.")
        return

    print(f"Filtering for brand: {brand}")
    pairs = build_conversations(df, brand)
    print(f"Conversation pairs found: {len(pairs):,}")

    os.makedirs("data/processed", exist_ok=True)
    pairs.to_csv(OUT, index=False)
    print(f"Saved → {OUT}")
    print("\nSample:")
    print(pairs[["customer_text", "reply_text"]].head(3).to_string())

if __name__ == "__main__":
    main()
