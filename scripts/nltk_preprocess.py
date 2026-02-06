import argparse
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

def ensure_nltk():
    # Se já baixou, ok; se não, baixa automaticamente
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(pkg)
        except Exception:
            nltk.download(pkg)

def normalize_basic(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # como vagas são EN geralmente
    text = re.sub(r"\s+", " ", text).strip()
    return text

def nltk_pipeline(text: str, sw_set, lemmatizer, stemmer) -> str:
    text = normalize_basic(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and len(t) > 2]
    tokens = [t for t in tokens if t not in sw_set]

    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    stems = [stemmer.stem(t) for t in lemmas]

    return " ".join(stems)

def build_tfidf(texts, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out().tolist()
    return X, vocab

def run(
    input_csv: str,
    output_csv: str,
    text_col: str,
    max_features: int,
    artifacts_dir: str = ".",
):
    ensure_nltk()
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Coluna '{text_col}' não existe. Colunas: {list(df.columns)}")

    sw_set = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    texts = df[text_col].fillna("").tolist()
    clean_texts = [nltk_pipeline(t, sw_set, lemmatizer, stemmer) for t in texts]

    df["text_clean"] = clean_texts
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"OK: salvo CSV preprocessado em {output_csv}")

    X, vocab = build_tfidf(clean_texts, max_features=max_features)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    tfidf_path = artifacts_path / "tfidf_features.npz"
    vocab_path = artifacts_path / "tfidf_vocab.json"

    sparse.save_npz(tfidf_path, X)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(
        "OK: salvo TF-IDF em "
        f"{tfidf_path} e vocabulário em {vocab_path} (features={len(vocab)})"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_stage1.csv")
    ap.add_argument("--output", default="data_nltk.csv")
    ap.add_argument("--text-col", default="descricao_texto")
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--artifacts-dir", default=".")
    args = ap.parse_args()
    run(
        args.input,
        args.output,
        args.text_col,
        args.max_features,
        artifacts_dir=args.artifacts_dir,
    )

if __name__ == "__main__":
    main()
