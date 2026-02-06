import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

LABELS = ["Dados", "Backend", "Frontend", "DevOps", "Mobile", "Produto", "Outro"]

KEYWORDS = {
    "Dados": [
        "etl", "data warehouse", "sql", "power bi", "tableau", "spark", "airflow", "dbt", "databricks",
        "snowflake", "analytics", "pipeline", "bigquery", "redshift"
    ],
    "DevOps": [
        "docker", "kubernetes", "ci/cd", "terraform", "ansible", "sre", "observability", "prometheus",
        "grafana", "helm", "aws", "gcp", "azure"
    ],
    "Frontend": [
        "react", "next.js", "vue", "angular", "javascript", "typescript", "html", "css", "tailwind"
    ],
    "Backend": [
         "microservices", "java", "spring", "node", "django", "flask", "fastapi", "go", "ruby", "rails"
    ],
    "Mobile": [
        "android", "ios", "swift", "kotlin", "react native", "flutter"
    ],
    "Produto": [
        "product manager", "product owner", "roadmap", "stakeholders", "requirements", "discovery", "okr"
    ],
}

def label_by_heuristic(text: str) -> str:
    t = (text or "").lower()
    scores = {k: 0 for k in KEYWORDS.keys()}
    for label, kws in KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[label] += 1
    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return "Outro"
    return best_label

def load_features_from_csv(df: pd.DataFrame, text_col: str):
    # usa a coluna text_clean gerada no nltk_preprocess.py se existir
    if "text_clean" in df.columns:
        return df["text_clean"].fillna("").astype(str).tolist()
    return df[text_col].fillna("").astype(str).tolist()

def run(
    input_csv: str,
    text_col: str,
    test_size: float = 0.2,
    max_rows: int | None = None,
    output_csv: str = "data_labeled.csv",
    artifacts_dir: str = ".",
):
    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.head(max_rows).copy()

    if text_col not in df.columns and "text_clean" not in df.columns:
        raise ValueError(f"Não encontrei '{text_col}' nem 'text_clean' no CSV.")

    texts = load_features_from_csv(df, text_col)
    y = [label_by_heuristic(t) for t in texts]
    df["label_area"] = y

    # treino simples
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    # TF-IDF interno (pra ficar self-contained)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    acc = accuracy_score(y_test, pred)
    print(f"Acurácia: {acc:.3f}\n")
    print(classification_report(y_test, pred, zero_division=0))
    cm = confusion_matrix(y_test, pred, labels=LABELS)

    # salva artefatos básicos pra plots.py
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")
    np.save(artifacts_path / "confusion_matrix.npy", cm)
    with open(artifacts_path / "labels_order.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LABELS))

    print(
        "OK: "
        f"{output_path}, "
        f"{artifacts_path / 'confusion_matrix.npy'}, "
        f"{artifacts_path / 'labels_order.txt'} salvos."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_spacy_adv.csv")
    ap.add_argument("--text-col", default="descricao_texto")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--output", default="data_labeled.csv")
    ap.add_argument("--artifacts-dir", default=".")
    args = ap.parse_args()
    run(
        args.input,
        args.text_col,
        args.test_size,
        (args.max_rows or None),
        output_csv=args.output,
        artifacts_dir=args.artifacts_dir,
    )

if __name__ == "__main__":
    main()
