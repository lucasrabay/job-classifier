import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

LABELS = ["Dados", "Backend", "Frontend", "DevOps", "Mobile", "Produto", "QA", "Design", "Security", "Outro"]

# (keyword, weight) — higher weight = stronger signal for that category
KEYWORDS = {
    "Dados": [
        # Strong signals (w=3)
        ("data engineer", 3), ("data scientist", 3), ("data analyst", 3), ("machine learning", 3),
        ("data warehouse", 3), ("business intelligence", 3), ("mlops", 3),
        # Medium signals (w=2)
        ("etl", 2), ("spark", 2), ("airflow", 2), ("dbt", 2), ("databricks", 2),
        ("snowflake", 2), ("bigquery", 2), ("redshift", 2), ("power bi", 2), ("tableau", 2),
        ("looker", 2), ("hadoop", 2), ("deep learning", 2), ("nlp", 2), ("computer vision", 2),
        ("tensorflow", 2), ("pytorch", 2), ("pandas", 2), ("scikit", 2), ("jupyter", 2),
        ("data lake", 2), ("feature engineering", 2), ("a/b test", 2),
        # Weak signals (w=1)
        ("sql", 1), ("analytics", 1), ("pipeline", 1), ("statistics", 1), ("regression", 1),
        ("kafka", 1), ("fivetran", 1), ("metabase", 1), ("superset", 1),
    ],
    "DevOps": [
        ("site reliability", 3), ("sre", 3), ("platform engineer", 3), ("devops", 3),
        ("infrastructure engineer", 3), ("cloud engineer", 3),
        ("kubernetes", 2), ("terraform", 2), ("ansible", 2), ("docker", 2), ("helm", 2),
        ("ci/cd", 2), ("jenkins", 2), ("github actions", 2), ("gitlab ci", 2), ("argocd", 2),
        ("prometheus", 2), ("grafana", 2), ("datadog", 2), ("observability", 2),
        ("cloudformation", 2), ("pulumi", 2), ("vault", 2), ("istio", 2), ("service mesh", 2),
        ("aws", 1), ("gcp", 1), ("azure", 1), ("linux", 1), ("nginx", 1), ("monitoring", 1),
    ],
    "Frontend": [
        ("frontend engineer", 3), ("front-end engineer", 3), ("front end developer", 3),
        ("ui developer", 3), ("ui engineer", 3),
        ("react", 2), ("next.js", 2), ("vue", 2), ("nuxt", 2), ("angular", 2), ("svelte", 2),
        ("tailwind", 2), ("sass", 2), ("webpack", 2), ("vite", 2), ("storybook", 2),
        ("responsive design", 2), ("web components", 2), ("redux", 2), ("zustand", 2),
        ("javascript", 1), ("typescript", 1), ("html", 1), ("css", 1), ("dom", 1),
        ("accessibility", 1), ("a11y", 1), ("figma", 1),
    ],
    "Backend": [
        ("backend engineer", 3), ("back-end engineer", 3), ("back end developer", 3),
        ("server-side", 3), ("api engineer", 3),
        ("microservices", 2), ("spring boot", 2), ("spring", 2), ("django", 2), ("flask", 2),
        ("fastapi", 2), ("express", 2), ("nestjs", 2), ("rails", 2), ("laravel", 2),
        ("graphql", 2), ("grpc", 2), ("rest api", 2), ("rabbitmq", 2), ("celery", 2),
        ("redis", 2), ("elasticsearch", 2), ("postgres", 2), ("mysql", 2), ("mongodb", 2),
        ("java", 1), ("python", 1), ("node", 1), ("go", 1), ("golang", 1),
        ("ruby", 1), ("rust", 1), ("c#", 1), (".net", 1), ("php", 1), ("scala", 1),
    ],
    "Mobile": [
        ("mobile engineer", 3), ("mobile developer", 3), ("ios developer", 3),
        ("android developer", 3), ("ios engineer", 3), ("android engineer", 3),
        ("react native", 3), ("flutter", 3), ("swift", 2), ("swiftui", 2),
        ("kotlin", 2), ("objective-c", 2), ("xcode", 2), ("android studio", 2),
        ("cocoapods", 2), ("gradle", 2), ("jetpack compose", 2), ("expo", 2),
        ("ios", 1), ("android", 1), ("mobile", 1), ("app store", 1), ("play store", 1),
    ],
    "Produto": [
        ("product manager", 3), ("product owner", 3), ("product lead", 3),
        ("product management", 3), ("head of product", 3), ("vp product", 3),
        ("roadmap", 2), ("stakeholders", 2), ("discovery", 2), ("okr", 2),
        ("user stories", 2), ("product strategy", 2), ("prioritization", 2),
        ("jira", 1), ("backlog", 1), ("agile", 1), ("scrum", 1), ("sprint", 1),
        ("requirements", 1), ("kpi", 1),
    ],
    "QA": [
        ("qa engineer", 3), ("quality assurance", 3), ("test engineer", 3),
        ("sdet", 3), ("test automation", 3), ("quality engineer", 3),
        ("selenium", 2), ("cypress", 2), ("playwright", 2), ("appium", 2),
        ("pytest", 2), ("jest", 2), ("test plan", 2), ("regression testing", 2),
        ("load testing", 2), ("performance testing", 2), ("manual testing", 2),
        ("testing", 1), ("bug", 1),
    ],
    "Design": [
        ("ux designer", 3), ("ui designer", 3), ("product designer", 3),
        ("ux researcher", 3), ("design system", 3), ("interaction designer", 3),
        ("figma", 2), ("sketch", 2), ("prototyping", 2), ("wireframe", 2),
        ("user research", 2), ("usability", 2), ("design tokens", 2),
        ("typography", 1), ("color theory", 1), ("visual design", 1),
    ],
    "Security": [
        ("security engineer", 3), ("cybersecurity", 3), ("infosec", 3),
        ("penetration test", 3), ("appsec", 3), ("devsecops", 3), ("soc analyst", 3),
        ("owasp", 2), ("vulnerability", 2), ("siem", 2), ("threat model", 2),
        ("encryption", 2), ("iam", 2), ("zero trust", 2), ("compliance", 2),
        ("firewall", 1), ("authentication", 1), ("authorization", 1),
    ],
}

# Title patterns that are strong indicators for a category (weight=5)
TITLE_PATTERNS = {
    "Dados": re.compile(r"\b(data\s*(engineer|scientist|analyst)|machine learning|ml engineer|bi analyst|analytics engineer)\b", re.I),
    "DevOps": re.compile(r"\b(devops|sre|site reliability|platform engineer|infrastructure|cloud engineer)\b", re.I),
    "Frontend": re.compile(r"\b(front[\s-]?end|ui\s*(developer|engineer)|react developer|web developer)\b", re.I),
    "Backend": re.compile(r"\b(back[\s-]?end|server[\s-]?side|api (developer|engineer)|software engineer)\b", re.I),
    "Mobile": re.compile(r"\b(mobile|ios|android|react native|flutter)\s*(developer|engineer)?\b", re.I),
    "Produto": re.compile(r"\b(product\s*(manager|owner|lead|director|vp)|head of product)\b", re.I),
    "QA": re.compile(r"\b(qa|quality assurance|sdet|test (engineer|lead|automation))\b", re.I),
    "Design": re.compile(r"\b(ux|ui|product)\s*designer|ux researcher|design (lead|director)\b", re.I),
    "Security": re.compile(r"\b(security|infosec|cybersecurity|appsec|devsecops|soc analyst)\b", re.I),
}

TITLE_BOOST = 5

def label_by_heuristic(text: str, title: str = "") -> str:
    t = (text or "").lower()
    title_lower = (title or "").lower()
    scores = {k: 0 for k in KEYWORDS}

    # Score from description keywords
    for label, kw_list in KEYWORDS.items():
        for kw, weight in kw_list:
            if kw in t:
                scores[label] += weight

    # Boost from title patterns
    for label, pattern in TITLE_PATTERNS.items():
        if pattern.search(title_lower) or pattern.search(t[:300]):
            scores[label] += TITLE_BOOST

    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return "Outro"

    # Require minimum score of 2 to avoid weak single-keyword matches
    if scores[best_label] < 2:
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
    titles = df["titulo"].fillna("").astype(str).tolist() if "titulo" in df.columns else [""] * len(texts)
    y = [label_by_heuristic(t, title=ti) for t, ti in zip(texts, titles)]
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
