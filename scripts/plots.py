import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

SKILLS = [
    "python","sql","spark","airflow","power bi","tableau","aws","gcp","azure","docker","kubernetes",
    "react","javascript","typescript","node","java","c#","go","rust","django","flask","fastapi",
    "postgres","mysql","snowflake","databricks","dbt","etl","api","ci/cd","terraform"
]

def count_skills(texts):
    cnt = Counter()
    for t in texts:
        tl = (t or "").lower()
        for s in SKILLS:
            if s in tl:
                cnt[s] += 1
    return cnt

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def plot_top_skills(df, text_col, output_dir: str = "."):
    texts = df[text_col].fillna("").astype(str).tolist()
    cnt = count_skills(texts)
    top = cnt.most_common(20)
    if not top:
        print("Sem skills para plotar.")
        return
    labels = [x for x,_ in top]
    values = [y for _,y in top]

    _ensure_dir(output_dir)
    out_path = Path(output_dir) / "plot_top_skills.png"
    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=70, ha="right")
    plt.title("Top 20 skills citadas")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_seniority(df, output_dir: str = "."):
    col = "senioridade_re"
    if col not in df.columns:
        print("senioridade_re não encontrada (rode regex_extractors.py antes).")
        return

    vals = []
    for x in df[col].fillna("").astype(str).tolist():
        if not x.strip():
            vals.append("Sem")
        else:
            # pega primeiro match só pra simplificar
            vals.append(x.split(",")[0].strip().lower())

    c = Counter(vals)
    labels, values = zip(*c.most_common(12))
    _ensure_dir(output_dir)
    out_path = Path(output_dir) / "plot_seniority.png"
    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=60, ha="right")
    plt.title("Distribuição de senioridade (regex)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_confusion_matrix(artifacts_dir: str = ".", output_dir: str = "."):
    try:
        cm_path = Path(artifacts_dir) / "confusion_matrix.npy"
        labels_path = Path(artifacts_dir) / "labels_order.txt"
        cm = np.load(cm_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
    except Exception:
        print("Arquivos confusion_matrix.npy / labels_order.txt não encontrados. Rode classifier.py.")
        return

    _ensure_dir(output_dir)
    out_path = Path(output_dir) / "plot_confusion_matrix.png"
    plt.figure()
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_cooccurrence(df, text_col, output_dir: str = "."):
    # co-ocorrência simples de pares de skills (top 10 pares)
    texts = df[text_col].fillna("").astype(str).tolist()
    pair_cnt = Counter()

    for t in texts:
        tl = (t or "").lower()
        present = [s for s in SKILLS if s in tl]
        present = sorted(set(present))
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                pair_cnt[(present[i], present[j])] += 1

    top = pair_cnt.most_common(10)
    if not top:
        print("Sem pares para co-ocorrência.")
        return

    labels = [f"{a}+{b}" for (a,b), _ in top]
    values = [v for _, v in top]

    _ensure_dir(output_dir)
    out_path = Path(output_dir) / "plot_cooccurrence.png"
    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=70, ha="right")
    plt.title("Top 10 co-ocorrências de skills")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_labeled.csv")
    ap.add_argument("--text-col", default="descricao_texto")
    ap.add_argument("--artifacts-dir", default=".")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    plot_top_skills(df, args.text_col, output_dir=args.output_dir)
    plot_seniority(df, output_dir=args.output_dir)
    plot_confusion_matrix(artifacts_dir=args.artifacts_dir, output_dir=args.output_dir)
    plot_cooccurrence(df, args.text_col, output_dir=args.output_dir)
    print("OK: plots gerados (PNG).")

if __name__ == "__main__":
    main()
