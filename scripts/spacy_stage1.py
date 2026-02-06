import argparse
import pandas as pd
import spacy
from collections import Counter

def load_spacy_model():
    # Pra vagas do WWR, inglês costuma funcionar bem:
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # fallback multi-idioma (se você não baixou o en)
        return spacy.load("xx_ent_wiki_sm")

def spacy_basic_features(nlp, text: str, top_k_lemmas: int = 8):
    text = text or ""
    doc = nlp(text)

    sent_count = sum(1 for _ in doc.sents)
    tokens = [t for t in doc if not t.is_space]
    token_count = len(tokens)

    lemmas = [
        t.lemma_.lower()
        for t in tokens
        if t.is_alpha and not t.is_stop and len(t.lemma_) > 2
    ]
    top_lemmas = [w for w, _ in Counter(lemmas).most_common(top_k_lemmas)]

    orgs = []
    places = []
    for ent in doc.ents:
        label = ent.label_
        if label in ("ORG",):
            orgs.append(ent.text.strip())
        if label in ("GPE", "LOC"):
            places.append(ent.text.strip())

    # dedup rápido
    orgs = list(dict.fromkeys(orgs))[:6]
    places = list(dict.fromkeys(places))[:6]

    return {
        "spacy_sentences": sent_count,
        "spacy_tokens": token_count,
        "spacy_top_lemmas": ", ".join(top_lemmas),
        "spacy_orgs": ", ".join(orgs),
        "spacy_places": ", ".join(places),
    }

def run(input_csv: str, output_csv: str, text_col: str):
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Coluna '{text_col}' não existe. Colunas: {list(df.columns)}")

    nlp = load_spacy_model()
    # melhora velocidade em batch
    texts = df[text_col].fillna("").tolist()

    rows = []
    for doc in nlp.pipe(texts, batch_size=32):
        # reaproveita a mesma lógica, mas com doc já pronto
        sent_count = sum(1 for _ in doc.sents)
        tokens = [t for t in doc if not t.is_space]
        lemmas = [t.lemma_.lower() for t in tokens if t.is_alpha and not t.is_stop and len(t.lemma_) > 2]
        top_lemmas = [w for w, _ in Counter(lemmas).most_common(8)]

        orgs, places = [], []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                orgs.append(ent.text.strip())
            if ent.label_ in ("GPE", "LOC"):
                places.append(ent.text.strip())

        rows.append({
            "spacy_sentences": sent_count,
            "spacy_tokens": len(tokens),
            "spacy_top_lemmas": ", ".join(top_lemmas),
            "spacy_orgs": ", ".join(list(dict.fromkeys(orgs))[:6]),
            "spacy_places": ", ".join(list(dict.fromkeys(places))[:6]),
        })

    out = pd.concat([df, pd.DataFrame(rows)], axis=1)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"OK: salvo em {output_csv} ({len(out)} linhas).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_regex.csv")
    ap.add_argument("--output", default="data_stage1.csv")
    ap.add_argument("--text-col", default="descricao_texto")
    args = ap.parse_args()
    run(args.input, args.output, args.text_col)

if __name__ == "__main__":
    main()
