import argparse
import pandas as pd
import spacy
from collections import Counter

def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return spacy.load("xx_ent_wiki_sm")

def summarize_pos(doc, top_k=6):
    tags = [t.pos_ for t in doc if not t.is_space]
    return ", ".join([p for p, _ in Counter(tags).most_common(top_k)])

def summarize_noun_chunks(doc, top_k=8):
    chunks = []
    for nc in doc.noun_chunks:
        txt = nc.text.strip().lower()
        if 2 <= len(txt) <= 50:
            chunks.append(txt)
    return ", ".join([c for c, _ in Counter(chunks).most_common(top_k)])

def summarize_ents(doc):
    orgs, places, others = [], [], []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.append(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC"):
            places.append(ent.text.strip())
        else:
            others.append(f"{ent.text.strip()}({ent.label_})")
    # dedup + limit
    orgs = list(dict.fromkeys(orgs))[:8]
    places = list(dict.fromkeys(places))[:8]
    others = list(dict.fromkeys(others))[:8]
    return ", ".join(orgs), ", ".join(places), ", ".join(others)

def run(input_csv: str, output_csv: str, text_col: str):
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Coluna '{text_col}' não existe. Colunas: {list(df.columns)}")

    nlp = load_model()
    texts = df[text_col].fillna("").tolist()

    rows = []
    for doc in nlp.pipe(texts, batch_size=32):
        token_count = sum(1 for t in doc if not t.is_space)
        sent_count = sum(1 for _ in doc.sents) or 1
        orgs, places, other_ents = summarize_ents(doc)

        rows.append({
            "adv_tokens": token_count,
            "adv_sentences": sent_count,
            "adv_tokens_per_sentence": round(token_count / sent_count, 2),
            "adv_top_pos": summarize_pos(doc),
            "adv_top_noun_chunks": summarize_noun_chunks(doc),
            "adv_orgs": orgs,
            "adv_places": places,
            "adv_other_ents": other_ents,
        })

    out = pd.concat([df, pd.DataFrame(rows)], axis=1)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"OK: salvo em {output_csv} ({len(out)} linhas).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_nltk.csv")
    ap.add_argument("--output", default="data_spacy_adv.csv")
    ap.add_argument("--text-col", default="descricao_texto")
    args = ap.parse_args()
    run(args.input, args.output, args.text_col)

if __name__ == "__main__":
    main()
