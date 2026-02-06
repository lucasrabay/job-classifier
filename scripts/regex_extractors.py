import re
import pandas as pd
import argparse
from typing import Dict, Any, List

# -----------------------------
# Regex patterns
# -----------------------------

SENIORIDADE_RE = re.compile(
    r"\b(est[aá]gio|intern(ship)?|junior|j[úu]nior|jr\.?|pleno|mid( level)?|senior|s[êe]nior|sr\.?|lead|staff|principal)\b",
    re.IGNORECASE
)

REGIME_RE = re.compile(
    r"\b(remoto|remote|h[ií]brido|hybrid|presencial|on[-\s]?site|in[-\s]?office)\b",
    re.IGNORECASE
)

SALARIO_RE = re.compile(
    r"(R\$\s?\d+(\.\d{3})*(,\d{2})?)|(\$\s?\d{2,3}(,\d{3})*(\.\d{2})?)|(\b\d{2,3}k\b)",
    re.IGNORECASE
)

CARGA_HORARIA_RE = re.compile(
    r"\b(\d{1,2}\s?h\s*/\s*semana|\d{1,2}\s?h/semana|\d{1,2}\s*hours?\s*/\s*week|\d{1,2}\s*hours?\s*per\s*week)\b",
    re.IGNORECASE
)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)

# Skills simples para contar depois (etapa 3/gráficos)
SKILLS = [
    "python","sql","spark","airflow","power bi","tableau","aws","gcp","azure","docker","kubernetes",
    "react","javascript","typescript","node","java","c#","go","rust","django","flask","fastapi",
    "postgres","mysql","snowflake","databricks","dbt","etl","api","ci/cd","terraform", "rails", "ruby"
]

def _findall_clean(regex: re.Pattern, text: str) -> List[str]:
    if not text:
        return []
    found = regex.findall(text)
    # findall pode retornar tuplas dependendo do regex; normaliza:
    cleaned = []
    for item in found:
        if isinstance(item, tuple):
            # pega primeiro item não vazio
            val = next((x for x in item if x), "")
        else:
            val = item
        val = val.strip()
        if val:
            cleaned.append(val)
    # dedup preservando ordem
    seen = set()
    out = []
    for v in cleaned:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out

def extract_regex_fields(text: str) -> Dict[str, Any]:
    text = text or ""
    senioridade = _findall_clean(SENIORIDADE_RE, text)
    regime = _findall_clean(REGIME_RE, text)
    salarios = _findall_clean(SALARIO_RE, text)
    carga = _findall_clean(CARGA_HORARIA_RE, text)
    urls = _findall_clean(URL_RE, text)
    emails = _findall_clean(EMAIL_RE, text)

    skills_found = []
    t_low = text.lower()
    for s in SKILLS:
        if s in t_low:
            skills_found.append(s)

    return {
        "senioridade_re": ", ".join(senioridade),
        "regime_re": ", ".join(regime),
        "salario_re": ", ".join(salarios),
        "carga_horaria_re": ", ".join(carga),
        "urls_re": ", ".join(urls[:5]),
        "emails_re": ", ".join(emails[:5]),
        "skills_re": ", ".join(skills_found),
    }

def apply_regex_to_csv(input_csv: str, output_csv: str, text_col: str) -> None:
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Coluna '{text_col}' não existe. Colunas disponíveis: {list(df.columns)}")

    extracted = df[text_col].fillna("").apply(extract_regex_fields)
    extracted_df = pd.DataFrame(list(extracted))
    out = pd.concat([df, extracted_df], axis=1)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"OK: salvo em {output_csv} com {len(out)} linhas.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data.csv")
    ap.add_argument("--output", default="data_regex.csv")
    ap.add_argument("--text-col", default="descricao_texto")
    args = ap.parse_args()
    apply_regex_to_csv(args.input, args.output, args.text_col)

if __name__ == "__main__":
    main()
