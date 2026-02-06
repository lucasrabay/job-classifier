# Job Classifier Pipeline

End-to-end pipeline to scrape remote job postings, enrich them with regex + NLP features, generate weak labels, and produce simple analytics/plots.

The project is organized as small, reusable modules and a single pipeline script that stitches everything together.

## Concept

1. **Collect** job postings from We Work Remotely (or supply your own CSV).
2. **Extract** structured signals with regex (seniority, regime, salary, skills, URLs, etc.).
3. **Enrich** with spaCy features (basic + advanced).
4. **Clean** text with NLTK, build TF‑IDF features.
5. **Label** with a heuristic model and train a quick baseline classifier.
6. **Plot** skill frequency, seniority distribution, confusion matrix, and skill co-occurrence.

All artifacts are saved to `data/` and `plots/`.

## File Structure

- `scripts/pipeline.py` – Orchestrates the full pipeline.
- `scripts/scrapper_weworkremotely.py` – Scrapes job listings.
- `scripts/regex_extractors.py` – Regex-based feature extraction.
- `scripts/spacy_stage1.py` – Basic spaCy features.
- `scripts/nltk_preprocess.py` – Tokenization, lemmatization, stemming, TF‑IDF.
- `scripts/spacy_advanced.py` – Advanced spaCy features.
- `scripts/classifier.py` – Heuristic labeling + baseline classifier.
- `scripts/plots.py` – Generates PNG plots.
- `data/` – Generated CSVs + artifacts:
  - `data.csv` (raw scrape)
  - `data_regex.csv`
  - `data_stage1.csv`
  - `data_nltk.csv`
  - `data_spacy_adv.csv`
  - `data_labeled.csv`
  - `tfidf_features.npz`, `tfidf_vocab.json`
  - `confusion_matrix.npy`, `labels_order.txt`
- `plots/` – Generated PNG charts.

## How To Run

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the full pipeline

Using an existing CSV:

```bash
python scripts/pipeline.py --input data/data.csv
```

Scrape from We Work Remotely and run everything:

```bash
python scripts/pipeline.py --scrape --pages 2 --max-jobs 120
```

Skip plots:

```bash
python scripts/pipeline.py --input data/data.csv --skip-plots
```

### Outputs

- All intermediate CSVs + artifacts are written to `data/`.
- Charts are written to `plots/`.

## Notes

- `descricao_texto` is the default text column. If your CSV uses a different name, pass `--text-col`.
- spaCy models are loaded dynamically. If you don’t have `en_core_web_sm` installed, the scripts fall back to a multilingual model.
- The classifier uses heuristic labels for a quick baseline; it is not a fully supervised model.
