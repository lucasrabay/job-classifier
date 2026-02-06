import argparse
from pathlib import Path

import pandas as pd

import scrapper_weworkremotely as wwr
import regex_extractors
import spacy_stage1
import nltk_preprocess
import spacy_advanced
import classifier
import plots


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_raw_csv(args, data_dir: Path) -> Path:
    default_raw = data_dir / "data.csv"
    if args.scrape:
        bot = wwr.ExtratorVagasWWR(
            base_url=args.base_url,
            qtd_paginas=args.pages,
            max_vagas=args.max_jobs,
        )
        bot.raspar_vagas()
        bot.salvar_csv(str(default_raw))
        return default_raw

    raw_csv = Path(args.input) if args.input else default_raw
    if not raw_csv.exists():
        raise SystemExit(
            "CSV de entrada não encontrado. Use --scrape ou passe --input com um CSV válido."
        )
    return raw_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline end-to-end de vagas")
    ap.add_argument("--scrape", action="store_true", help="Executa o scraper do WWR.")
    ap.add_argument("--input", default="", help="CSV bruto (quando --scrape não é usado).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--text-col", default="descricao_texto")
    ap.add_argument("--base-url", default="https://weworkremotely.com/remote-full-time-jobs")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--max-jobs", type=int, default=120)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    plots_dir = Path(args.plots_dir)
    ensure_dir(data_dir)
    ensure_dir(plots_dir)

    raw_csv = resolve_raw_csv(args, data_dir)

    regex_csv = data_dir / "data_regex.csv"
    regex_extractors.apply_regex_to_csv(str(raw_csv), str(regex_csv), args.text_col)

    stage1_csv = data_dir / "data_stage1.csv"
    spacy_stage1.run(str(regex_csv), str(stage1_csv), args.text_col)

    nltk_csv = data_dir / "data_nltk.csv"
    nltk_preprocess.run(
        str(stage1_csv),
        str(nltk_csv),
        args.text_col,
        args.max_features,
        artifacts_dir=str(data_dir),
    )

    spacy_adv_csv = data_dir / "data_spacy_adv.csv"
    spacy_advanced.run(str(nltk_csv), str(spacy_adv_csv), args.text_col)

    labeled_csv = data_dir / "data_labeled.csv"
    classifier.run(
        str(spacy_adv_csv),
        args.text_col,
        test_size=args.test_size,
        max_rows=(args.max_rows or None),
        output_csv=str(labeled_csv),
        artifacts_dir=str(data_dir),
    )

    if not args.skip_plots:
        df = pd.read_csv(labeled_csv)
        plots.plot_top_skills(df, args.text_col, output_dir=str(plots_dir))
        plots.plot_seniority(df, output_dir=str(plots_dir))
        plots.plot_confusion_matrix(artifacts_dir=str(data_dir), output_dir=str(plots_dir))
        plots.plot_cooccurrence(df, args.text_col, output_dir=str(plots_dir))
        print(f"OK: plots gerados em {plots_dir}")

    print("Pipeline concluído.")


if __name__ == "__main__":
    main()
