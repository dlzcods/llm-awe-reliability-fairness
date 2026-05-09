#!/usr/bin/env python3
"""Preprocessing wrapper: run cleaning and sampling steps from one CLI.

Usage examples:
  python src/preprocess.py --clean --input dataset/flattened_answers_report.xlsx --out-dir data/processed
  python src/preprocess.py --sample --input dataset/flattened_answers_report.xlsx --per-question 50
"""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing.clean import read_xlsx, process  # re-exported in preprocessing.clean
from preprocessing.sample import detect_essay_question_ids, sample_per_question
from utils.ioutils import write_jsonl, write_json


def run_clean(input_path, sheet, anonymize, dedupe, min_length, out_dir, preview=0, salt=None):
    df = read_xlsx(input_path, sheet_name=sheet)
    class Args: pass
    args = Args()
    args.response_col = None
    args.question_col = None
    args.type_col = None
    args.respondent_col = None
    args.anonymize = anonymize
    args.dedupe = dedupe
    args.min_length = min_length
    args.preview = preview
    args.out_dir = out_dir
    args.input = input_path
    args.salt = salt
    records, counts = process(df, args)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, str(out_dir / 'cleaned.jsonl'))
    write_json(records, str(out_dir / 'cleaned.json'))
    write_json(counts, str(out_dir / 'metadata.json'))
    print('Cleaned:', len(records))


def run_sample(input_path, sheet, per_question, min_length, out_dir):
    df = read_xlsx(input_path, sheet_name=sheet)
    qids = detect_essay_question_ids(df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for qid in qids:
        total, samples = sample_per_question(df, qid, min_length=min_length, n=per_question)
        summary[qid] = {'total_valid': total, 'sampled': len(samples)}
        write_jsonl(samples, str(out_dir / f'question_{qid}_samples.jsonl'))
    write_json(summary, str(out_dir / 'summary.json'))
    print('Sampling done, questions:', len(qids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--sheet', default='flattened_report')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--anonymize', action='store_true')
    parser.add_argument('--dedupe', action='store_true')
    parser.add_argument('--min-length', type=int, default=40)
    parser.add_argument('--per-question', type=int, default=50)
    parser.add_argument('--out-dir', default='english-bias-framework/data/processed')
    parser.add_argument('--preview', type=int, default=0)
    parser.add_argument('--salt', default=None)

    args = parser.parse_args()
    if args.clean:
        run_clean(args.input, args.sheet, args.anonymize, args.dedupe, args.min_length, args.out_dir, preview=args.preview, salt=args.salt)
    if args.sample:
        run_sample(args.input, args.sheet, args.per_question, args.min_length, args.out_dir)


if __name__ == '__main__':
    main()
