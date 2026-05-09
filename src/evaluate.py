#!/usr/bin/env python3
"""Evaluation scaffold for running LLM scoring (Groq/OpenAI client) on samples.

This is a scaffold: it reads per-question samples (`.jsonl`) and sends prompts to configured models,
saves model responses and parsed scores to an output folder for later analysis.

NOTE: set `GROQ_API_KEY` in env or pass via CLI. The script uses the `openai` client as in the notebook.
"""
import argparse
from pathlib import Path
import os
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from utils.ioutils import write_jsonl, write_json


# Try to load a .env file from the project root (english-bias-framework/.env) or parent folders.
def _load_local_dotenv():
    try:
        # look for a .env file in the repo (two levels up from this file: src/ -> project/)
        p = Path(__file__).resolve().parent.parent
        env_path = p / '.env'
        # fallback: search upward a few levels
        if not env_path.exists():
            q = Path(__file__).resolve().parent
            for _ in range(4):
                cand = q / '.env'
                if cand.exists():
                    env_path = cand
                    break
                q = q.parent

        if env_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path))
                return
            except Exception:
                # fallback to manual parse
                with env_path.open('r', encoding='utf-8') as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln or ln.startswith('#'):
                            continue
                        if '=' in ln:
                            k, v = ln.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            os.environ.setdefault(k, v)
    except Exception:
        return


# load .env early
_load_local_dotenv()


def load_samples(path):
    records = []
    p = Path(path)
    if not p.exists():
        return records
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def build_prompt(question, response):
    return (
        "You are an objective, rubric-based grader.\n"
        "INSTRUCTIONS: Output EXACTLY one JSON object and NOTHING else. Do NOT include explanations, reasoning, or any additional text. No bullet lists, no code fences, no commentary.\n\n"
        "OUTPUT SCHEMA: {\"grammar\": int, \"lexical\": int, \"coherence\": int, \"task\": int }\n"
        "- Each value MUST be an integer between 0 and 9 (inclusive). Round to the nearest integer if needed.\n\n"
        "SCORING BRIEFS (one line each):\n"
        "- grammar: accuracy and range of grammar, spelling, punctuation.\n"
        "- lexical: vocabulary range and word choice appropriateness.\n"
        "- coherence: organization, logic, and flow of ideas.\n"
        "- task: how well the response addresses the prompt and fulfills requirements.\n\n"
        "RULES: If uncertain, choose the closest integer. If the response cannot be scored, set all fields to null. Do NOT reveal chain-of-thought. Keep the JSON compact (no extra spaces or newlines).\n\n"
        f"Question: {question}\n\nResponse: {response}\n"
    )


def call_model(prompt, model, api_key, base_url=None, temperature=0.7):
    # Uses the Groq Python client per documentation. Caller must ensure api_key available.
    try:
        from groq import Groq
    except Exception:
        raise RuntimeError('groq client not installed. pip install groq')

    # Instantiate client. Groq will read env vars if api_key is None.
    client = Groq(api_key=api_key) if api_key else Groq()

    # Build messages structure expected by the API
    messages = [{"role": "user", "content": prompt}]

    # Call Groq chat completions
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        reasoning_format="hidden",
        max_completion_tokens=700,
        top_p=1,
        stream=False,
    )

    # Mirror previous behavior: return the assistant content string
    try:
        return resp.choices[0].message.content
    except Exception:
        # best-effort: stringify response
        return str(resp)


def parse_scores_from_text(text):
    # Lightweight parser similar to notebook: try to extract JSON-like object of 4 numbers.
    import re
    m = re.search(r"\{\s*\"grammar\"\s*:\s*(\d+),\s*\"lexical\"\s*:\s*(\d+),\s*\"coherence\"\s*:\s*(\d+),\s*\"task\"\s*:\s*(\d+)\s*\}", text)
    if m:
        return {'grammar': int(m.group(1)), 'lexical': int(m.group(2)), 'coherence': int(m.group(3)), 'task': int(m.group(4))}
    # fallback: find 4 single-digit numbers
    nums = re.findall(r"\b([0-9])\b", text)
    if len(nums) >= 4:
        return {'grammar': int(nums[0]), 'lexical': int(nums[1]), 'coherence': int(nums[2]), 'task': int(nums[3])}
    return None


def evaluate_file(sample_path, model, api_key, base_url, out_dir, runs=1, dry_run=False, temperature=0.7, max_samples=None, checkpoint_path=None, skip_set=None):
    samples = load_samples(sample_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    results = []
    if checkpoint_path is None:
        checkpoint_path = str(Path(out_dir) / 'checkpoint.jsonl')

    for s in samples:
        prompt = build_prompt(s.get('question', ''), s.get('response', ''))
        for run in range(runs):
            # check skip set to avoid duplicate requests when resuming
            if skip_set is not None:
                key = (model, run + 1, int(s.get('row') or -1), s.get('respondent_id'))
                if key in skip_set:
                    # already done — skip making a request
                    continue
            try:
                if dry_run:
                    text = '{"grammar": 7, "lexical": 7, "coherence": 7, "task": 7}'
                else:
                    text = call_model(prompt, model, api_key, base_url=base_url, temperature=temperature)
            except Exception as e:
                text = f'ERROR: {e}'

            parsed = parse_scores_from_text(text)
            entry = {
                'model': model,
                'run': run + 1,
                'sample_row': s.get('row'),
                'question_id': s.get('question_id'),
                'respondent_id': s.get('respondent_id'),
                'response': s.get('response'),
                'raw': text,
                'parsed': parsed
            }
            results.append(entry)

            # append checkpoint immediately to reduce data-loss on failure
            try:
                with open(checkpoint_path, 'a', encoding='utf-8') as cf:
                    cf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

            # pause between requests
            try:
                import time
                time.sleep(4)
            except Exception:
                pass

    # Create per-sample output directory and write per-question file
    sample_stem = Path(sample_path).stem
    # remove any trailing suffix like '_samples' from stem
    if sample_stem.endswith('_samples'):
        sample_stem = sample_stem[: -len('_samples')]
    model_safe = model.replace('/', '_').replace('\\', '_')
    # try to infer question id from samples
    question_id = None
    if samples:
        question_id = samples[0].get('question_id') or samples[0].get('question')
    if not question_id:
        question_id = sample_stem

    sample_out_dir = Path(out_dir) / sample_stem / model_safe
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_out_dir / f"{question_id}.jsonl"
    write_jsonl(results, str(out_path))
    return len(results)


def evaluate_sample(s, model, api_key, base_url, out_dir, runs=1, dry_run=False, temperature=0.7, checkpoint_path=None, skip_set=None):
    # Evaluate a single sample record for a given model. Uses same checkpoint semantics as evaluate_file.
    if checkpoint_path is None:
        checkpoint_path = str(Path(out_dir) / 'checkpoint.jsonl')

    results = []
    question = s.get('question') or s.get('question_id')
    prompt = build_prompt(s.get('question', ''), s.get('response', ''))

    for run in range(runs):
        if skip_set is not None:
            key = (model, run + 1, int(s.get('row') or -1), s.get('respondent_id'))
            if key in skip_set:
                continue
        try:
            if dry_run:
                text = '{"grammar": 7, "lexical": 7, "coherence": 7, "task": 7}'
            else:
                text = call_model(prompt, model, api_key, base_url=base_url, temperature=temperature)
        except Exception as e:
            text = f'ERROR: {e}'

        parsed = parse_scores_from_text(text)
        entry = {
            'model': model,
            'run': run + 1,
            'sample_row': s.get('row'),
            'question_id': s.get('question_id'),
            'respondent_id': s.get('respondent_id'),
            'response': s.get('response'),
            'raw': text,
            'parsed': parsed
        }

        # append checkpoint immediately
        try:
            with open(checkpoint_path, 'a', encoding='utf-8') as cf:
                cf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # also append to per-model per-question file for convenience
        try:
            sample_stem = (s.get('source_file') or '')
            # fallback to question id as stem
            if not sample_stem:
                sample_stem = 'question_' + str(s.get('question_id') or '')
            model_safe = model.replace('/', '_').replace('\\', '_')
            question_id = s.get('question_id') or s.get('question') or sample_stem
            sample_out_dir = Path(out_dir) / Path(sample_stem).stem / model_safe
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = sample_out_dir / f"{question_id}.jsonl"
            with open(out_path, 'a', encoding='utf-8') as of:
                of.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        results.append(entry)

        try:
            import time
            time.sleep(4)
        except Exception:
            pass

    return len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', required=True)
    parser.add_argument('--models', nargs='+', required=True, help='Model names to evaluate')
    parser.add_argument('--api-key', default=os.getenv('GROQ_API_KEY'))
    parser.add_argument('--base-url', default=os.getenv('GROQ_BASE_URL'))
    parser.add_argument('--dry-run', action='store_true', help='Do not call external API; synthesize responses')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model requests')
    parser.add_argument('--max-questions', type=int, default=None, help='Limit number of question files to evaluate')
    parser.add_argument('--samples-per-question', type=int, default=None, help='Limit number of samples per question to evaluate')
    parser.add_argument('--sample-files', nargs='*', default=None, help='Specific sample filenames (relative to --samples-dir) to evaluate')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint jsonl file')
    parser.add_argument('--resume', action='store_true', help='Skip requests already present in the checkpoint')
    parser.add_argument('--out-dir', default='english-bias-framework/data/processed/eval')
    args = parser.parse_args()

    if not args.api_key and not args.dry_run:
        raise RuntimeError('API key missing: set GROQ_API_KEY or pass --api-key')

    p = Path(args.samples_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(list(p.glob('question_*_samples.jsonl')))
    sample_files = all_files
    # If specific sample filenames were provided, filter by them
    if args.sample_files:
        picks = [str(Path(args.samples_dir) / s) for s in args.sample_files]
        sample_files = [f for f in all_files if str(f) in picks or f.name in args.sample_files]
    if args.max_questions is not None:
        sample_files = sample_files[: args.max_questions]

    # determine checkpoint path
    checkpoint_path = args.checkpoint or str(Path(args.out_dir) / 'checkpoint.jsonl')

    # if resume requested, load existing checkpoint entries into a set of keys to skip
    # We will rely on per-response skip entries (skip_set) instead of skipping whole
    # model+question combos to avoid false skips. evaluate_file checks skip_set.
    skip_set = set()
    if args.resume and Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as cf:
                for ln in cf:
                    try:
                        j = json.loads(ln)
                        key = (
                            j.get('model'),
                            int(j.get('run') or 0),
                            int(j.get('sample_row') or -1),
                            j.get('respondent_id')
                        )
                        skip_set.add(key)
                    except Exception:
                        continue
        except Exception:
            skip_set = set()

    for sample_file in sample_files:
        # load samples for this question
        tmp = []
        try:
            tmp = load_samples(str(sample_file))
        except Exception:
            tmp = []

        if args.samples_per_question is not None:
            tmp = tmp[: args.samples_per_question]

        # process per-respondent: for each sample record, call each model in turn
        total_done = 0
        for s in tmp:
            for model in args.models:
                n = evaluate_sample(
                    s,
                    model,
                    args.api_key,
                    args.base_url,
                    out_dir,
                    runs=args.runs,
                    dry_run=args.dry_run,
                    temperature=args.temperature,
                    checkpoint_path=checkpoint_path,
                    skip_set=skip_set,
                )
                if n:
                    print(f'Wrote {n} entries for {s.get("respondent_id")} with {model}')
                total_done += n

        print(f'Evaluated {total_done} model responses for {sample_file.name} (per-respondent ordering)')


if __name__ == '__main__':
    main()
