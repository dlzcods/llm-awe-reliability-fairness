from pathlib import Path
from collections import Counter
from typing import Optional

from utils.ioutils import read_xlsx
from utils.textutils import normalize_text, anonymize_id, detect_category, word_count, is_noisy_text


def find_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def process(df, args, defaults=None):
    # args: object with attributes used previously
    resp_col = getattr(args, 'response_col', None) or find_column(df, defaults['response_col_candidates'])
    q_col = getattr(args, 'question_col', None) or find_column(df, defaults['question_col_candidates'])
    t_col = getattr(args, 'type_col', None) or find_column(df, defaults['type_col_candidates'])
    respondent_col = getattr(args, 'respondent_col', None) or find_column(df, defaults['respondent_col_candidates'])

    records = []
    counts = Counter()
    seen = set()
    salt = getattr(args, 'salt', defaults.get('salt'))

    for idx, row in df.iterrows():
        counts['total_rows'] += 1
        raw_resp = row.get(resp_col) if resp_col in row else None
        raw_audio = row.get('audioResponseUrl') if 'audioResponseUrl' in df.columns else None
        raw_q = row.get(q_col) if q_col in row else None
        raw_qid = row.get('questionId') if 'questionId' in df.columns else None
        raw_type = row.get(t_col) if t_col in row else None
        raw_respondent = row.get(respondent_col) if respondent_col in row else None
        submission_id = row.get('submissionId') if 'submissionId' in df.columns else None
        exam_id = row.get('examId') if 'examId' in df.columns else None
        submitted_at = row.get('submittedAt') if 'submittedAt' in df.columns else None

        resp = normalize_text(raw_resp)
        if not resp and raw_audio:
            counts['audio_only'] += 1
            record = {
                'question': normalize_text(raw_q) if raw_q else None,
                'question_id': raw_qid if raw_qid else None,
                'type': raw_type if raw_type else None,
                'response': None,
                'response_type': 'audio',
                'audio_url': raw_audio,
                'respondent_id': anonymize_id(raw_respondent, salt) if getattr(args, 'anonymize', False) and raw_respondent else None,
                'submission_id': submission_id,
                'exam_id': exam_id,
                'submitted_at': str(submitted_at) if submitted_at else None,
                'source_file': getattr(args, 'input', None),
                'row': int(idx) + 1
            }
            key = 'audio:' + (str(raw_audio) or '')
        else:
            if not resp or word_count(resp) < getattr(args, 'min_length', defaults.get('min_length', 40)):
                counts['short_or_empty'] += 1
                continue
            if is_noisy_text(resp):
                counts['filtered_noisy'] += 1
                continue
            cat = detect_category(raw_q if raw_q else '')
            respondent_id = anonymize_id(raw_respondent, salt) if getattr(args, 'anonymize', False) and raw_respondent else None
            record = {
                'question': normalize_text(raw_q) if raw_q else None,
                'question_id': raw_qid if raw_qid else None,
                'type': raw_type if raw_type else None,
                'category': cat,
                'response': resp,
                'response_type': 'text',
                'respondent_id': respondent_id,
                'submission_id': submission_id,
                'exam_id': exam_id,
                'submitted_at': str(submitted_at) if submitted_at else None,
                'source_file': getattr(args, 'input', None),
                'row': int(idx) + 1
            }
            key = 'text:' + (record.get('question_id') or str(record.get('question') or '')) + '|' + resp

        # dedupe
        import hashlib
        h = hashlib.sha256()
        h.update(key.encode('utf-8'))
        k = h.hexdigest()[:16]
        if getattr(args, 'dedupe', False):
            if k in seen:
                counts['deduped'] += 1
                continue
            seen.add(k)

        records.append(record)
        counts['kept'] += 1

    return records, counts


def hashlib_sha(s: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()[:16]
