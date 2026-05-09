from pathlib import Path
from utils.ioutils import read_xlsx
from utils.textutils import normalize_text, anonymize_id, detect_category


def detect_essay_question_ids(df, qid_col='questionId', qtype_col='questionType', resp_col='textResponse'):
    if qtype_col in df.columns:
        mask = df[qtype_col].astype(str).str.lower().fillna('')
        essay_ids = df.loc[mask.str.contains('essay|writing'), qid_col].dropna().unique().tolist()
        if len(essay_ids) >= 1:
            return list(map(str, essay_ids))
    if qid_col in df.columns and resp_col in df.columns:
        resp_mask = df[resp_col].astype(str).str.strip().replace({'': None}).notna()
        counts = df.loc[resp_mask].groupby(qid_col).size().sort_values(ascending=False)
        return [str(x) for x in counts.head(3).index.tolist()]
    return []


def sample_per_question(df, qid, qid_col='questionId', qtext_col='questionText', resp_col='textResponse', type_col='questionType', min_length=40, n=50, salt='english-bias-framework-default-salt'):
    qdf = df[df[qid_col].astype(str) == str(qid)].copy()
    qdf['norm_resp'] = qdf[resp_col].astype(str).apply(normalize_text)
    valid = qdf[qdf['norm_resp'].str.split().str.len() >= min_length]
    total = len(valid)
    samples = []
    if total > 0:
        samples_df = valid.sample(n=min(n, total), random_state=42)
        for _, row in samples_df.iterrows():
            respondent_raw = row.get('nim') if 'nim' in df.columns else row.get('nama') if 'nama' in df.columns else None
            respondent_id = anonymize_id(respondent_raw, salt) if respondent_raw else None
            question_text = normalize_text(row.get(qtext_col)) if qtext_col in df.columns else None
            qtype = row.get(type_col) if type_col in df.columns else None
            category = detect_category(question_text)
            samples.append({
                'respondent_id': respondent_id,
                'question': question_text,
                'question_id': str(row.get(qid_col)),
                'type': qtype,
                'category': category,
                'response': row.get('norm_resp'),
                'submitted_at': str(row.get('submittedAt')) if 'submittedAt' in df.columns else None,
                'source_file': None,
                'row': int(row.name) + 1
            })
    return total, samples
