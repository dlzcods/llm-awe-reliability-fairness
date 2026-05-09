#!/usr/bin/env python3
"""Automated essay scoring analysis.

Pipeline:
1) Read and align JSONL inputs (per-question files preferred).
2) Essay-level metrics across runs (mean, SD, range, CV).
3) Model-level ICC for reliability (two-way mixed, consistency).
4) Severity bias (paired t-test) and topic bias (prompt_category means).
5) HITL routing per essay (CV and score range only).
6) Plots and reports (CSV, Markdown, HTML).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg
except Exception as exc:
    pg = None
    _PINGOUIN_ERR = str(exc)
else:
    _PINGOUIN_ERR = None

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder


SCORE_FIELDS = ['grammar', 'lexical', 'coherence', 'task']


def read_jsonl(path):
    records = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open('r', encoding='utf-8') as fh:
        for ln in fh:
            try:
                records.append(json.loads(ln))
            except Exception:
                continue
    return records


def load_inputs(inputs_path):
    inp = Path(inputs_path)
    if inp.is_file() and inp.name.endswith('.jsonl'):
        return pd.DataFrame(read_jsonl(inp))

    # Prefer per-question per-model JSONL files under the eval folder.
    sample_files = [p for p in inp.rglob('*.jsonl') if p.name != 'checkpoint.jsonl']
    if sample_files:
        records = []
        for f in sample_files:
            records.extend(read_jsonl(f))
        if not records:
            raise FileNotFoundError(f'No readable sample jsonl files found under {inp}')
        return pd.DataFrame(records)

    # Fallback to checkpoint.jsonl
    ck = inp / 'checkpoint.jsonl'
    if not ck.exists():
        raise FileNotFoundError(f'No sample files or checkpoint found under {inp}')
    return pd.DataFrame(read_jsonl(ck))


def load_sample_mapping(inputs_path):
    inp = Path(inputs_path)
    base = inp.parent if inp.is_file() else inp
    candidates = [base.parent / 'samples', base / 'samples']
    samples_dir = next((p for p in candidates if p.exists()), None)
    if samples_dir is None:
        return None, 'question_id'

    sample_files = list(samples_dir.glob('question_*_samples.jsonl'))
    if not sample_files:
        return None, 'question_id'

    records = []
    for f in sample_files:
        for rec in read_jsonl(f):
            question_id = rec.get('question_id')
            if not question_id:
                continue
            records.append({
                'question_id': str(question_id),
                'prompt_category': rec.get('category') or rec.get('prompt_category'),
                'prompt_question': rec.get('question') or rec.get('prompt')
            })

    if not records:
        return None, 'question_id'

    df_map = pd.DataFrame(records)
    df_map['question_id'] = df_map['question_id'].astype(str)
    df_map = df_map.dropna(subset=['question_id'])
    df_map = (
        df_map.groupby('question_id')
        .agg(
            prompt_category=('prompt_category', 'first'),
            prompt_question=('prompt_question', 'first')
        )
        .reset_index()
    )
    return df_map, 'samples.category'


def extract_score_field(row, key):
    parsed = row.get('parsed') if isinstance(row.get('parsed'), dict) else None
    if parsed and key in parsed:
        return parsed.get(key)
    raw = row.get('raw')
    if isinstance(raw, str):
        try:
            raw_json = json.loads(raw)
            if isinstance(raw_json, dict) and key in raw_json:
                return raw_json.get(key)
        except Exception:
            return None
    return None


def expand_scores(df):
    for k in SCORE_FIELDS:
        df[k] = df.apply(lambda r: extract_score_field(r.to_dict(), k), axis=1)
        df[k] = pd.to_numeric(df[k], errors='coerce')
    # Require all rubric fields to compute total score.
    df['total_score'] = df[SCORE_FIELDS].sum(axis=1, min_count=len(SCORE_FIELDS))
    return df


def align_inputs(df, sample_map):
    df['run'] = pd.to_numeric(df.get('run'), errors='coerce').fillna(0).astype(int)
    df['sample_row'] = df.get('sample_row').astype(str)
    df['respondent_id'] = df.get('respondent_id').astype(str)
    df['question_id'] = df.get('question_id').astype(str)

    prompt_note = 'question_id'

    if sample_map is not None:
        df = df.merge(sample_map, on='question_id', how='left', suffixes=('', '_map'))

    if 'prompt_category' in df.columns and df['prompt_category'].notna().any():
        df['prompt_category'] = df['prompt_category'].astype(str)
        prompt_note = 'samples.category'
    elif 'category' in df.columns:
        df['prompt_category'] = df['category'].astype(str)
        prompt_note = 'category'
    elif 'topic' in df.columns:
        df['prompt_category'] = df['topic'].astype(str)
        prompt_note = 'topic'
    else:
        df['prompt_category'] = df['question_id'].astype(str)
        prompt_note = 'question_id'

    if 'prompt_question' not in df.columns:
        if 'question' in df.columns:
            df['prompt_question'] = df['question']
        elif 'prompt' in df.columns:
            df['prompt_question'] = df['prompt']
        else:
            df['prompt_question'] = None

    df['essay_id'] = (
        df['respondent_id'].astype(str)
        + '|'
        + df['question_id'].astype(str)
        + '|'
        + df['sample_row'].astype(str)
    )
    return df, prompt_note


def compute_essay_metrics(df):
    agg = df.groupby(['model', 'essay_id']).agg(
        respondent_id=('respondent_id', 'first'),
        question_id=('question_id', 'first'),
        sample_row=('sample_row', 'first'),
        prompt_category=('prompt_category', 'first'),
        prompt_question=('prompt_question', 'first'),
        mean_total=('total_score', 'mean'),
        sd_total=('total_score', 'std'),
        min_total=('total_score', 'min'),
        max_total=('total_score', 'max'),
        n_runs=('total_score', 'count')
    ).reset_index()
    agg['sd_total'] = agg['sd_total'].fillna(0.0)
    agg['range_total'] = (agg['max_total'] - agg['min_total']).fillna(0.0)
    agg['cv_total'] = agg.apply(
        lambda r: float(r['sd_total'] / r['mean_total']) if r['mean_total'] and not math.isnan(r['mean_total']) else 0.0,
        axis=1
    )
    return agg


def compute_model_icc(df):
    if pg is None:
        raise ImportError(f'pingouin is required for ICC. Import error: {_PINGOUIN_ERR}')

    out = {}
    for model in sorted(df['model'].unique()):
        sub = df[df['model'] == model][['essay_id', 'run', 'total_score']].dropna()
        sub['run'] = pd.to_numeric(sub['run'], errors='coerce')
        sub['total_score'] = pd.to_numeric(sub['total_score'], errors='coerce')
        sub = sub.dropna(subset=['essay_id', 'run', 'total_score'])

        # Collapse duplicates per (essay, run).
        sub = sub.groupby(['essay_id', 'run'], as_index=False)['total_score'].mean()

        n_raters = int(sub['run'].nunique()) if not sub.empty else 0
        per_essay_counts = sub.groupby('essay_id').size()
        balanced_essays = per_essay_counts[per_essay_counts == n_raters].index
        sub_bal = sub[sub['essay_id'].isin(balanced_essays)]

        icc_val = None
        if not sub_bal.empty:
            icc_table = pg.intraclass_corr(
                data=sub_bal,
                targets='essay_id',
                raters='run',
                ratings='total_score'
            )
            pick = icc_table[icc_table['Type'].str.contains('ICC3')]
            if pick.empty:
                pick = icc_table.iloc[0]
            else:
                pick = pick.iloc[0]
            icc_val = float(pick['ICC'])

        out[model] = {
            'icc_total': icc_val,
            'n_essays_total': int(sub['essay_id'].nunique()) if not sub.empty else 0,
            'n_essays_balanced': int(sub_bal['essay_id'].nunique()) if not sub_bal.empty else 0,
            'n_raters': n_raters
        }
    return out


def compute_severity_ttest(essay_metrics):
    models = sorted(essay_metrics['model'].unique())
    if len(models) < 2:
        return None
    m1, m2 = models[0], models[1]
    left = essay_metrics[essay_metrics['model'] == m1][['essay_id', 'mean_total']]
    right = essay_metrics[essay_metrics['model'] == m2][['essay_id', 'mean_total']]
    merged = left.merge(right, on='essay_id', suffixes=('_m1', '_m2'))
    merged = merged.dropna(subset=['mean_total_m1', 'mean_total_m2'])
    if merged.empty:
        return None
    diffs = merged['mean_total_m1'] - merged['mean_total_m2']
    if diffs.nunique() <= 1:
        return {
            'model_1': m1,
            'model_2': m2,
            'tstat': 0.0,
            'pvalue': 1.0,
            'n_pairs': int(len(merged))
        }
    tstat, p = stats.ttest_rel(merged['mean_total_m1'], merged['mean_total_m2'], nan_policy='omit')
    return {
        'model_1': m1,
        'model_2': m2,
        'tstat': float(tstat),
        'pvalue': float(p),
        'n_pairs': int(len(merged))
    }


def compute_topic_bias(essay_metrics):
    topic_means = essay_metrics.groupby(['model', 'prompt_category']).agg(
        topic_mean=('mean_total', 'mean')
    ).reset_index()
    return topic_means


def compute_category_questions(essay_metrics, task_image):
    df = essay_metrics[['prompt_category', 'prompt_question']].dropna().drop_duplicates()
    if df.empty:
        empty = pd.DataFrame(columns=['prompt_category', 'questions', 'visual'])
        return empty, empty

    grouped = (
        df.groupby('prompt_category')['prompt_question']
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index(name='questions')
    )
    grouped['questions'] = grouped['questions'].apply(lambda qs: '\n'.join(qs))

    md = grouped.copy()
    html = grouped.copy()
    md['visual'] = ''
    html['visual'] = ''

    if task_image:
        mask = md['prompt_category'] == 'data_report'
        md.loc[mask, 'visual'] = f'![task_1]({task_image})'
        html.loc[mask, 'visual'] = (
            f'<img class="zoomable" src="{task_image}" alt="task_1" '
            'style="max-width:220px;border-radius:8px;border:1px solid #e5e7eb;">'
        )

    return md, html


def apply_hitl(essay_metrics, cv_thresh, range_thresh):
    essay_metrics = essay_metrics.copy()
    essay_metrics['hitl_status'] = np.where(
        (essay_metrics['cv_total'] < cv_thresh) & (essay_metrics['range_total'] <= range_thresh),
        'Auto-Approve',
        'Flagged for Review'
    )
    dist = (
        essay_metrics.groupby(['model', 'hitl_status']).size()
        .reset_index(name='count')
    )
    totals = essay_metrics.groupby('model').size().reset_index(name='total')
    dist = dist.merge(totals, on='model')
    dist['pct'] = (dist['count'] / dist['total']) * 100.0
    return essay_metrics, dist


def build_boxplot_fig(essay_metrics):
    # Color palette for different models
    colors = ['indianred', 'lightseagreen', 'royalblue', 'orange', 'mediumpurple', 'forestgreen']
    models = sorted(essay_metrics['model'].unique())
    
    fig = go.Figure()
    for i, model in enumerate(models):
        model_data = essay_metrics[essay_metrics['model'] == model]['mean_total']
        fig.add_trace(go.Box(
            y=model_data,
            name=model,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='Score Distribution by Model',
        yaxis_title='Mean Total Score'
    )
    return fig


def _heatmap_text(z):
    text = []
    for row in z:
        row_text = []
        for val in row:
            if val is None:
                row_text.append('')
            else:
                row_text.append(f'{val:.2f}')
        text.append(row_text)
    return text


def _build_heatmap(z, x, y, title, colorscale, zmin=None, zmax=None, zmid=None):
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=_heatmap_text(z),
            texttemplate='%{text}',
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=zmid,
            showscale=True,
            hoverongaps=False
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='Prompt Category',
        yaxis_title='Model'
    )
    return fig


def build_heatmap_variants(topic_means):
    pivot = topic_means.pivot(index='model', columns='prompt_category', values='topic_mean')
    if pivot.empty:
        z = [[0.0]]
        x = ['no_data']
        y = ['no_data']
        raw_fig = _build_heatmap(z, x, y, 'Topic Bias Heatmap (Mean Scores)', 'Viridis', zmin=0.0, zmax=1.0)
        delta_fig = _build_heatmap(z, x, y, 'Topic Bias Heatmap (Delta vs Model Mean)', 'RdBu', zmin=-1.0, zmax=1.0, zmid=0.0)
        return raw_fig, delta_fig

    values = pivot.astype(float)
    raw = values.where(values.notna(), None)
    x = values.columns.tolist()
    y = values.index.tolist()

    valid = values.stack()
    zmin = float(valid.min()) if not valid.empty else 0.0
    zmax = float(valid.max()) if not valid.empty else 1.0
    if zmin == zmax:
        zmax = zmin + 1.0

    raw_fig = _build_heatmap(
        raw.values.tolist(),
        x,
        y,
        'Topic Bias Heatmap (Mean Scores)',
        'Viridis',
        zmin=zmin,
        zmax=zmax
    )

    # Delta heatmap: deviation from model mean
    row_means = values.mean(axis=1)
    delta = values.sub(row_means, axis=0)
    delta_valid = delta.stack()
    dmin = float(delta_valid.min()) if not delta_valid.empty else -1.0
    dmax = float(delta_valid.max()) if not delta_valid.empty else 1.0
    if dmin == dmax:
        dmax = dmin + 1.0

    delta_fig = _build_heatmap(
        delta.where(delta.notna(), None).values.tolist(),
        x,
        y,
        'Topic Bias Heatmap (Delta vs Model Mean)',
        'RdBu',
        zmin=dmin,
        zmax=dmax,
        zmid=0.0
    )

    return raw_fig, delta_fig


def build_bland_altman_fig(essay_metrics):
    models = sorted(essay_metrics['model'].unique())
    if len(models) < 2:
        return None
    m1, m2 = models[0], models[1]
    left = essay_metrics[essay_metrics['model'] == m1][['essay_id', 'mean_total']]
    right = essay_metrics[essay_metrics['model'] == m2][['essay_id', 'mean_total']]
    merged = left.merge(right, on='essay_id', suffixes=('_m1', '_m2'))
    if merged.empty:
        return None
    mean_pair = (merged['mean_total_m1'] + merged['mean_total_m2']) / 2.0
    diff = merged['mean_total_m1'] - merged['mean_total_m2']
    mean_diff = diff.mean()
    sd_diff = diff.std()
    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_pair,
        y=diff,
        mode='markers',
        name='Pairs'
    ))
    fig.add_hline(y=mean_diff, line_dash='dash', line_color='red', annotation_text='Mean Diff')
    fig.add_hline(y=loa_upper, line_dash='dash', line_color='gray', annotation_text='Upper LoA')
    fig.add_hline(y=loa_lower, line_dash='dash', line_color='gray', annotation_text='Lower LoA')
    fig.update_layout(
        title='Bland-Altman Plot',
        xaxis_title='Mean of Two Models',
        yaxis_title=f'Difference ({m1} - {m2})',
        xaxis_tickangle=-30
    )
    return fig


def build_hitl_fig(hitl_dist):
    fig = px.pie(
        hitl_dist,
        values='count',
        names='hitl_status',
        facet_col='model',
        title='HITL Distribution by Model'
    )
    fig.update_traces(textinfo='percent+label')
    return fig


def build_metrics_summary(essay_metrics, icc_map, hitl_dist):
    summary = essay_metrics.groupby('model').agg(
        n_essays=('essay_id', 'count'),
        mean_total=('mean_total', 'mean'),
        sd_total=('mean_total', 'std'),
        mean_sd=('sd_total', 'mean'),
        mean_range=('range_total', 'mean'),
        mean_cv=('cv_total', 'mean')
    ).reset_index()

    icc_rows = []
    for model, stats_row in icc_map.items():
        icc_rows.append({
            'model': model,
            'icc_total': stats_row.get('icc_total'),
            'icc_n_essays_total': stats_row.get('n_essays_total'),
            'icc_n_essays_balanced': stats_row.get('n_essays_balanced'),
            'icc_n_raters': stats_row.get('n_raters')
        })
    icc_df = pd.DataFrame(icc_rows)
    summary = summary.merge(icc_df, on='model', how='left')

    hitl_pivot = hitl_dist.pivot(index='model', columns='hitl_status', values='pct').reset_index()
    hitl_pivot.columns = [c.replace(' ', '_').lower() if isinstance(c, str) else c for c in hitl_pivot.columns]
    summary = summary.merge(hitl_pivot, on='model', how='left')
    return summary


def write_report_md(out_dir, metrics_summary, ttest_res, prompt_note, topic_means, hitl_dist, plots, category_questions, task_image):
    lines = []
    lines.append('# Analysis Report')
    lines.append('')
    lines.append('## Inputs')
    lines.append(f'- prompt_category source: {prompt_note}')
    lines.append('')

    lines.append('## Model Metrics')
    lines.append('')
    lines.append(metrics_summary.to_markdown(index=False))
    lines.append('')

    lines.append('## Severity Bias (Paired t-test)')
    if ttest_res:
        lines.append(f"- Model 1: {ttest_res['model_1']}")
        lines.append(f"- Model 2: {ttest_res['model_2']}")
        lines.append(f"- t={ttest_res['tstat']:.4f}, p={ttest_res['pvalue']:.6f}, n_pairs={ttest_res['n_pairs']}")
    else:
        lines.append('- Not enough paired essays to compute the test.')
    lines.append('')

    lines.append('## Topic Bias (Mean Total by Prompt Category)')
    lines.append(topic_means.head(20).to_markdown(index=False))
    lines.append('')

    lines.append('## Category → Question Map')
    if category_questions.empty:
        lines.append('No prompt questions found in input data.')
    else:
        lines.append(category_questions.to_markdown(index=False))
    lines.append('')

    lines.append('## HITL Distribution')
    lines.append(hitl_dist.to_markdown(index=False))
    lines.append('')

    lines.append('## Plots')
    if not plots:
        lines.append('Interactive Plotly charts are available in report.html.')
    else:
        for key, path in plots.items():
            lines.append(f'- {key}: {path}')
        lines.append('')
        for key, path in plots.items():
            lines.append(f'![{key}]({path})')
    lines.append('')

    out_path = Path(out_dir) / 'report.md'
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def write_report_html(out_dir, metrics_summary, ttest_res, prompt_note, topic_means, hitl_dist, plots, category_questions, task_image, fig_jsons):
    html = []
    html.append('<!doctype html>')
    html.append('<html><head><meta charset="utf-8"><title>Analysis Dashboard</title>')
    html.append('<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>')
    html.append('<style>')
    html.append('@import url("https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap");')
    html.append(':root{--bg:#f6f7fb;--card:#ffffff;--text:#111827;--muted:#6b7280;--border:#e5e7eb;--accent:#2563eb;}')
    html.append('[data-theme="dark"]{--bg:#0f172a;--card:#111827;--text:#e5e7eb;--muted:#94a3b8;--border:#1f2937;--accent:#60a5fa;}')
    html.append('body{font-family:"Source Sans 3", system-ui, sans-serif; margin:0; background:var(--bg); color:var(--text);}')
    html.append('header{padding:24px 0; background:linear-gradient(135deg,var(--card),var(--bg)); border-bottom:1px solid var(--border);}')
    html.append('.container{max-width:1100px; margin:0 auto; padding:0 20px;}')
    html.append('h1{font-family:"Fraunces", serif; margin:0 0 6px; font-size:32px;}')
    html.append('h2{margin:24px 0 8px; font-size:18px;}')
    html.append('.meta{color:var(--muted); font-size:13px;}')
    html.append('.toolbar{display:flex; gap:12px; align-items:center; justify-content:space-between;}')
    html.append('.btn{background:var(--card); border:1px solid var(--border); padding:8px 12px; border-radius:8px; cursor:pointer; color:var(--text);}')
    html.append('.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-top:16px;}')
    html.append('.stack{display:flex; flex-direction:column; gap:16px; margin-top:16px;}')
    html.append('.card{background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px;}')
    html.append('.kpi{font-size:20px; font-weight:600;}')
    html.append('.kpi-label{color:var(--muted); font-size:12px;}')
    html.append('.chart{height:420px;}')
    html.append('.table-wrap{overflow-x:auto; background:var(--card); border:1px solid var(--border); border-radius:12px; padding:10px;}')
    html.append('table{border-collapse:collapse; width:100%; font-size:14px;}')
    html.append('th,td{border:1px solid var(--border); padding:8px 10px; vertical-align:top;}')
    html.append('th{background:rgba(0,0,0,0.02); text-align:left;}')
    html.append('img.zoomable{max-width:220px; border:1px solid var(--border); border-radius:8px; cursor:zoom-in;}')
    html.append('.lightbox{position:fixed; inset:0; background:rgba(0,0,0,0.75); display:none; align-items:center; justify-content:center; z-index:9999;}')
    html.append('.lightbox img{max-width:92vw; max-height:92vh; border-radius:12px; cursor:zoom-out;}')
    html.append('.lightbox.active{display:flex;}')
    html.append('.theme-icon{width:16px; height:16px; vertical-align:middle;}')
    html.append('.theme-icon.moon{display:none;}')
    html.append('[data-theme="dark"] .theme-icon.sun{display:none;}')
    html.append('[data-theme="dark"] .theme-icon.moon{display:inline-block;}')
    html.append('@media (max-width:720px){.chart{height:300px;} h1{font-size:26px;} }')
    html.append('</style>')
    html.append('</head><body>')

    html.append('<header>')
    html.append('<div class="container toolbar">')
    html.append('<div>')
    html.append('<h1>Analysis Dashboard</h1>')
    html.append(f'<div class="meta">prompt_category source: {prompt_note}</div>')
    html.append('</div>')
    html.append('<button id="theme-toggle" class="btn" aria-label="Toggle theme">')
    html.append('<svg class="theme-icon sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="4"/><line x1="12" y1="20" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="6.34" y2="6.34"/><line x1="17.66" y1="17.66" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="4" y2="12"/><line x1="20" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="6.34" y2="17.66"/><line x1="17.66" y1="6.34" x2="19.07" y2="4.93"/></svg>')
    html.append('<svg class="theme-icon moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>')
    html.append('<span style="margin-left:8px;">Theme</span></button>')
    html.append('</div></header>')

    html.append('<main class="container">')

    # KPI cards
    html.append('<section>')
    html.append('<h2>Model KPIs</h2>')
    html.append('<div class="grid">')
    for _, row in metrics_summary.iterrows():
        model = row['model']
        html.append('<div class="card">')
        html.append(f'<div class="meta">{model}</div>')
        html.append(f'<div class="kpi">{row["mean_total"]:.2f}</div>')
        html.append('<div class="kpi-label">Mean Total Score</div>')
        html.append(f'<div class="kpi" style="margin-top:10px;">{row.get("icc_total", 0):.3f}</div>')
        html.append('<div class="kpi-label">ICC Total</div>')
        html.append(f'<div class="kpi" style="margin-top:10px;">{row.get("auto-approve", 0):.1f}%</div>')
        html.append('<div class="kpi-label">Auto-Approve</div>')
        html.append('</div>')
    html.append('</div></section>')

    # Charts
    html.append('<section>')
    html.append('<h2>Visualizations</h2>')
    html.append('<div class="stack">')
    html.append('<div class="card"><div class="meta">Boxplot</div><div id="chart-boxplot" class="chart"></div></div>')
    html.append('<div class="card"><div class="meta">Topic Heatmap (Mean)</div><div id="chart-heatmap-raw" class="chart"></div></div>')
    html.append('<div class="card"><div class="meta">Topic Heatmap (Delta vs Model Mean)</div><div id="chart-heatmap-delta" class="chart"></div></div>')
    html.append('<div class="card"><div class="meta">Bland-Altman</div><div id="chart-bland" class="chart"></div></div>')
    html.append('<div class="card"><div class="meta">HITL Distribution</div><div id="chart-hitl" class="chart"></div></div>')
    html.append('</div></section>')

    # Tables
    html.append('<section>')
    html.append('<h2>Model Metrics</h2>')
    html.append('<div class="table-wrap">')
    html.append(metrics_summary.to_html(index=False))
    html.append('</div>')
    html.append('</section>')

    html.append('<section>')
    html.append('<h2>Severity Bias (Paired t-test)</h2>')
    if ttest_res:
        html.append('<div class="card">')
        html.append(f'<div>Model 1: {ttest_res["model_1"]}</div>')
        html.append(f'<div>Model 2: {ttest_res["model_2"]}</div>')
        html.append(f'<div>t={ttest_res["tstat"]:.4f}, p={ttest_res["pvalue"]:.6f}, n_pairs={ttest_res["n_pairs"]}</div>')
        html.append('</div>')
    else:
        html.append('<div class="card">Not enough paired essays to compute the test.</div>')
    html.append('</section>')

    html.append('<section>')
    html.append('<h2>Topic Bias (Mean Total by Prompt Category)</h2>')
    html.append('<div class="table-wrap">')
    html.append(topic_means.to_html(index=False))
    html.append('</div>')
    html.append('</section>')

    html.append('<section>')
    html.append('<h2>Category → Question Map</h2>')
    if category_questions.empty:
        html.append('<div class="card">No prompt questions found in input data.</div>')
    else:
        html.append('<div class="table-wrap">')
        html.append(category_questions.to_html(index=False, escape=False))
        html.append('</div>')
    html.append('</section>')

    html.append('<section>')
    html.append('<h2>HITL Distribution</h2>')
    html.append('<div class="table-wrap">')
    html.append(hitl_dist.to_html(index=False))
    html.append('</div>')
    html.append('</section>')

    html.append('</main>')

    # Lightbox for images
    html.append('<div id="lightbox" class="lightbox"><img id="lightbox-img" src="" alt="zoom"></div>')

    # Plotly + theme script
    html.append('<script>')
    html.append(f'const chartData = {json.dumps(fig_jsons, cls=PlotlyJSONEncoder)};')
    html.append('function applyCharts(theme){')
    html.append('  const template = theme === "dark" ? "plotly_dark" : "plotly_white";')
    html.append('  const grid = theme === "dark" ? "#334155" : "#e5e7eb";')
    html.append('  const text = theme === "dark" ? "#e5e7eb" : "#111827";')
    html.append('  const base = {template, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)", font:{color:text}};')
    html.append('  const cfg = {displayModeBar:false, responsive:true};')
    html.append('  function mergeLayout(layout){')
    html.append('    const xaxis = {...(layout.xaxis||{}), gridcolor:grid, color:text};')
    html.append('    const yaxis = {...(layout.yaxis||{}), gridcolor:grid, color:text};')
    html.append('    const coloraxis = {...(layout.coloraxis||{}), colorbar:{...(layout.coloraxis?.colorbar||{}), tickfont:{color:text}}};')
    html.append('    return {...layout, ...base, xaxis, yaxis, coloraxis};')
    html.append('  }')
    html.append('  if(chartData.boxplot){Plotly.react("chart-boxplot", chartData.boxplot.data, mergeLayout(chartData.boxplot.layout), cfg);}')
    html.append('  if(chartData.heatmap_raw){Plotly.react("chart-heatmap-raw", chartData.heatmap_raw.data, mergeLayout(chartData.heatmap_raw.layout), cfg);}')
    html.append('  if(chartData.heatmap_delta){Plotly.react("chart-heatmap-delta", chartData.heatmap_delta.data, mergeLayout(chartData.heatmap_delta.layout), cfg);}')
    html.append('  if(chartData.bland_altman){Plotly.react("chart-bland", chartData.bland_altman.data, mergeLayout(chartData.bland_altman.layout), cfg);}')
    html.append('  if(chartData.hitl){Plotly.react("chart-hitl", chartData.hitl.data, mergeLayout(chartData.hitl.layout), cfg);}')
    html.append('}')
    html.append('const saved = localStorage.getItem("theme") || "light";')
    html.append('document.body.dataset.theme = saved; applyCharts(saved);')
    html.append('document.getElementById("theme-toggle").addEventListener("click",()=>{')
    html.append('  const next = document.body.dataset.theme === "dark" ? "light" : "dark";')
    html.append('  document.body.dataset.theme = next; localStorage.setItem("theme", next); applyCharts(next);');
    html.append('});')
    html.append('const lb=document.getElementById("lightbox");const lbImg=document.getElementById("lightbox-img");')
    html.append('document.querySelectorAll("img.zoomable").forEach(img=>{img.addEventListener("click",()=>{lbImg.src=img.src;lb.classList.add("active");});});')
    html.append('lb.addEventListener("click",()=>{lb.classList.remove("active");});')
    html.append('</script>')
    html.append('</body></html>')

    out_path = Path(out_dir) / 'report.html'
    out_path.write_text('\n'.join(html), encoding='utf-8')


def write_report_json(out_dir, metrics_summary, ttest_res, prompt_note, topic_means, hitl_dist):
    """Write numerical metrics to JSON file (matching markdown tables)."""
    # Helper to convert DataFrame to dict with native Python types
    def df_to_records(df):
        records = df.to_dict(orient='records')
        for row in records:
            for key, val in row.items():
                if hasattr(val, 'item'):  # numpy types
                    row[key] = val.item()
                elif pd.isna(val):
                    row[key] = None
        return records

    report = {
        'prompt_category_source': prompt_note,
        'metrics_summary': df_to_records(metrics_summary),
        'severity_bias_ttest': ttest_res,
        'topic_bias_means': df_to_records(topic_means),
        'hitl_distribution': df_to_records(hitl_dist)
    }

    out_path = Path(out_dir) / 'report.json'
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'JSON report saved to {out_path}')


def save_fig_as_png(fig, filepath, width=1200, height=600):
    """Save a Plotly figure as PNG using kaleido."""
    try:
        fig.write_image(filepath, width=width, height=height, scale=2)
        print(f'Saved PNG: {filepath}')
    except Exception as e:
        print(f'Warning: Could not save PNG to {filepath}: {e}')
        print('Make sure kaleido is installed: pip install kaleido')


def parse_args():
    p = argparse.ArgumentParser(description='Automated essay scoring analysis')
    p.add_argument('--inputs', required=True, help='Path to eval folder or checkpoint.jsonl')
    p.add_argument('--out-dir', default='english-bias-framework/data/processed/analysis', help='Output folder')
    p.add_argument('--hitl-cv-threshold', type=float, default=0.15)
    p.add_argument('--hitl-range-threshold', type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / 'image'
    image_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(args.inputs)
    df = expand_scores(df)
    sample_map, prompt_note = load_sample_mapping(args.inputs)
    df, prompt_note = align_inputs(df, sample_map)

    essay_metrics = compute_essay_metrics(df)
    icc_map = compute_model_icc(df)
    ttest_res = compute_severity_ttest(essay_metrics)
    topic_means = compute_topic_bias(essay_metrics)
    task_image = None
    for candidate in [out_dir / 'image' / 'task_1.png', out_dir / 'task_1.png']:
        if candidate.exists():
            task_image = str(candidate.relative_to(out_dir))
            break

    category_questions_md, category_questions_html = compute_category_questions(essay_metrics, task_image)
    essay_metrics, hitl_dist = apply_hitl(
        essay_metrics,
        cv_thresh=args.hitl_cv_threshold,
        range_thresh=args.hitl_range_threshold
    )

    metrics_summary = build_metrics_summary(essay_metrics, icc_map, hitl_dist)

    # Save outputs
    metrics_summary.to_csv(out_dir / 'metrics_summary.csv', index=False)
    essay_metrics.to_csv(out_dir / 'hitl_routing.csv', index=False)

    # Plotly figures
    fig_box = build_boxplot_fig(essay_metrics)
    fig_heatmap_raw, fig_heatmap_delta = build_heatmap_variants(topic_means)
    fig_ba = build_bland_altman_fig(essay_metrics)
    fig_hitl = build_hitl_fig(hitl_dist)

    # Save figures as PNG
    if fig_box:
        save_fig_as_png(fig_box, image_dir / 'boxplot.png')
    if fig_heatmap_raw:
        save_fig_as_png(fig_heatmap_raw, image_dir / 'heatmap_raw.png')
    if fig_heatmap_delta:
        save_fig_as_png(fig_heatmap_delta, image_dir / 'heatmap_delta.png')
    if fig_ba:
        save_fig_as_png(fig_ba, image_dir / 'bland_altman.png')
    if fig_hitl:
        save_fig_as_png(fig_hitl, image_dir / 'hitl.png')

    # PNG paths for report
    plots = {}
    for name in ['boxplot', 'heatmap_raw', 'heatmap_delta', 'bland_altman', 'hitl']:
        png_path = image_dir / f'{name}.png'
        if png_path.exists():
            plots[name] = str(png_path.relative_to(out_dir))

    # Prepare JSON for HTML report
    fig_jsons = {
        'boxplot': fig_box.to_plotly_json() if fig_box else None,
        'heatmap_raw': fig_heatmap_raw.to_plotly_json() if fig_heatmap_raw else None,
        'heatmap_delta': fig_heatmap_delta.to_plotly_json() if fig_heatmap_delta else None,
        'bland_altman': fig_ba.to_plotly_json() if fig_ba else None,
        'hitl': fig_hitl.to_plotly_json() if fig_hitl else None
    }

    # Reports
    write_report_md(out_dir, metrics_summary, ttest_res, prompt_note, topic_means, hitl_dist, plots, category_questions_md, task_image)
    write_report_html(out_dir, metrics_summary, ttest_res, prompt_note, topic_means, hitl_dist, plots, category_questions_html, task_image, fig_jsons)

    # Write JSON report (numerical metrics only)
    write_report_json(out_dir, metrics_summary, ttest_res, prompt_note, topic_means, hitl_dist)

    print(f'Analysis completed. Outputs saved to {out_dir}')


if __name__ == '__main__':
    main()
