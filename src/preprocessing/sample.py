"""Wrapper module exposing sampling functions under `preprocessing.sample`.

Re-exports sampling helpers from `sample_questions.py`.
"""
from preprocessing.sample_impl import detect_essay_question_ids, sample_per_question

__all__ = ["detect_essay_question_ids", "sample_per_question"]
