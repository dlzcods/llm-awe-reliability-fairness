"""Wrapper module exposing cleaning functions under `preprocessing.clean`.

This module imports the existing cleaning implementation from `clean.py` and
re-exports the public helpers so other code can import from
`preprocessing.clean` for a clearer layout.
"""
from preprocessing.clean_impl import process as _process_impl, hashlib_sha
from utils.ioutils import read_xlsx
from utils.config import DEFAULTS


def process(df, args):
	"""Compatibility wrapper: supply DEFAULTS to the implementation when not provided."""
	return _process_impl(df, args, defaults=DEFAULTS)


__all__ = ["read_xlsx", "process", "hashlib_sha"]
