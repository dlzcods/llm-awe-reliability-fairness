import re
import hashlib
from typing import Optional


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # Replace common tokens
    s = s.replace("@CAPS1", "").replace("@CAPS2", "")
    # Remove non-printable/control chars
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    # Unescape basic HTML entities (minimal)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def anonymize_id(value: Optional[str], salt: str, length: int = 12) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    h = hashlib.sha256()
    h.update((value + salt).encode("utf-8"))
    return h.hexdigest()[:length]


def detect_category(question_text: Optional[str]) -> str:
    if not question_text:
        return "unknown"
    q = question_text.lower()
    # Data report
    if any(k in q for k in ["chart", "charts", "pie", "bar chart", "describe the information", "write a report"]):
        return "data_report"
    # Social policy opinion
    if any(k in q for k in ["government", "retire", "retired", "pension", "support", "take care of themselves"]):
        return "social_policy_opinion"
    # Tech & society opinion
    if any(k in q for k in ["robot", "robots", "technology", "dangerous", "future", "society"]):
        return "tech_society_opinion"
    # Fallback heuristics
    if any(k in q for k in ["discuss", "opinion", "give your opinion"]):
        return "opinion"
    return "other"


def word_count(s: Optional[str]) -> int:
    if not s:
        return 0
    return len(normalize_text(s).split())


def contains_indonesian(s: Optional[str]) -> bool:
    if not s:
        return False
    t = normalize_text(s).lower()
    indon_words = [
        'dan', 'yang', 'tidak', 'ada', 'saya', 'kamu', 'kita', 'kami', 'untuk', 'dengan', 'di', 'ke', 'oleh',
        'ini', 'itu', 'sebuah', 'adalah', 'atau', 'pada', 'dari'
    ]
    hits = sum(1 for w in indon_words if f' {w} ' in f' {t} ')
    return hits >= 2


def is_gibberish(s: Optional[str]) -> bool:
    if not s:
        return False
    t = s.strip()
    non_alnum = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
    if len(t) == 0:
        return True
    if non_alnum / max(1, len(t)) > 0.3:
        return True
    if re.search(r'(?:qwerty|asdf|zxcv|hjkl|11111|aaaaa|\\W{5,})', t.lower()):
        return True
    tokens = t.split()
    single_letters = sum(1 for tok in tokens if len(tok) == 1)
    if tokens and (single_letters / len(tokens)) > 0.4:
        return True
    return False


def is_noisy_text(s: Optional[str]) -> bool:
    if is_gibberish(s):
        return True
    try:
        from lingua import Language, LanguageDetectorBuilder
        if not hasattr(is_noisy_text, '_detector') or is_noisy_text._detector is None:
            is_noisy_text._detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.INDONESIAN).build()
        detected = is_noisy_text._detector.detect_language_of(s)
        if detected == Language.INDONESIAN:
            return True
    except Exception:
        if contains_indonesian(s):
            return True
    return False
