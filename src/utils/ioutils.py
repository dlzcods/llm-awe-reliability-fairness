import pandas as pd
import json
from pathlib import Path
from typing import List


def read_xlsx(path: str, sheet_name: str = None):
    xl = pd.ExcelFile(path)
    if sheet_name is None:
        sheet_name = xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet_name)
    return df


def write_jsonl(records: List[dict], out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
