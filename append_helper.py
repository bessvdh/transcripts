# append_helper.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

EMPTY_SET = {'', 'na', 'n/a', '-', '—', '–', 'null', 'none'}


def _norm_key(k: object) -> str:
    s = re.sub(r"\D", "", str(k or ''))
    if not s:
        return ''
    return (s[-4:]).zfill(4)


def _is_blank(x) -> bool:
    s = str(x).strip().lower()
    return s in EMPTY_SET or s == ''


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _try_make_key_from_filename(df: pd.DataFrame) -> Optional[pd.Series]:
    cand = _first_present(df, ['filename', 'file', 'source', 'source_file', 'path'])
    if not cand:
        return None
    ser = df[cand].astype(str).fillna('')
    return ser.apply(lambda s: _norm_key(s))


def _pack_modules_df(mod_df: pd.DataFrame) -> Dict[str, str]:
    if mod_df is None or mod_df.empty:
        return {}
    df = mod_df.copy()

    key_col = _first_present(df, ['application_key', 'app_key', 'key', 'id', 'Application Key', 'Applicant Key'])
    if key_col is None:
        derived = _try_make_key_from_filename(df)
        if derived is None:
            return {}
        df['__appkey__'] = derived
        key_col = '__appkey__'
    df[key_col] = df[key_col].apply(_norm_key)

    subj_col = _first_present(df, ['subject', 'dept', 'department', 'course_subject'])
    num_col = _first_present(df, ['number', 'course_number', 'catalog', 'catalog_number', 'cat_num'])
    title_col = _first_present(df, ['title', 'course_title', 'name'])
    grade_col = _first_present(df, ['grade', 'final_grade', 'mark', 'score'])
    scale_col = _first_present(df, ['grade_scale', 'scale'])
    credits_col = _first_present(df, ['credits', 'credit', 'units', 'ects'])
    cu_col = _first_present(df, ['credit_unit', 'unit', 'units_label'])
    term_col = _first_present(df, ['term', 'semester', 'session', 'term_name'])
    pf_col = _first_present(df, ['pf', 'pass_fail', 'passfail', 'is_pf'])

    if subj_col is None: df['__subject__'] = ''; subj_col = '__subject__'
    if num_col is None: df['__number__'] = ''; num_col = '__number__'
    if title_col is None: df['__title__'] = ''; title_col = '__title__'
    if grade_col is None: df['__grade__'] = ''; grade_col = '__grade__'
    if scale_col is None: df['__scale__'] = ''; scale_col = '__scale__'
    if credits_col is None: df['__cr__'] = ''; credits_col = '__cr__'
    if cu_col is None: df['__cu__'] = 'credits'; cu_col = '__cu__'
    if term_col is None: df['__term__'] = ''; term_col = '__term__'
    if pf_col is None: df['__pf__'] = ''; pf_col = '__pf__'

    def _norm_subj(x: Any) -> str:
        s = re.sub(r'[^A-Za-z-]', '', str(x or '')).upper()
        if s in ('EC', 'ECN'): return 'ECON'
        if s in ('APMA', 'AM'): return 'MATH'
        if s == 'STS': return 'STAT'  # adjust if STS ≠ STAT in your data
        return s

    def _norm_pf(v: Any, grade: Any) -> bool:
        s = str(v).strip().lower()
        if s in ('1', 'true', 't', 'yes', 'y'): return True
        if s in ('0', 'false', 'f', 'no', 'n'): return False
        g = str(grade).strip().upper()
        return g in ('P', 'S', 'U', 'CR', 'NC', 'PASS', 'FAIL')

    packed: Dict[str, str] = {}
    rows: List[tuple[str, Dict[str, Any]]] = []

    for _, r in df.iterrows():
        appkey = _norm_key(r.get(key_col, ''))
        if not appkey:
            continue
        subj = _norm_subj(r.get(subj_col, ''))
        num = r.get(num_col, '')
        try:
            if str(num).strip():
                num = int(str(num).strip())
            else:
                num = ''
        except Exception:
            num = str(num).strip()

        title = str(r.get(title_col, '')).strip()
        grade = r.get(grade_col, '')
        scale = str(r.get(scale_col, '')).strip().lower() or None
        credits = r.get(credits_col, '')
        try:
            credits = float(str(credits).strip()) if str(credits).strip() else ''
        except Exception:
            credits = ''
        cu = str(r.get(cu_col, '') or 'credits')
        term = str(r.get(term_col, '')).strip()
        pf = _norm_pf(r.get(pf_col, ''), grade)

        rows.append((
            appkey,
            {
                'subject': subj,
                'number': num,
                'title': title,
                'grade': (None if pf else grade),
                'grade_scale': scale,
                'credits': credits if credits != '' else 3.0,
                'credit_unit': cu,
                'term': term,
                'pf': bool(pf),
            }
        ))

    from collections import defaultdict
    grp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for k, rec in rows:
        grp[k].append(rec)

    def key(m: Dict[str, Any]):
        return (
            str(m.get('subject', '')).upper(),
            str(m.get('number', '')).strip(),
            re.sub(r'\s+', ' ', str(m.get('title', '')).strip().lower()),
            re.sub(r'\s+', ' ', str(m.get('term', '')).strip().lower()),
            str(m.get('grade', '')).strip().upper(),
        )

    for k, recs in grp.items():
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for m in recs:
            kk = key(m)
            if kk in seen:
                continue
            seen.add(kk)
            uniq.append(m)
        packed[k] = json.dumps(uniq, ensure_ascii=False)

    return packed


def _merge_modules(a: str, b: str) -> str:
    try:
        la = json.loads(a) if a and str(a).strip() else []
    except Exception:
        la = []
    try:
        lb = json.loads(b) if b and str(b).strip() else []
    except Exception:
        lb = []

    def key(m: Dict[str, Any]):
        return (
            str(m.get('subject', '')).upper(),
            str(m.get('number', '')).strip(),
            re.sub(r'\s+', ' ', str(m.get('title', '')).strip().lower()),
            re.sub(r'\s+', ' ', str(m.get('term', '')).strip().lower()),
            str(m.get('grade', '')).strip().upper(),
        )

    seen = set()
    out: List[Dict[str, Any]] = []
    for m in la + lb:
        k = key(m)
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    return json.dumps(out, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--master', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--modules', default=None)
    p.add_argument('--merge-log', default=None)
    args = p.parse_args()

    if os.path.exists(args.master):
        dfM = pd.read_csv(args.master)
    else:
        dfM = pd.DataFrame()

    dfN = pd.read_csv(args.new)

    for df in (dfM, dfN):
        if 'application_key' not in df.columns:
            df['application_key'] = ''
        df['application_key'] = df['application_key'].apply(_norm_key)

    packed: Dict[str, str] = {}
    if args.modules and os.path.exists(args.modules):
        try:
            mod_df = pd.read_csv(args.modules)
            packed = _pack_modules_df(mod_df)
        except Exception as e:
            print(f"[warn] failed to read modules file {args.modules}: {e}", file=sys.stderr)

    if 'module_records' not in dfN.columns:
        dfN['module_records'] = ''

    for i, row in dfN.iterrows():
        k = row.get('application_key', '')
        if _is_blank(row.get('module_records', '')) and k in packed:
            dfN.at[i, 'module_records'] = packed[k]

    all_cols = list(dict.fromkeys(list(dfM.columns) + list(dfN.columns)))

    if dfM.empty:
        dfOut = dfN.copy()
    else:
        dfM = dfM.set_index('application_key', drop=False)
        dfN = dfN.set_index('application_key', drop=False)
        keys = sorted(set(dfM.index.tolist()) | set(dfN.index.tolist()))
        rows: List[Dict[str, Any]] = []
        merges: List[str] = []

        for k in keys:
            if k in dfM.index and k not in dfN.index:
                rows.append(dfM.loc[k].to_dict()); continue
            if k not in dfM.index and k in dfN.index:
                rows.append(dfN.loc[k].to_dict()); continue

            a = dfM.loc[k].to_dict()
            b = dfN.loc[k].to_dict()
            merged: Dict[str, Any] = {}
            for c in set(a.keys()) | set(b.keys()):
                va = a.get(c, '')
                vb = b.get(c, '')
                if c == 'module_records':
                    merged[c] = _merge_modules(va, vb)
                else:
                    merged[c] = vb if _is_blank(va) and not _is_blank(vb) else va
            rows.append(merged)
            merges.append(k)

        dfOut = pd.DataFrame(rows)
        if args.merge_log:
            with open(args.merge_log, 'a', encoding='utf-8') as f:
                for k in merges:
                    f.write(f"merged application_key={k}\n")

    cols = [c for c in all_cols if c != 'application_key']
    cols = ['application_key'] + cols
    dfOut = dfOut.reindex(columns=cols)
    dfOut.to_csv(args.out, index=False)
    print(f"Wrote {args.out} (rows={len(dfOut)})")


if __name__ == '__main__':
    main()
