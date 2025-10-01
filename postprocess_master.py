# postprocess_master.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import html
import math
from typing import Optional, Any, Dict

import pandas as pd

# Your conversion rules module
from scale_rules import apply_conversion, INSTITUTION_HINTS


# ==========================================================
# Column model / constants
# ==========================================================
COLUMN_ORDER = [
    'application_key','student_name','university_name','major_most_recent',
    'degree_most_recent','degree_awarded_date','degree_classification',
    'num_semesters_with_grades',
    'num_courses_graded_econ','num_courses_pf_econ','avg_gpa_econ',
    'num_courses_graded_stats','num_courses_pf_stats','avg_gpa_stats',
    'num_courses_graded_math','num_courses_pf_math','avg_gpa_math',
    'cum_gpa_most_recent','converted gpa','conversion_rule',
    'highest_econ_course_full_title',
    'institution_1_name','institution_1_dates','institution_1_degree',
    'institution_2_name','institution_2_dates','institution_2_degree',
    'date_of_birth','merged_from_application_keys',
    'scale_hint','country_hint'
]

SUFFIXES = {"Jr", "Sr", "II", "III", "IV"}
NAME_COMMA_RE = re.compile(r",\s*")
NOISE_TOKENS = (
    'birth date','university','registrar','degree awarded',
    'requirements are accepted','transcript','unofficial',
    'tsrpt','app ssr','ssr','report','official','student no','student id'
)
MONTHS = {'january','february','march','april','may','june','july','august',
          'september','october','november','december'}

# A blocky UVA segment header some exports include
UVA_SEGMENT_PAT = re.compile(
    r'^(Primary\s+College:\nCollege\s*&\s*Graduate\s*Arts\s*&\s*Sci\nUniversity\s+\nSeminar)',
    re.IGNORECASE
)

# Admin/header phrases that are NOT institutions
_UVA_NOISE = re.compile(
    r'(?i)\b('
    r'College\s*&\s*Graduate\s*Arts\s*&\s*Sci(?:ences)?|'
    r'University\s*Seminar|'
    r'Primary\s*College|'
    r'Intro\s*to\s*College|'
    r'Intro\s*College|'
    r'Advising\s*Seminars?|'
    r'Office\s*of\s*the\s*University|'
    r'The\s*Goods\s*of\s*the\s*University|'
    r'University\s*Singers'            # choir/ensemble, not an institution
    r')\b'
)

# Generic institution keywords we accept
_INSTITUTION_KEYWORD = re.compile(r'\b(University|Community\s+College|College|Institute)\b', re.IGNORECASE)

# JSON fragment detector — used ONLY on name fields (student/university), not on module_records
JSON_FRAGMENT_RE = re.compile(r'"\s*(grade|subject|title|number)\s*":|\{\s*"?subject"?\s*:', re.IGNORECASE)

# Very specific spill pattern seen in the bad row (e.g., "0 KINE"" ...")
SPILLY_NAME_RE = re.compile(r'^\s*0\s+[A-Z]{2,6}["]')

# UVA hints
UVA_NAME = "University of Virginia"
UVA_MAJOR_HINT = re.compile(
    r'(?i)\b('
    r'Arts\s*&\s*Sciences\s*Undeclared|'
    r'Public\s*Policy\s*and\s*Leadership|'
    r'Foreign\s*Affairs|'
    r'EPP\s*-\s*Environmental\s*Policy\s*and\s*Planning'
    r')\b'
)
UVA_SUBJECT_HINTS = {'LPPL', 'LPPP', 'LPPS', 'LPPA', 'KINE'}  # Batten/Kinesiology subjects
UVA_ECON_NUMS = {2010, 2020, 3010, 3020}  # common UVA ECON numeration


# ==========================================================
# Small helpers
# ==========================================================
def _to_str_safe(x: Any) -> str:
    if x is None:
        return ''
    try:
        if isinstance(x, float) and math.isnan(x):
            return ''
    except Exception:
        pass
    s = str(x)
    if s.strip().lower() in ('nan','none','null'):
        return ''
    return s

def _squash_repeats_univ(s: str) -> str:
    """Collapse repeated institution names (esp. UVA repeats)."""
    s2 = re.sub(r'(?i)\b(University of Virginia)\b(?:\s+\1\b)+', r'\1', s or '')
    s2 = re.sub(r'\s{2,}', ' ', s2).strip()
    return s2

def _normalize_uva_case(s: str) -> str:
    """Canonicalize UVA name if present anywhere."""
    return UVA_NAME if re.search(r'(?i)\buniversity of virginia\b', s or '') else (s or '')

def _clean_inst_text(s: Any) -> str:
    """Clean institution names; drop administrative headers; map UVA canonical form."""
    t = html.unescape(_to_str_safe(s)).strip()
    if not t:
        return ''
    if re.search(r'(?i)\buniversity of virginia\b', t):
        t = UVA_NAME
    if _UVA_NOISE.search(t):
        return ''  # header, not an institution
    t = _squash_repeats_univ(t)
    return t[:90].rstrip()

def _sanitize(x: Any) -> str:
    s = str(x or '').strip()
    return '' if s.lower() in ('nan','none','null') else s

def normalize_name_last_first(name: str) -> str:
    """Normalize to 'Last, First [Suffix]' while avoiding obvious noise."""
    s = _sanitize(name)
    if not s:
        return s
    lo = s.lower()
    if any(tok in lo for tok in NOISE_TOKENS) or lo in MONTHS:
        return s
    if "," in s:
        parts = NAME_COMMA_RE.split(s, maxsplit=1)
        if len(parts) == 2:
            return f"{parts[0].strip()}, {parts[1].strip()}"
        return s
    parts = s.split()
    if len(parts) == 1:
        return s
    suffix = None
    if parts[-1].strip(".").title() in SUFFIXES:
        suffix = parts[-1].strip(".")
        parts = parts[:-1]
    last = parts[-1]
    first_mid = " ".join(parts[:-1])
    out = f"{last}, {first_mid}"
    if suffix:
        out += f" {suffix}"
    return out

def normalize_app_key_4(key: str) -> str:
    digits = re.sub(r"\D", "", str(key or ''))
    return digits[-4:].zfill(4) if digits else ''

def enforce_column_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in ['application_key','student_name','university_name','date_of_birth']:
        if c not in df.columns:
            df[c] = ''
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    return df[ordered + tail]


# ==========================================================
# JSON corruption cleaner
# ==========================================================
def clean_json_corruption(df: pd.DataFrame, verbose: bool = False) -> int:
    """Clean JSON data that has spilled into text fields."""
    cleaned_count = 0
    
    # Fields that should not contain JSON
    text_fields = [
        'student_name', 'university_name', 'major_most_recent', 
        'degree_most_recent', 'degree_awarded_date', 'degree_classification',
        'institution_1_name', 'institution_1_dates', 'institution_1_degree',
        'institution_2_name', 'institution_2_dates', 'institution_2_degree'
    ]
    
    json_indicators = ['"grade":', '"subject":', '"title":', '"credits":', '"number":']
    
    for field in text_fields:
        if field not in df.columns:
            continue
            
        for i, value in enumerate(df[field]):
            if pd.isna(value):
                continue
                
            value_str = str(value)
            
            # Check if this field contains JSON-like content
            if any(indicator in value_str for indicator in json_indicators):
                # Find where JSON starts
                json_start = len(value_str)
                for indicator in json_indicators:
                    pos = value_str.find(indicator)
                    if pos != -1 and pos < json_start:
                        json_start = pos
                
                # Also look for opening quotes that might indicate JSON
                quote_pos = value_str.find('"')
                if quote_pos != -1 and quote_pos < json_start:
                    json_start = quote_pos
                
                # Keep text before JSON corruption
                if json_start > 5:  # Keep meaningful text
                    clean_value = value_str[:json_start].strip()
                    # Remove trailing punctuation that might be artifacts
                    clean_value = re.sub(r'["\s]+$', '', clean_value).strip()
                    
                    if clean_value != value_str:
                        df.at[i, field] = clean_value
                        cleaned_count += 1
                        if verbose:
                            print(f"[clean] {field}[{i}]: '{value_str[:50]}...' -> '{clean_value}'")
                else:
                    # If JSON starts too early, blank the field
                    df.at[i, field] = ''
                    cleaned_count += 1
                    if verbose:
                        print(f"[clean] {field}[{i}]: Blanked JSON-corrupted field")
    
    return cleaned_count


# ==========================================================
# GPA synthesis & discipline metrics
# ==========================================================
LETTER_TO_4 = {
    'A+':4.0,'A':4.0,'A-':3.7,
    'B+':3.3,'B':3.0,'B-':2.7,
    'C+':2.3,'C':2.0,'C-':1.7,
    'D+':1.3,'D':1.0,'D-':0.7,
    'F':0.0
}

DISCIPLINES = {
    'ECON': ('num_courses_graded_econ','num_courses_pf_econ','avg_gpa_econ'),
    'STAT': ('num_courses_graded_stats','num_courses_pf_stats','avg_gpa_stats'),
    'MATH': ('num_courses_graded_math','num_courses_pf_math','avg_gpa_math'),
}

def _grade_any_to_4(grade: Any, scale: str) -> float | None:
    """Convert a letter/percent/numeric grade on assorted scales to 4.0 if possible."""
    from scale_rules import apply_conversion as conv
    if grade is None:
        return None
    if isinstance(grade, str):
        g = grade.strip().upper()
        if g in LETTER_TO_4:
            return LETTER_TO_4[g]
        if g.endswith('%'):
            try:
                pct = float(g[:-1])
                return max(0.0, min(pct/100.0*4.0, 4.0))
            except Exception:
                return None
        if g in {'P', 'S', 'CR', 'PASS'}:
            return None  # exclude P/S/CR from average
        if g in {'U', 'F', 'FAIL'}:
            return 0.0
        try:
            v = float(g)
            grade = v
        except Exception:
            return None
    try:
        v = float(grade)
    except Exception:
        return None
    if 0.0 <= v <= 4.0:
        return max(0.0, min(v, 4.0))
    out, _ = conv(v, scale_hint=(scale or '').lower() or None)
    return out

def synthesize_from_modules(mod_json_str):
    """Compute weighted average GPA from module_records when cum GPA is missing."""
    if not isinstance(mod_json_str, str) or mod_json_str.strip() == '':
        return None, None, None
    try:
        modules = json.loads(mod_json_str)
    except Exception:
        return None, None, None

    valid_modules = []
    for m in modules:
        subject = str(m.get('subject', '')).upper().strip()
        if subject in {'TERM', 'THE', 'OF', 'AND', 'AGES', 'SINCE'}:
            continue
        if m.get('grade') is None and not m.get('pf', False):
            continue
        try:
            credits = float(m.get('credits', 3.0))
            if credits > 10 or credits <= 0:
                continue
        except Exception:
            continue
        valid_modules.append(m)

    if not valid_modules:
        return None, None, None

    scales = [(m.get('grade_scale') or '').lower() for m in valid_modules if m.get('grade') is not None]
    scales = [x for x in scales if x and x != 'null']
    scale = max(set(scales), key=scales.count) if scales else 'us_4'

    ws, wt = 0.0, 0.0
    for m in valid_modules:
        g = m.get('grade')
        if g is None or m.get('pf', False):
            continue
        try:
            c = float(m.get('credits', 3.0))
        except Exception:
            c = 3.0
        if c <= 0:
            c = 3.0
        g4 = _grade_any_to_4(g, m.get('grade_scale', 'us_4'))
        if g4 is None:
            continue
        ws += g4 * c
        wt += c
    if wt == 0:
        return None, None, None

    weighting = 'ects_weighted' if any((str(m.get('credit_unit','')).upper() == 'ECTS') for m in valid_modules) else 'credit_weighted'
    return (ws/wt), scale, weighting

def apply_scale_conversion(raw_avg, scale):
    from scale_rules import apply_conversion as conv
    if raw_avg is None or scale is None:
        return None, None
    s = str(scale).lower()
    out, rule = conv(raw_avg, scale_hint=s)
    return out, s if rule is None else rule

def derive_discipline_metrics(mod_json_str: str) -> dict:
    """Compute discipline-specific counts and average GPAs from module_records."""
    out = {v: '' for d in DISCIPLINES.values() for v in d}
    if not isinstance(mod_json_str, str) or not mod_json_str.strip():
        return out
    try:
        mods = json.loads(mod_json_str)
    except Exception:
        return out

    cnt_g = {k:0 for k in DISCIPLINES}
    cnt_pf= {k:0 for k in DISCIPLINES}
    sum4  = {k:0.0 for k in DISCIPLINES}
    wts   = {k:0.0 for k in DISCIPLINES}

    for m in mods:
        subj = str(m.get('subject','')).upper().strip()
        if subj in {'TERM', 'THE', 'OF', 'AND', 'AGES', 'SINCE'}:
            continue
        if subj in ('EC','ECN') or subj.startswith('ECO'):
            subj = 'ECON'
        if subj in ('APMA','AM','MTH') or subj.startswith('MATH'):
            subj = 'MATH'
        if subj in ('STA','STS') or subj.startswith('STAT'):
            subj = 'STAT'
        if subj not in DISCIPLINES:
            continue

        try:
            credits = float(m.get('credits', 3.0))
        except Exception:
            credits = 3.0
        if credits <= 0 or credits > 10:
            credits = 3.0

        scale = m.get('grade_scale', 'us_4')
        grade = m.get('grade', None)

        # Count PF or unconvertible as PF
        if m.get('pf') or str(grade).strip().upper() in ('P','S','U','CR','NC','PASS','FAIL'):
            cnt_pf[subj] += 1
            continue

        g4 = _grade_any_to_4(grade, scale)
        if g4 is None:
            cnt_pf[subj] += 1
            continue

        cnt_g[subj] += 1
        sum4[subj]  += g4 * credits
        wts[subj]   += credits

    for subj, (n_g, n_pf, avg) in DISCIPLINES.items():
        out[n_g]  = cnt_g[subj]
        out[n_pf] = cnt_pf[subj]
        out[avg]  = round(sum4[subj]/wts[subj], 3) if wts[subj] > 0 else ''
    return out


# ==========================================================
# Highest ECON course detection
# ==========================================================
CORE_PRIORITY = {"intro": 1, "intermediate": 2, "econometrics": 3}
INTRO_PAT = re.compile(r"\b(Principles|Intro(?:duction)?)\b.*\b(Micro|Macro)\b", re.IGNORECASE)
INTER_PAT = re.compile(r"\bIntermediate\b.*\b(Micro|Macro)\b", re.IGNORECASE)
ECONMET_PAT = re.compile(r"\bEconometrics?\b", re.IGNORECASE)

def econ_core_category(title: str, number: Optional[int]) -> Optional[str]:
    t = str(title or "")
    if ECONMET_PAT.search(t): return "econometrics"
    if INTER_PAT.search(t):   return "intermediate"
    if INTRO_PAT.search(t):   return "intro"
    if number:
        if number in (2010, 2020, 101, 102): return "intro"
        if number in (3010, 3020, 201, 202): return "intermediate"
        if 3700 <= number <= 3999 or number in (371, 372): return "econometrics"
    return None


# ==========================================================
# Module packer (optional CSV)
# ==========================================================
def _pack_modules_from_csv_if_present(df: pd.DataFrame, verbose=False):
    path = '_tmp_modules.csv' if os.path.exists('_tmp_modules.csv') else None
    if not path:
        if verbose: print("[modules] no _tmp_modules.csv found to pack.")
        return
    try:
        mod = pd.read_csv(path)
    except Exception as e:
        if verbose: print(f"[modules] failed to read {path}: {e}")
        return

    def _norm_key(k: object) -> str:
        s = re.sub(r"\D", "", str(k or ''))
        return (s[-4:]).zfill(4) if s else ''

    def _first(df, cands):
        for c in cands:
            if c in df.columns: return c
        lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in lower: return lower[c.lower()]
        return None

    key_col = _first(mod, ['application_key','app_key','key','id','Application Key','Applicant Key'])
    if key_col is None:
        if verbose: print("[modules] no key column; skipping pack.")
        return

    subj = _first(mod, ['subject','dept','department','course_subject']) or ''
    num  = _first(mod, ['number','course_number','catalog','catalog_number','cat_num']) or ''
    titl = _first(mod, ['title','course_title','name']) or ''
    grad = _first(mod, ['grade','final_grade','mark','score']) or ''
    scal = _first(mod, ['grade_scale','scale']) or ''
    cred = _first(mod, ['credits','credit','units','ects']) or ''
    cu   = _first(mod, ['credit_unit','unit','units_label']) or ''
    term = _first(mod, ['term','semester','session','term_name']) or ''
    pf   = _first(mod, ['pf','pass_fail','passfail','is_pf']) or ''

    mod = mod.fillna('')

    bucket: Dict[str, list] = {}
    for _, r in mod.iterrows():
        k = _norm_key(r.get(key_col, ''))
        if not k: continue
        subject_raw = str(r.get(subj, '')).upper().strip()
        if subject_raw in {'TERM', 'THE', 'OF', 'AND', 'AGES', 'SINCE'}:
            continue
        ent = {
            'subject': subject_raw,
            'number': r.get(num, ''),
            'title': str(r.get(titl, '')).strip(),
            'grade': r.get(grad, ''),
            'grade_scale': (str(r.get(scal, '')).strip().lower() or None),
            'credits': r.get(cred, '') or 3.0,
            'credit_unit': r.get(cu, '') or 'credits',
            'term': str(r.get(term, '')).strip(),
            'pf': str(r.get(pf, '')).strip().lower() in ('1','true','t','yes','y'),
        }
        try:
            cred_val = float(ent['credits'])
            if cred_val <= 0 or cred_val > 10:
                ent['credits'] = 3.0
            else:
                ent['credits'] = cred_val
        except Exception:
            ent['credits'] = 3.0
        bucket.setdefault(k, []).append(ent)

    if 'module_records' not in df.columns:
        df['module_records'] = ''
    cnt = 0
    for i, row in df.iterrows():
        k = _norm_key(row.get('application_key', ''))
        cur = str(row.get('module_records', '')).strip()
        if cur and cur not in ('[]','[ ]'): continue
        if k in bucket and bucket[k]:
            df.at[i, 'module_records'] = json.dumps(bucket[k], ensure_ascii=False)
            cnt += 1
    if verbose:
        print(f"[modules] packed module_records for {cnt} rows from _tmp_modules.csv.")


# ==========================================================
# Institution helpers
# ==========================================================
def _inst_score(name: str) -> tuple[int, int]:
    """
    Score institutions: prefer University > College > Community College, shorter better.
    Returns (score, negative_length) for sorting descending.
    """
    n = (name or '').strip().lower()
    if not name: return (-999, -999)
    if _UVA_NOISE.search(name): return (-999, -999)
    pts = 0
    if 'university' in n: pts += 3
    if 'community college' in n: pts += 2  # keep below 'university' but above plain college
    if 'college' in n and 'community college' not in n: pts += 2
    if 'institute' in n: pts += 1
    return (pts, -len(name))


def _is_institution_like(s: str) -> bool:
    if not s: return False
    if _UVA_NOISE.search(s): return False
    return bool(_INSTITUTION_KEYWORD.search(s))


def _has_uva_hints(row: pd.Series) -> bool:
    """Return True if this row strongly suggests UVA (used only as a last resort)."""
    # Any institution column mentions UVA
    for c in [cc for cc in row.index if re.match(r'^institution_\d+_name$', cc)]:
        val = _to_str_safe(row.get(c, ''))
        if re.search(r'(?i)\buniversity of virginia\b', val):
            return True
    # Major hint?
    if UVA_MAJOR_HINT.search(_to_str_safe(row.get('major_most_recent',''))):
        return True
    # Subject hints / common UVA ECON numbers
    mj = _to_str_safe(row.get('module_records',''))
    if mj and mj not in ('[]','[ ]'):
        try:
            mods = json.loads(mj)
        except Exception:
            return False
        for m in mods:
            subj = str(m.get('subject','')).upper().strip()
            if subj in UVA_SUBJECT_HINTS:
                return True
        for m in mods:
            subj = str(m.get('subject','')).upper().strip()
            if subj in ('ECON','EC','ECN'):
                try:
                    num = int(str(m.get('number','')).strip())
                except Exception:
                    num = None
                if num in UVA_ECON_NUMS:
                    return True
    return False


# ==========================================================
# Main pipeline
# ==========================================================
def main(inp, outp, *, force=False, merge=True, merge_log=None, merge_mode='strict',
         hints=None, no_fill_missing=False, verbose=False):
    hints = hints or []
    for h in hints:
        if '=' in h:
            k, v = h.split('=', 1)
            INSTITUTION_HINTS[k.strip().lower()] = v.strip()

    df = pd.read_csv(inp)

    # ---- JSON corruption cleaning (NEW) ----
    if verbose:
        print("[clean] Checking for JSON corruption in text fields...")
    cleaned_count = clean_json_corruption(df, verbose=verbose)
    if cleaned_count > 0:
        print(f"[clean] Cleaned {cleaned_count} JSON-corrupted fields")

    # ---- Drop only obvious garbage rows (conservative) ----
    # Keep if application_key is exactly 4 digits
    df['application_key'] = df['application_key'].apply(normalize_app_key_4)
    mask_valid_key = df['application_key'].str.match(r'^\d{4}$', na=False)

    # Drop if JSON fragments appear in name fields (student/university) — NOT module_records
    mask_not_json_names = ~(
        df['student_name'].astype(str).str.contains(JSON_FRAGMENT_RE, regex=True, na=False) |
        df['university_name'].astype(str).str.contains(JSON_FRAGMENT_RE, regex=True, na=False)
    )

    # Drop the specific "0 KINE""..." spill pattern if it went into student_name
    mask_not_spilly_name = ~df['student_name'].astype(str).str.match(SPILLY_NAME_RE, na=False)

    keep = mask_valid_key & mask_not_json_names & mask_not_spilly_name
    if verbose:
        dropped = int((~keep).sum())
        if dropped:
            print(f"[hygiene] Dropping {dropped} malformed rows (bad key / JSON spill in names).")
    df = df[keep].copy()

    # Ensure baseline columns exist
    for col in [
        'converted gpa','cum_gpa_most_recent','university_name','module_records','conversion_rule',
        'degree_most_recent','degree_awarded_date','degree_classification',
        'num_semesters_with_grades','scale_hint','country_hint','student_name',
        'institution_1_name','institution_1_dates','institution_1_degree'
    ]:
        if col not in df.columns:
            df[col] = ''

    # Drop Unnamed:* columns
    for c in list(df.columns):
        if str(c).lower().startswith('unnamed:'):
            df.drop(columns=[c], inplace=True)

    # Sanitize text fields
    for c in ['student_name','university_name','major_most_recent','degree_most_recent',
              'degree_awarded_date','degree_classification','scale_hint','country_hint']:
        df[c] = df[c].apply(_sanitize)

    # Normalize/blank current university_name -> allow backfill/override
    for i, row in df.iterrows():
        un_raw = _to_str_safe(row.get('university_name',''))
        un = html.unescape(un_raw).strip()
        if UVA_SEGMENT_PAT.search(un):
            df.at[i,'university_name'] = UVA_NAME
            continue
        # If admin header or not institution-like, blank out so we can backfill
        if _UVA_NOISE.search(un) or not _INSTITUTION_KEYWORD.search(un):
            df.at[i,'university_name'] = ''
            continue
        df.at[i,'university_name'] = _normalize_uva_case(_squash_repeats_univ(un))

    # Pack modules from CSV if present (fill empty module_records)
    _pack_modules_from_csv_if_present(df, verbose=verbose)

    # Clean institution_*_name values
    inst_name_cols = [c for c in df.columns if re.match(r'^institution_\d+_name$', c)]
    for col in inst_name_cols:
        df[col] = df[col].apply(_clean_inst_text)

    # Backfill / override university_name:
    # 1) If institution_1_name looks valid, prefer it.
    # 2) Else, if blank/noisy, pick the best scored institution from all institution_*_name.
    # 3) Else, if UVA hints from modules or major, set University of Virginia.
    for i, row in df.iterrows():
        cur = _to_str_safe(row.get('university_name','')).strip()
        inst1 = _clean_inst_text(row.get('institution_1_name',''))

        if _is_institution_like(inst1):
            df.at[i,'university_name'] = _normalize_uva_case(_squash_repeats_univ(inst1))
            continue

        needs = (not cur) or (not _is_institution_like(cur))
        if needs:
            picks = []
            for c in inst_name_cols:
                v = _clean_inst_text(row.get(c, ''))
                if _is_institution_like(v):
                    picks.append(v)
            if picks:
                # stable de-dupe (keep order as they appear)
                seen = set()
                picks = [p for p in picks if not (p in seen or seen.add(p))]
                # choose best by simple score
                picks.sort(key=_inst_score, reverse=True)
                best = picks[0]
                df.at[i,'university_name'] = _normalize_uva_case(_squash_repeats_univ(best))
                continue

            # Last resort: UVA hints (only if still blank/noisy)
            if _has_uva_hints(row):
                df.at[i,'university_name'] = UVA_NAME

    # Normalize student names (Last, First)
    df['student_name'] = df['student_name'].apply(normalize_name_last_first)

    # Clean GPA field strings
    df['cum_gpa_most_recent'] = (
        df['cum_gpa_most_recent'].astype(str).str.replace('\u00A0',' ', regex=False).str.strip()
    )
    df.loc[df['cum_gpa_most_recent'].str.lower().isin(['nan','null','none']), 'cum_gpa_most_recent'] = ''

    # Convert provided cum GPAs to 4.0 (or overwrite if --force)
    updates = 0
    if 'converted gpa' not in df.columns:
        df['converted gpa'] = ''
    if 'conversion_rule' not in df.columns:
        df['conversion_rule'] = ''

    for i, row in df.iterrows():
        conv_raw = row.get('converted gpa', None)
        conv_str = '' if pd.isna(conv_raw) else str(conv_raw).strip().lower()
        empty_conv = conv_str in ('', 'n/a', 'na', '-', '—', '–', 'null', 'none')
        if (not empty_conv) and (not force):
            continue
        raw = row.get('cum_gpa_most_recent','')
        if raw:
            conv, rule = apply_conversion(
                raw,
                university_name=row.get('university_name',''),
                scale_hint=row.get('scale_hint','') or None,
                country_hint=row.get('country_hint','') or None
            )
            if conv is not None:
                df.at[i,'converted gpa'] = round(conv, 3)
                if rule:
                    df.at[i,'conversion_rule'] = rule
                updates += 1
                if verbose:
                    print(f"[convert] {row.get('application_key','')} raw={raw} -> {df.at[i,'converted gpa']} ({df.at[i,'conversion_rule']})")

    # Synthesize GPA from modules if still missing
    synth_count = 0
    for i, row in df.iterrows():
        if str(row.get('cum_gpa_most_recent','')).strip():
            continue
        raw_avg, scale, weighting = synthesize_from_modules(row.get('module_records',''))
        if raw_avg is None:
            continue
        df.at[i,'cum_gpa_most_recent'] = round(raw_avg, 3)
        conv, rule = apply_scale_conversion(raw_avg, scale)
        if conv is not None:
            df.at[i,'converted gpa'] = round(conv, 3)
            df.at[i,'conversion_rule'] = f"{rule}_synthesized_{weighting or 'creditweighted'}"
            synth_count += 1
            if verbose:
                print(f"[synth ] {row.get('application_key','')} avg={raw_avg} scale={scale} -> {df.at[i,'converted gpa']}")

    # Derive semesters + discipline metrics
    non_empty_modules = 0
    if 'num_semesters_with_grades' not in df.columns:
        df['num_semesters_with_grades'] = 0

    for i, row in df.iterrows():
        mj = str(row.get('module_records','')).strip()
        if mj and mj not in ('[]','[ ]'):
            non_empty_modules += 1
        try:
            mods = json.loads(mj)
            # count unique terms with any grade or PF
            terms = set()
            for m in mods:
                if m.get('grade') is not None or m.get('pf'):
                    term = (m.get('term') or '').strip()
                    if term:
                        terms.add(term)
            if not str(row.get('num_semesters_with_grades','')).strip():
                df.at[i,'num_semesters_with_grades'] = len(terms)
        except Exception:
            pass
        metrics = derive_discipline_metrics(mj)
        for k, v in metrics.items():
            if k not in df.columns:
                df[k] = ''
            df.at[i,k] = v

    if verbose:
        print(f"[metrics] module_records present for {non_empty_modules} rows")
        print(f"[metrics] synthesized {synth_count} GPAs from course data")

    # Fill numeric; leave text blanks
    if not no_fill_missing:
        INT_COLS = [
            'num_semesters_with_grades',
            'num_courses_graded_econ','num_courses_pf_econ',
            'num_courses_graded_stats','num_courses_pf_stats',
            'num_courses_graded_math','num_courses_pf_math'
        ]
        for c in INT_COLS:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

        AVG_COLS = ['avg_gpa_econ','avg_gpa_stats','avg_gpa_math']
        for c in AVG_COLS:
            if c not in df.columns:
                df[c] = ''
            df[c] = df[c].apply(lambda x: '' if str(x).strip()=='' else float(x))

        TEXT_COLS = [
            'student_name','university_name','major_most_recent',
            'degree_most_recent','degree_awarded_date','degree_classification',
            'conversion_rule','highest_econ_course_full_title',
            'institution_1_name','institution_1_dates','institution_1_degree',
            'institution_2_name','institution_2_dates','institution_2_degree',
            'scale_hint','country_hint'
        ]
        for c in TEXT_COLS:
            if c not in df.columns:
                df[c] = ''
            df[c] = df[c].fillna('').astype(str).str.strip()

    # Highest ECON course
    if 'highest_econ_course_full_title' not in df.columns:
        df['highest_econ_course_full_title'] = ''
    econ_count = 0
    for i, row in df.iterrows():
        if str(df.at[i,'highest_econ_course_full_title']).strip():
            continue
        mj = row.get('module_records','')
        if not str(mj).strip():
            continue
        try:
            mods = json.loads(mj)
        except Exception:
            continue
        best = None
        best_rank = -1
        for m in mods:
            subj = str(m.get('subject','')).upper()
            if subj not in ('ECON', 'ECN', 'EC'):
                continue
            try:
                num = int(str(m.get('number','')).strip())
            except Exception:
                num = None
            if num is None:
                continue
            cat = econ_core_category(m.get('title',''), num)
            if not cat:
                continue
            rank = CORE_PRIORITY[cat]
            if rank > best_rank or (rank == best_rank and (best is None or (num or 0) > (best.get('number') or 0))):
                best_rank = rank
                best = m
        if best:
            n = best.get('number')
            prefix = f"ECON {n} " if n else ""
            title = str(best.get('title','')).strip()
            df.at[i,'highest_econ_course_full_title'] = (prefix + title).strip()
            econ_count += 1
            if verbose:
                print(f"[econ  ] {row.get('application_key','')} -> {df.at[i,'highest_econ_course_full_title']}")

    if verbose and econ_count > 0:
        print(f"[econ  ] identified highest ECON course for {econ_count} students")

    # Sort by application key (numeric)
    try:
        df['__k__'] = df['application_key'].astype(int)
        df = df.sort_values(['__k__']).drop(columns=['__k__'])
    except Exception:
        pass

    # ---- Drop module_records column from final output ----
    if 'module_records' in df.columns:
        df = df.drop(columns=['module_records'])
        if verbose:
            print("[final] Removed module_records column from output")

    df = enforce_column_order(df)
    df.to_csv(outp, index=False)
    print(f"Wrote {outp} (updated {updates} conversions, synthesized {synth_count} GPAs)")


# ==========================================================
# CLI
# ==========================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='outp', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-merge', action='store_true')  # accepted for compatibility; not used
    p.add_argument('--merge-log', dest='merge_log', default=None)
    p.add_argument('--merge-mode', choices=['strict','loose'], default='strict')
    p.add_argument('--hint', action='append', default=[])
    p.add_argument('--no-fill-missing', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    main(
        args.inp,
        args.outp,
        force=args.force,
        merge=(not args.no_merge),
        merge_log=args.merge_log,
        merge_mode=args.merge_mode,
        hints=args.hint,
        no_fill_missing=args.no_fill_missing,
        verbose=args.verbose
        )


