# postprocess_master.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, re, html
from typing import Optional, Any, Dict, List
import pandas as pd
from scale_rules import apply_conversion, INSTITUTION_HINTS

COLUMN_ORDER = [
    'application_key','student_name','university_name','major_most_recent',
    'degree_most_recent','degree_awarded_date','degree_classification',
    'num_semesters_with_grades',
    'num_courses_graded_econ','num_courses_graded_stats','num_courses_graded_math',
    'num_courses_pf_econ','num_courses_pf_stats','num_courses_pf_math',
    'avg_gpa_econ','avg_gpa_stats','avg_gpa_math',
    'cum_gpa_most_recent','converted gpa','conversion_rule',
    'highest_econ_course_full_title',
    'institution_1_name','institution_1_dates','institution_1_degree',
    'institution_2_name','institution_2_dates','institution_2_degree',
    'module_records','date_of_birth','merged_from_application_keys',
    'scale_hint','country_hint'
]

SUFFIXES = {"Jr", "Sr", "II", "III", "IV"}
NAME_COMMA_RE = re.compile(r",\s*")
NOISE_TOKENS = (
    'birth date','university','registrar','degree awarded',
    'requirements are accepted','transcript','unofficial',
    'tsrpt','app ssr','ssr','report','official','student no','student id'
)
MONTHS = {'january','february','march','april','may','june','july','august','september','october','november','december'}
UVA_SEGMENT_PAT = re.compile(r'^(Primary\s+College:|College\s*&\s*Graduate\s*Arts\s*&\s*Sci|University\s+Seminar)', re.IGNORECASE)

STOP_SUBJECTS = {
    'AUG','SEP','SEPT','OCT','NOV','DEC','JAN','FEB','MAR','APR','MAY','JUN','JUL',
    'THE','AND','OF','TO','IN','ON','AT','BY','FOR','FROM',
    'SINCE','TERM','SUITE','CREDENTIAL','DATE','ISSUED','AGES','NEWS',
    'SPRING','FALL','SUMMER','WINTER'
}
LETTER_TO_4 = {
    'A+':4.0,'A':4.0,'A-':3.7,'B+':3.3,'B':3.0,'B-':2.7,
    'C+':2.3,'C':2.0,'C-':1.7,'D+':1.3,'D':1.0,'D-':0.7,'F':0.0
}
DISCIPLINES = {
    'ECON': ('num_courses_graded_econ','num_courses_pf_econ','avg_gpa_econ'),
    'STAT': ('num_courses_graded_stats','num_courses_pf_stats','avg_gpa_stats'),
    'MATH': ('num_courses_graded_math','num_courses_pf_math','avg_gpa_math'),
}

def _sanitize(x: Any) -> str:
    s = str(x or '').strip()
    return '' if s.lower() in ('nan','none','null') else s

def normalize_name_last_first(name: str) -> str:
    s = _sanitize(name)
    if not s: return s
    lo = s.lower()
    if any(tok in lo for tok in NOISE_TOKENS) or lo in MONTHS:
        return s
    if "," in s:
        parts = NAME_COMMA_RE.split(s, maxsplit=1)
        if len(parts) == 2:
            return f"{parts[0].strip()}, {parts[1].strip()}"
        return s
    parts = s.split()
    if len(parts) == 1: return s
    suffix = None
    if parts[-1].strip(".").title() in SUFFIXES:
        suffix = parts[-1].strip("."); parts = parts[:-1]
    last = parts[-1]; first_mid = " ".join(parts[:-1])
    out = f"{last}, {first_mid}"
    if suffix: out += f" {suffix}"
    return out

def normalize_app_key_4(key: str) -> str:
    digits = re.sub(r"\D", "", str(key or ''))
    if not digits: return ''
    return digits[-4:].zfill(4)

def enforce_column_order(df: pd.DataFrame) -> pd.DataFrame:
    for c in ['application_key','student_name','university_name','date_of_birth']:
        if c not in df.columns: df[c] = ''
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    return df[ordered + tail]

# ---------------- Module filters ----------------
def _ok_subject_token(subj: str) -> bool:
    if not subj: return False
    s = subj.upper()
    if s in STOP_SUBJECTS: return False
    return bool(re.fullmatch(r'[A-Z&]{2,6}', s))

def _is_reasonable_credits(x) -> bool:
    try:
        v = float(x)
        return 0.25 <= v <= 10.0
    except Exception:
        return False

def _normalize_subject_aliases(subj: str) -> str:
    s = subj.upper()
    if s in ('EC','ECN'): return 'ECON'
    if s in ('APMA','AM'): return 'MATH'
    if s == 'STS': return 'STAT'
    return s

def _valid_for_overall(m: dict) -> bool:
    # Keep valid modules for overall GPA (all subjects)
    subj = str(m.get('subject','')).upper().strip()
    if not _ok_subject_token(subj): return False
    cr = m.get('credits', 3.0)
    if not _is_reasonable_credits(cr): return False
    # Accept graded or PF courses; GPA will ignore PF (but they can contribute to semester counts)
    g = m.get('grade', None)
    if g is None and not m.get('pf', False):
        return False
    # If numeric grade exists, ensure it is within reasonable range (0..100)
    if g is not None and not isinstance(g, str):
        try:
            v = float(g)
            if v < 0 or v > 100: return False
        except Exception:
            return False
    return True

def _valid_for_discipline(m: dict) -> bool:
    # Keep modules only if subject maps to ECON/STAT/MATH and otherwise valid
    if not _valid_for_overall(m): return False
    subj = _normalize_subject_aliases(str(m.get('subject','')))
    return subj in DISCIPLINES

# ---------------- GPA conversion helpers ----------------
def _grade_any_to_4(grade: Any, scale: str) -> float | None:
    from scale_rules import apply_conversion as conv
    if grade is None: return None
    if isinstance(grade, str):
        g = grade.strip().upper()
        if g in LETTER_TO_4: return LETTER_TO_4[g]
        if g.endswith('%'):
            try: pct = float(g[:-1]); return max(0.0, min(pct/100.0*4.0, 4.0))
            except: return None
        try: v = float(g); grade = v
        except: return None
    try: v = float(grade)
    except: return None
    # If appears to be already on 0–4 scale
    if 0.0 <= v <= 4.3: return max(0.0, min(v, 4.0))
    # Else convert with best-effort rule detection
    out, _ = conv(v, scale_hint=(scale or '').lower() or None)
    return out

def _gpa4_from_modules(mods: List[dict]) -> Optional[float]:
    """Credit-weighted GPA on 4.0 scale using all valid **graded** modules (ignore PF)."""
    ws, wt = 0.0, 0.0
    for m in mods:
        if m.get('pf', False):  # pass/fail doesn't count toward GPA
            continue
        g = m.get('grade'); c = m.get('credits', 1.0)
        if g is None: continue
        try: c = float(c)
        except: c = 1.0
        if c <= 0: c = 1.0
        g4 = _grade_any_to_4(g, (m.get('grade_scale') or 'us_4'))
        if g4 is None: continue
        ws += g4 * c; wt += c
    if wt == 0: return None
    return round(ws / wt, 3)

# ---------------- Discipline metrics ----------------
def derive_discipline_metrics(mods_all: List[dict]) -> dict:
    out = {v: '' for d in DISCIPLINES.values() for v in d}
    if not mods_all: return out
    # Filter to discipline subset only
    mods = []
    for m in mods_all:
        mm = dict(m)
        mm['subject'] = _normalize_subject_aliases(str(m.get('subject','')))
        if _valid_for_discipline(mm):
            mods.append(mm)
    cnt_g = {k:0 for k in DISCIPLINES}
    cnt_pf= {k:0 for k in DISCIPLINES}
    sum4 = {k:0.0 for k in DISCIPLINES}
    wts  = {k:0.0 for k in DISCIPLINES}
    for m in mods:
        subj = str(m.get('subject','')).upper().strip()
        credits = m.get('credits', 3.0)
        try: credits = float(credits or 3.0)
        except: credits = 3.0
        if credits <= 0: credits = 3.0
        scale = (m.get('grade_scale') or 'us_4')
        grade = m.get('grade', None)
        if m.get('pf') or str(grade).strip().upper() in ('P','S','U','CR','NC','PASS','FAIL'):
            cnt_pf[subj] += 1; continue
        g4 = _grade_any_to_4(grade, scale)
        if g4 is None:
            cnt_pf[subj] += 1; continue
        cnt_g[subj] += 1
        sum4[subj] += g4 * credits
        wts[subj]  += credits
    for subj, (n_g, n_pf, avg) in DISCIPLINES.items():
        out[n_g] = cnt_g[subj]
        out[n_pf]= cnt_pf[subj]
        out[avg] = round(sum4[subj]/wts[subj], 3) if wts[subj] > 0 else ''
    return out

# ---------------- ECON highest course ----------------
CORE_PRIORITY = {"intro": 1, "intermediate": 2, "econometrics": 3}
INTRO_PAT     = re.compile(r"\b(Principles|Intro(?:duction)?)\b.*\b(Micro|Macro)\b", re.IGNORECASE)
INTER_PAT     = re.compile(r"\bIntermediate\b.*\b(Micro|Macro)\b", re.IGNORECASE)
ECONMET_PAT   = re.compile(r"\bEconometrics?\b", re.IGNORECASE)

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

# ---------------- Pack modules from CSV if missing ----------------
def _pack_modules_from_csv_if_present(df: pd.DataFrame, verbose=False):
    path = '_tmp_modules.csv'
    if not os.path.exists(path):
        if verbose: print("[modules] no _tmp_modules.csv found to pack.")
        return
    try:
        mod = pd.read_csv(path)
    except Exception as e:
        if verbose: print(f"[modules] failed to read {path}: {e}")
        return

    def _norm_key(k: object) -> str:
        s = re.sub(r"\D", "", str(k or ''))
        if not s: return ''
        return (s[-4:]).zfill(4)
    def _first(df, cands):
        for c in cands:
            if c in df.columns: return c
        lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in lower: return lower[c.lower()]
        return None

    key_col = _first(mod, ['application_key','app_key','key','id','Application Key'])
    subj = _first(mod, ['subject','dept','department','course_subject']) or ''
    num  = _first(mod, ['number','course_number','catalog','catalog_number','cat_num']) or ''
    titl = _first(mod, ['title','course_title','name']) or ''
    grad = _first(mod, ['grade','final_grade','mark','score']) or ''
    scal = _first(mod, ['grade_scale','scale']) or ''
    cred = _first(mod, ['credits','credit','units','ects']) or ''
    cu   = _first(mod, ['credit_unit','unit','units_label']) or ''
    term = _first(mod, ['term','semester','session','term_name']) or ''
    pf   = _first(mod, ['pf','pass_fail','passfail','is_pf']) or ''
    if key_col is None:
        if verbose: print("[modules] no key column; skipping pack.")
        return
    mod = mod.fillna('')
    bucket: Dict[str, list] = {}
    for _, r in mod.iterrows():
        k = _norm_key(r.get(key_col, ''))
        if not k: continue
        ent = {
            'subject': str(r.get(subj, '')).upper(),
            'number': r.get(num, ''),
            'title': str(r.get(titl, '')).strip(),
            'grade': r.get(grad, ''),
            'grade_scale': (str(r.get(scal, '')).strip().lower() or None),
            'credits': r.get(cred, '') or 3.0,
            'credit_unit': r.get(cu, '') or 'credits',
            'term': str(r.get(term, '')).strip(),
            'pf': str(r.get(pf, '')).strip().lower() in ('1','true','t','yes','y'),
        }
        bucket.setdefault(k, []).append(ent)
    if 'module_records' not in df.columns:
        df['module_records'] = ''
    cnt = 0
    for i, row in df.iterrows():
        k = _norm_key(row.get('application_key', ''))
        cur = str(row.get('module_records','')).strip()
        if cur and cur not in ('[]','[ ]'): continue
        if k in bucket and bucket[k]:
            df.at[i, 'module_records'] = json.dumps(bucket[k], ensure_ascii=False)
            cnt += 1
    if verbose: print(f"[modules] packed module_records for {cnt} rows from _tmp_modules.csv.")

# ---------------- Main ----------------
def main(inp, outp, *, force=False, merge=True, merge_log=None, merge_mode='strict',
         hints=None, no_fill_missing=False, verbose=False):
    hints = hints or []
    for h in hints:
        if '=' in h:
            k, v = h.split('=', 1)
            INSTITUTION_HINTS[k.strip().lower()] = v.strip()

    df = pd.read_csv(inp)

    # Ensure columns
    for col in ['converted gpa','cum_gpa_most_recent','university_name','module_records','conversion_rule',
                'degree_most_recent','degree_awarded_date','degree_classification',
                'num_semesters_with_grades','scale_hint','country_hint','student_name','application_key',
                'institution_1_name','institution_1_dates','institution_1_degree']:
        if col not in df.columns: df[col]=''

    # Remove 'Unnamed:*'
    for c in list(df.columns):
        if str(c).lower().startswith('unnamed:'):
            df.drop(columns=[c], inplace=True)

    # Normalize key & student name
    df['application_key'] = df['application_key'].apply(normalize_app_key_4)
    df['student_name'] = df['student_name'].apply(normalize_name_last_first)

    # UVA segment headers => University of Virginia; fallback to institution_1 if blank but looks like a school
    for i, row in df.iterrows():
        un = str(row.get('university_name','')).strip()
        if UVA_SEGMENT_PAT.search(html.unescape(un)) or ('university of virginia' in un.lower() and len(un) > 40):
            df.at[i,'university_name'] = 'University of Virginia'
        if not df.at[i,'university_name'] and str(row.get('institution_1_name','')).strip():
            i1 = str(row.get('institution_1_name','')).strip()
            if re.search(r"\b(university|college|institute|polytechnic|school)\b", i1, re.IGNORECASE):
                df.at[i,'university_name'] = i1

    # GPA 'nan' -> empty
    df['cum_gpa_most_recent'] = df['cum_gpa_most_recent'].astype(str).str.replace('\u00A0',' ', regex=False).str.strip()
    df.loc[df['cum_gpa_most_recent'].str.lower().isin(['nan','null','none']), 'cum_gpa_most_recent'] = ''

    # If module_records empty, pack from _tmp_modules.csv
    _pack_modules_from_csv_if_present(df, verbose=verbose)

    # ---------------- Overall GPA synthesis (ALL VALID COURSES) ----------------
    synth_count = 0
    for i, row in df.iterrows():
        # parse modules
        mr = str(row.get('module_records','')).strip()
        mods_all: List[dict] = []
        if mr and mr not in ('[]','[ ]'):
            try:
                mods_raw = json.loads(mr)
            except Exception:
                mods_raw = []
            # keep all valid modules for overall GPA
            for m in mods_raw:
                if _valid_for_overall(m):
                    mods_all.append(m)

        # If cum_gpa present in transcript, convert it to 4.0 and keep; else synthesize from all valid modules
        conv_raw = row.get('converted gpa', None)
        conv_str = '' if pd.isna(conv_raw) else str(conv_raw).strip().lower()
        empty_conv = conv_str in ('', 'n/a', 'na', '-', '—', '–', 'null', 'none')
        has_raw_cum = bool(str(row.get('cum_gpa_most_recent','')).strip())

        if has_raw_cum:
            if empty_conv or force:
                raw = row.get('cum_gpa_most_recent','')
                conv, rule = apply_conversion(raw,
                    university_name=row.get('university_name',''),
                    scale_hint=row.get('scale_hint','') or None,
                    country_hint=row.get('country_hint','') or None)
                if conv is not None:
                    df.at[i,'converted gpa'] = round(conv, 3)
                    df.at[i,'conversion_rule'] = rule or 'us_4'
        else:
            # synthesize overall 4.0 GPA from all valid modules
            gpa4 = _gpa4_from_modules(mods_all)
            if gpa4 is not None:
                df.at[i,'cum_gpa_most_recent'] = gpa4
                df.at[i,'converted gpa'] = gpa4
                df.at[i,'conversion_rule'] = 'us_4_synthesized_creditweighted'
                synth_count += 1
    if verbose:
        print(f"[overall] synthesized cumulative GPA for {synth_count} rows")

    # ---------------- Discipline metrics (ECON / STAT / MATH only) ----------------
    have_modules = 0
    for i, row in df.iterrows():
        mr = str(row.get('module_records','')).strip()
        if not mr or mr in ('[]','[ ]'):
            continue
        try:
            mods_raw = json.loads(mr)
        except Exception:
            continue
        # keep only valid modules (any subject) for semester counting
        mods_valid_all = [m for m in mods_raw if _valid_for_overall(m)]
        # recompute semesters (non-empty terms with graded or PF)
        terms = { (m.get('term') or '').strip() for m in mods_valid_all if (m.get('grade') is not None) or m.get('pf') }
        terms.discard('')
        if not str(row.get('num_semesters_with_grades','')).strip() or int(row.get('num_semesters_with_grades') or 0) == 0:
            df.at[i,'num_semesters_with_grades'] = len(terms)
        # fill discipline metrics
        metrics = derive_discipline_metrics(mods_valid_all)
        for k, v in metrics.items():
            if k not in df.columns: df[k] = ''
            df.at[i,k] = v
        have_modules += 1
    if verbose:
        print(f"[metrics] computed discipline metrics for {have_modules} rows")

    # ---------------- Highest ECON course ----------------
    if 'highest_econ_course_full_title' not in df.columns:
        df['highest_econ_course_full_title'] = ''
    for i, row in df.iterrows():
        if str(df.at[i,'highest_econ_course_full_title']).strip():
            continue
        mr = row.get('module_records','')
        if not str(mr).strip():
            continue
        try:
            mods = json.loads(mr)
        except Exception:
            continue
        best = None; best_rank = -1
        for m in mods:
            subj = _normalize_subject_aliases(str(m.get('subject','')))
            if subj != 'ECON': continue
            try: num = int(str(m.get('number','')).strip())
            except Exception: num = None
            cat = econ_core_category(m.get('title',''), num)
            if not cat: continue
            rank = CORE_PRIORITY[cat]
            if rank > best_rank or (rank == best_rank and (best is None or (num or 0) > (best.get('number') or 0))):
                best_rank = rank; best = m
        if best:
            n = best.get('number'); prefix = f"ECON {n} " if n else ""
            df.at[i,'highest_econ_course_full_title'] = (prefix + str(best.get('title','')).strip()).strip()
            if verbose:
                print(f"[econ  ] {row.get('application_key','')} -> {df.at[i,'highest_econ_course_full_title']}")

    # ---------------- Fill missing & write ----------------
    if not no_fill_missing:
        INT_COLS = ['num_semesters_with_grades',
                    'num_courses_graded_econ','num_courses_graded_stats','num_courses_graded_math',
                    'num_courses_pf_econ','num_courses_pf_stats','num_courses_pf_math']
        for c in INT_COLS:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
        AVG_COLS = ['avg_gpa_econ','avg_gpa_stats','avg_gpa_math']
        for c in AVG_COLS:
            if c not in df.columns: df[c] = ''
            df[c] = df[c].apply(lambda x: '' if str(x).strip()=='' else float(x))
        TEXT_COLS = ['student_name','university_name','major_most_recent',
                     'degree_most_recent','degree_awarded_date','degree_classification',
                     'conversion_rule','highest_econ_course_full_title',
                     'institution_1_name','institution_1_dates','institution_1_degree',
                     'institution_2_name','institution_2_dates','institution_2_degree',
                     'scale_hint','country_hint']
        for c in TEXT_COLS:
            if c not in df.columns: df[c] = ''
            df[c] = df[c].fillna('')
            df.loc[df[c].astype(str).str.strip().isin(['','nan','null','none']), c] = 'N/A'

    try:
        df['__k__'] = df['application_key'].astype(int)
        df = df.sort_values(['__k__']).drop(columns=['__k__'])
    except Exception:
        pass

    df = enforce_column_order(df)
    df.to_csv(outp, index=False)
    print(f"Wrote {outp}")
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='outp', required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-merge', action='store_true')
    p.add_argument('--merge-log', dest='merge_log', default=None)
    p.add_argument('--merge-mode', choices=['strict','loose'], default='strict')
    p.add_argument('--hint', action='append', default=[])
    p.add_argument('--no-fill-missing', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    main(args.inp, args.outp, force=args.force, merge=(not args.no_merge),
         merge_log=args.merge_log, merge_mode=args.merge_mode, hints=args.hint,
         no_fill_missing=args.no_fill_missing, verbose=args.verbose)
