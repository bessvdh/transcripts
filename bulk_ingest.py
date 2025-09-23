# bulk_ingest.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Try to import PyMuPDF
try:
    import fitz  # type: ignore
except Exception:
    fitz = None

# ---------------------------- Patterns & constants ----------------------------

TERM_PAT = re.compile(r"\b(Fall|Spring|Summer|Winter|Autumn)\s+(\d{4})\b", re.IGNORECASE)

# NEW: UVA-friendly pattern (single spaces between title / grade / credits)
COURSE_PAT_UVA = re.compile(
    r"^(?P<subject>[A-Za-z&]{2,6})\s*-?\s*(?P<number>\d{3,4})\s+"
    r"(?P<title>.+?)\s+"
    r"(?P<grade>(?:A\+|A\-|A|B\+|B\-|B|C\+|C\-|C|D\+|D\-|D|F|AP|HP|H\*?|P|S|U|CR|NC|GC|Pass|Fail|\d+(?:\.\d+)?%?))\s+"
    r"(?P<credits>\d+(?:\.\d+)?)\s*$"
)

# Strict “columnar” patterns seen in SIS PDFs (title then grade then credits)
COURSE_PAT_1 = re.compile(
    r"^(?P<subject>[A-Za-z&]{2,6})\s*-?\s*(?P<number>\d{3,4})\s+"
    r"(?P<title>.+?)\s{2,}"
    r"(?P<grade>(?:A\+|A\-|A|B\+|B\-|B|C\+|C\-|C|D\+|D\-|D|F|AP|HP|H\*?|P|S|U|CR|NC|GC|Pass|Fail|\d+(?:\.\d+)?%?))\s+"
    r"(?P<credits>\d+(?:\.\d+)?)\s*$"
)

# Grade / Credits labeled variants
COURSE_PAT_2 = re.compile(
    r"^(?P<subject>[A-Za-z&]{2,6})\s*-?\s*(?P<number>\d{3,4})\s*[\-–:]?\s*(?P<title>.+?)\s+"
    r"(?:Grade\s*[: ]\s*(?P<grade>(?:A\+|A\-|A|B\+|B\-|B|C\+|C\-|C|D\+|D\-|D|F|AP|HP|H\*?|P|S|U|CR|NC|GC|Pass|Fail|\d+(?:\.\d+)?%?)))"
    r"(?:\s+Credits?\s*[: ]\s*(?P<credits>\d+(?:\.\d+)?))?\s*$"
)

# Credits appearing before grade in some layouts
COURSE_PAT_3 = re.compile(
    r"^(?P<subject>[A-Za-z&]{2,6})\s*-?\s*(?P<number>\d{3,4})\s+"
    r"(?P<title>.+?)\s{2,}(?P<credits>\d+(?:\.\d+)?)\s+"
    r"(?P<grade>(?:A\+|A\-|A|B\+|B\-|B|C\+|C\-|C|D\+|D\-|D|F|AP|HP|H\*?|P|S|U|CR|NC|GC|Pass|Fail|\d+(?:\.\d+)?%?))\s*$"
)

# Sliding-window catch-all: SUBJ 1234
ANY_COURSE_HEAD = re.compile(r"\b([A-Za-z&]{2,6})\s*[- ]?(\d{3,4})\b")
GRADE_TOKEN = re.compile(
    r"\b(?:A\+|A\-|A|B\+|B\-|B|C\+|C\-|C|D\+|D\-|D|F|AP|HP|H\*?|P|S|U|CR|NC|GC|Pass|Fail|\d+(?:\.\d+)?%?)\b",
    re.IGNORECASE,
)
CREDITS_TOKEN = re.compile(r"(?:Credits?|Units?|Hours?)\s*[: \t]\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
TRAILING_CRED = re.compile(r"(\d+(?:\.\d+)?)\s*$")

# Cumulative GPA patterns + STRICT fallback (must contain 'GPA')
CUM_GPA_PAT = re.compile(
    r"\b("
    r"Cumulative\s*(?:GPA|Average|CGPA)"
    r"|Overall\s*GPA"
    r"|CGPA"
    r"|UVA\s*GPA"
    r")\b"
    r"[:\-\s]*"
    r"(?P<gpa>(\d{1,2}(?:\.\d+)?(?:/\d{1,2}(?:\.\d+)?)?|\d{1,3}(?:\.\d+)?%))",
    re.IGNORECASE,
)
GPA_FALLBACK_PAT = re.compile(
    r"(?P<gpa>(\d{1,2}(?:\.\d+)?(?:/\d{1,2}(?:\.\d+)?)?|\d{1,3}(?:\.\d+)?%))",
    re.IGNORECASE,
)

# Labeled fields
NAME_LINE_PAT = re.compile(r"\b(Student\s*Name|Name)\s*[:\-]\s*(?P<name>.+)$", re.IGNORECASE)
MAJOR_PAT = re.compile(r"\b(Major|Plan|Program)\s*[:\-]\s*(?P<major>.+)$", re.IGNORECASE)
DEGREE_PAT = re.compile(r"\b(Degree|Degree\s*Awarded)\s*[:\-]\s*(?P<deg>.+)$", re.IGNORECASE)
DEGREE_DATE_PAT = re.compile(
    r"\b(Awarded|Conferred|Graduation\s*Date)\s*[:\-]\s*(?P<date>[A-Za-z]{3,9}\s+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
    re.IGNORECASE,
)
DEGREE_CLASS_PAT = re.compile(r"\b(Honou?rs?|Honor:|Class|Cum\s*Laude|Magna|Summa)\b.*", re.IGNORECASE)

# Noise guards for “university”
ADDRESS_TOKENS = (
    "street", "st ", "st.", "road", "rd ", "rd.", "avenue", "ave", "ave.", "suite", "ste",
    "po box", "p.o.", "blvd", "drive", "dr ",
)
STUDENT_ID_TOKENS = ("student no", "student id", "id:", "sid", "uvid")

UVA_SEGMENT_PAT = re.compile(
    r'^(Primary\s+College:|College\s*&\s*Graduate\s*Arts\s*&\s*Sci|University\s+Seminar)',
    re.IGNORECASE,
)

LETTER_GRADE = set("A A- B+ B B- C+ C C- D+ D D- F".split())
PF_TOKENS = {"P", "S", "U", "CR", "NC", "PASS", "FAIL", "AP", "HP", "H", "H*", "GC"}  # NEW: GC as PF-like

STOP_SUBJECTS = {
    "AUG","SEP","SEPT","OCT","NOV","DEC","JAN","FEB","MAR","APR","MAY","JUN","JUL",
    "THE","AND","OF","TO","IN","ON","AT","BY","FOR","FROM",
    "SINCE","TERM","SUITE","CREDENTIAL","DATE","ISSUED","AGES","NEWS",
    "SPRING","FALL","SUMMER","WINTER",
    "COVID","COVD"  # NEW: filter COVID banners like “COVD 019”
}

TITLE_NOISE = (
    "date issued","credential awarded","credentials solutions","acting on behalf",
    "may not be released","west monroe street","suite","honor","honour","dean",
    "academic transcript","printed","print date","page","registrar","student name",
    "student id","dob","birth","covid","covid-19","pandemic"
)

NAME_NOISE_TOKENS = (
    "academic transcript","transcript","printed","print date","date printed",
    "page","registrar","student id","student no","report","official",
    "unofficial","dob","birth","issued","verified",
)
MONTH_TOKENS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec"
}
COURSE_NOISE_TOKENS = (
    "grade","credits","units","hours","lab","lecture","seminar",
    "intro","introduction","principles","microeconomics","macroeconomics",
    "calculus","chemistry","physics","biology","english","spanish",
    "course","catalog","catalog number",
)
UNIV_KEYWORDS = ("university","community college","college")
GPA_CONTEXT_TOKENS = ("gpa","cgpa","average","overall")

# ------------------------------- OCR helpers ----------------------------------

def _nonspace_len(s: str) -> int:
    return len(re.sub(r"\s+", "", s or ""))

def _ocr_with_ocrmypdf(pdf_path: str, lang: str) -> str:
    if shutil.which('ocrmypdf') is None:
        return ''
    with tempfile.TemporaryDirectory() as td:
        sidecar = os.path.join(td, 'sidecar.txt')
        outpdf = os.path.join(td, 'out.pdf')
        cmd = ['ocrmypdf', '--sidecar', sidecar, '--force-ocr', '--language', lang, pdf_path, outpdf]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(sidecar):
                with open(sidecar, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception:
            return ''
    return ''

def _ocr_with_tesseract(pdf_path: str, lang: str) -> str:
    if shutil.which('tesseract') is None:
        return ''
    cmd = ['tesseract', pdf_path, 'stdout', '-l', lang]
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.stdout.decode('utf-8', errors='ignore')
    except Exception:
        return ''

# ------------------------------------ IO --------------------------------------

def read_text_from_file(path: str, *, ocr: bool, ocr_threshold: int, ocr_lang: str,
                        debug_dump_dir: Optional[str], verbose: bool) -> str:
    ext = os.path.splitext(path)[1].lower()
    text = ''
    if ext == '.txt':
        for enc in ('utf-8', 'latin-1'):
            try:
                with open(path, 'r', encoding=enc, errors='ignore') as f:
                    text = f.read(); break
            except Exception:
                pass
    elif ext == '.pdf':
        if fitz is not None:
            try:
                doc = fitz.open(path)
                parts = [page.get_text('text') for page in doc]
                doc.close()
                text = "\n".join(parts)
            except Exception:
                text = ''
        if ocr and _nonspace_len(text) < max(0, ocr_threshold):
            if verbose: print(f"[ocr] attempting OCR for {os.path.basename(path)}")
            ocr_txt = _ocr_with_ocrmypdf(path, ocr_lang) or _ocr_with_tesseract(path, ocr_lang)
            if _nonspace_len(ocr_txt) > _nonspace_len(text):
                text = ocr_txt

    if debug_dump_dir:
        try:
            os.makedirs(debug_dump_dir, exist_ok=True)
            with open(os.path.join(debug_dump_dir, os.path.basename(path) + '.txt'),
                      'w', encoding='utf-8', errors='ignore') as f:
                f.write(text)
        except Exception:
            pass

    return text

def application_key_from_filename(filename: str) -> str:
    digits = re.sub(r"\D", "", filename)
    return digits[-4:].zfill(4) if digits else ''

# -------------------------- Top-of-page field finders -------------------------

def _looks_like_person(s: str) -> bool:
    t = re.sub(r'\s+', ' ', s or '').strip()
    if not t or any(ch.isdigit() for ch in t): return False
    lo = t.lower()
    if any(tok in lo for tok in NAME_NOISE_TOKENS): return False
    if any(w in ('university','college','institute','polytechnic','school') for w in lo.split()): return False
    if any(m in lo for m in MONTH_TOKENS): return False
    parts = t.split()
    if not (2 <= len(parts) <= 4): return False
    return all(re.match(r'^[A-Z][a-zA-Z\.\-]*$', p) for p in parts)

def detect_university(lines: List[str]) -> str:
    def _looks_like_university_line(s: str) -> bool:
        s0 = html.unescape(s or '').strip()
        if not s0: return False
        lo = s0.lower()
        if UVA_SEGMENT_PAT.search(s0): return False
        if any(tok in lo for tok in STUDENT_ID_TOKENS): return False
        if any(tok in lo for tok in ADDRESS_TOKENS): return False
        if any(tok in lo for tok in NAME_NOISE_TOKENS): return False
        if any(m in lo for m in MONTH_TOKENS): return False
        if any(tok in lo for tok in COURSE_NOISE_TOKENS): return False
        if re.search(r'\d{2,}', s0): return False
        if not any(k in lo for k in UNIV_KEYWORDS): return False
        return True

    candidates: List[str] = []
    for l in lines[:120]:
        s = (l or '').strip()
        if _looks_like_university_line(s):
            candidates.append(s)
    if candidates:
        def _rank(u: str) -> Tuple[int, int]:
            score = 0
            if re.search(r'\bUniversity of\b', u): score += 3
            if re.search(r'\b(University|College)\b$', u): score += 2
            return (-score, -min(len(u), 80))
        candidates.sort(key=_rank)
        return candidates[0]
    if any('university of virginia' in (ln or '').lower() for ln in lines[:80]):
        return 'University of Virginia'
    return ''

def detect_student_name(lines: List[str]) -> str:
    for line in lines[:160]:
        m = NAME_LINE_PAT.search(line)
        if m: return m.group('name').strip()
    for line in lines[:160]:
        s = line.strip()
        if _looks_like_person(s): return s
    for line in lines[:160]:
        s = line.strip()
        if ',' in s and _looks_like_person(s.replace(',', ' ')): return s
    return ''

def detect_major(lines: List[str]) -> str:
    major = ''
    for line in lines:
        m = MAJOR_PAT.search(line)
        if m:
            txt = m.group('major').strip()
            if not re.search(r"No\s+Show|Never\s+Attended|Disciplin", txt, re.IGNORECASE):
                major = txt
    return major

def detect_degree_fields(lines: List[str]) -> Tuple[str, str, str]:
    degree = ''; date = ''; klass = ''
    for line in lines:
        if not degree:
            m1 = DEGREE_PAT.search(line)
            if m1: degree = m1.group('deg').strip()
        m2 = DEGREE_DATE_PAT.search(line)
        if m2: date = m2.group('date').strip()
        if not klass and DEGREE_CLASS_PAT.search(line):
            klass = line.strip()
    return degree, date, klass

def detect_cumulative_gpa(text: str) -> Tuple[str, Optional[str]]:
    # Prefer explicit cumulative pattern first
    m = CUM_GPA_PAT.search(text)
    if m:
        s = m.group('gpa').strip()
        if s.endswith('%'): return s, 'percent_100'
        if '/' in s:
            try:
                den = float(s.split('/')[-1])
                if 4.2 <= den <= 4.3: return s, 'four_point_three'
                if 4.0 < den <= 5.0:  return s, 'five_point'
                if 5.0 < den <= 6.0:  return s, 'six_point'
                if 9.0 < den <= 10.0: return s, 'ten_point'
                if 19.0 < den <= 20.0: return s, 'twenty_point'
                if 99.0 < den <= 100.0:return s, 'percent_100'
            except Exception: pass
        return s, None

    # Fallback: only consider lines that *mention* GPA; skip term/semester & date-like lines
    for line in text.splitlines():
        lo = line.lower()
        if 'gpa' not in lo: continue
        if any(tok in lo for tok in ('term gpa','semester gpa','session gpa','term average','semester average')):
            continue
        if any(m in lo for m in MONTH_TOKENS) or any(tok in lo for tok in ('printed','print date','issued','dob','birth','page','registrar')):
            continue
        m2 = GPA_FALLBACK_PAT.search(line)
        if not m2: continue
        s = m2.group('gpa').strip()
        if s.endswith('%'): return s, 'percent_100'
        if '/' in s:
            try:
                den = float(s.split('/')[-1])
                if 4.2 <= den <= 4.3: return s, 'four_point_three'
                if 4.0 < den <= 5.0:  return s, 'five_point'
                if 5.0 < den <= 6.0:  return s, 'six_point'
                if 9.0 < den <= 10.0: return s, 'ten_point'
                if 19.0 < den <= 20.0: return s, 'twenty_point'
                if 99.0 < den <= 100.0:return s, 'percent_100'
            except Exception: pass
        return s, None

    return '', None

# -------------------------------- Course parsing -------------------------------

def _ok_subject_token(subj: str) -> bool:
    if not subj: return False
    s = subj.upper()
    if s in STOP_SUBJECTS: return False
    return bool(re.fullmatch(r'[A-Z&]{2,6}', s))

def _is_reasonable_credits(x) -> bool:
    try:
        v = float(x); return 0.25 <= v <= 10.0
    except Exception:
        return False

def _is_reasonable_grade_token(g: str) -> bool:
    G = (g or '').upper()
    if G in PF_TOKENS: return True
    if G in LETTER_GRADE: return True
    if G.endswith('%'):
        try: v = float(G[:-1]); return 0.0 <= v <= 100.0
        except Exception: return False
    try: v = float(G); return 0.0 <= v <= 100.0
    except Exception: return False

def _title_is_noise(s: str) -> bool:
    lo = (s or '').lower()
    return any(tok in lo for tok in TITLE_NOISE)

def _build_module(subj, num, title, grade_raw, credits, term) -> Optional[dict]:
    if not _ok_subject_token(subj): return None
    try: n = int(num)
    except Exception: return None
    t = (title or '').strip()
    if not t or len(t) < 3 or _title_is_noise(t): return None
    if not _is_reasonable_credits(credits): return None
    G = str(grade_raw or '').strip()
    pf = G.upper() in PF_TOKENS
    if not pf and G and not _is_reasonable_grade_token(G): return None

    grade_scale = None
    grade_value: Any = None if pf else G
    if grade_value is not None and isinstance(grade_value, str):
        if grade_value.upper() in LETTER_GRADE:
            grade_scale = 'us_4'
        elif grade_value.endswith('%'):
            try: grade_scale = 'percent_100'; grade_value = float(grade_value[:-1])
            except Exception: pass
        else:
            try: grade_value = float(grade_value)
            except Exception: return None

    return {
        'subject': subj.upper(),
        'number': n,
        'title': t,
        'grade': None if pf else grade_value,
        'grade_scale': grade_scale,
        'credits': float(credits),
        'credit_unit': 'credits',
        'term': term,
        'pf': bool(pf),
    }

def parse_modules(lines: List[str]) -> Tuple[List[dict], int]:
    modules: List[dict] = []
    current_term: Optional[str] = None
    semesters = set()

    # Pass 1: strict patterns (now includes UVA-friendly first)
    for raw in lines:
        l = raw.strip()
        if not l: continue
        t = TERM_PAT.search(l)
        if t:
            current_term = f"{t.group(1).title()} {t.group(2)}"
            continue
        matched = False
        for pat in (COURSE_PAT_UVA, COURSE_PAT_1, COURSE_PAT_2, COURSE_PAT_3):
            m = pat.match(l)
            if not m: continue
            gd = m.groupdict()
            mod = _build_module(
                gd.get('subject'), gd.get('number'),
                gd.get('title') or '',
                gd.get('grade') or '',
                gd.get('credits') or '3.0',
                current_term
            )
            if mod:
                modules.append(mod)
                if mod['grade'] is not None or mod['pf']:
                    semesters.add(current_term or '__unknown__')
            matched = True
            break
        if matched: continue

    # Pass 2: sliding-window catch-all
    if not modules:
        N = len(lines)
        for i in range(N):
            s = lines[i].strip()
            if not s: continue
            t = TERM_PAT.search(s)
            if t:
                current_term = f"{t.group(1).title()} {t.group(2)}"
                continue
            h = ANY_COURSE_HEAD.search(s)
            if not h: continue
            subj = h.group(1)
            try: num = int(h.group(2))
            except Exception: continue

            title = ''
            post = s[h.end():].strip(" -–:\t")
            if post and not re.search(r"\b(Grade|Credits?|Units?|Hours?)\b", post, re.IGNORECASE):
                if not _title_is_noise(post): title = post

            grade_raw = ''
            credits: Optional[str] = None
            for j in range(i, min(i+3, N)):
                ln = lines[j].strip()
                if not grade_raw:
                    mg = GRADE_TOKEN.search(ln)
                    if mg: grade_raw = mg.group(0)
                if credits is None:
                    mc = CREDITS_TOKEN.search(ln)
                    if mc: credits = mc.group(1)
                if credits is None:
                    mt = TRAILING_CRED.search(ln)
                    if mt and len(ln) - len(mt.group(1)) > 5:
                        credits = mt.group(1)
                if not title:
                    if not re.search(r"\b(Grade|Credits?|Units?|Hours?)\b", ln, re.IGNORECASE) and len(ln) > 4:
                        if not _title_is_noise(ln): title = ln
            if credits is None: credits = '3.0'

            mod = _build_module(subj, num, title, grade_raw, credits, current_term)
            if mod:
                modules.append(mod)
                if mod['grade'] is not None or mod['pf']:
                    semesters.add(current_term or '__unknown__')

    return modules, len(semesters)

# ---------------------------- Main row building -------------------------------

def _strip_transfer_prefix(s: str) -> str:
    return re.sub(r'^\s*(Transfer\s*Credit\s*from|Transfer)\s+', '', s or '', flags=re.IGNORECASE)

def _parse_hint_map(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not s: return out
    parts = [p for p in s.split(';') if p.strip()]
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            out[k.strip().lower()] = v.strip()
    return out

def _collect_files(root: str, patterns: List[str], recursive: bool) -> List[str]:
    import glob
    files: List[str] = []
    if not recursive:
        for pat in patterns: files.extend(glob.glob(os.path.join(root, pat)))
    else:
        for dirpath, _, _ in os.walk(root):
            for pat in patterns:
                files.extend(glob.glob(os.path.join(dirpath, pat)))
    return sorted(set(files))

def process_one(path: str, *, institutions_limit: int, default_country_hint: Optional[str],
                hint_map: Dict[str, str] | str, ocr: bool, ocr_threshold: int, ocr_lang: str,
                debug_dump_dir: Optional[str], verbose: bool) -> Dict[str, object]:
    text = read_text_from_file(path, ocr=ocr, ocr_threshold=ocr_threshold, ocr_lang=ocr_lang,
                               debug_dump_dir=debug_dump_dir, verbose=verbose)
    lines = [ln.rstrip('\n') for ln in text.splitlines() if ln.strip()]
    base = os.path.basename(path)

    app_key = application_key_from_filename(base)
    uni = detect_university(lines)
    name = detect_student_name(lines)

    if (not name) and _looks_like_person(uni):
        name, uni = uni, ''

    if uni:
        u0 = html.unescape(uni).strip()
        lo = u0.lower()
        if any(tok in lo for tok in ADDRESS_TOKENS) or any(tok in lo for tok in STUDENT_ID_TOKENS) or UVA_SEGMENT_PAT.search(u0):
            uni = ''
        else:
            uni = _strip_transfer_prefix(u0)

    if not uni and any('university of virginia' in ln.lower() for ln in lines[:80]):
        uni = 'University of Virginia'

    major = detect_major(lines)
    degree, degree_date, degree_class = detect_degree_fields(lines)
    cum_raw, scale_hint = detect_cumulative_gpa(text)

    uni_l = (uni or '').lower()
    path_l = path.lower()
    hmap = _parse_hint_map(hint_map) if isinstance(hint_map, str) else (hint_map or {})
    for k, v in hmap.items():
        if k and (k in uni_l or k in path_l):
            if not scale_hint: scale_hint = v
            break

    modules, n_semesters = parse_modules(lines)

    insts: List[Tuple[str, str, str]] = []
    for l in lines:
        if re.search(r"Transfer\s+Credit\s+from", l, re.IGNORECASE):
            nm = _strip_transfer_prefix(l).strip()
            if nm and len(insts) < institutions_limit:
                insts.append((nm, '', ''))

    row: Dict[str, object] = {
        'application_key': app_key,
        'student_name': name,
        'university_name': uni,
        'major_most_recent': major,
        'degree_most_recent': degree,
        'degree_awarded_date': degree_date,
        'degree_classification': degree_class,
        'num_semesters_with_grades': n_semesters,
        'cum_gpa_most_recent': cum_raw,
        'scale_hint': scale_hint or '',
        'country_hint': default_country_hint or '',
        'module_records': json.dumps(modules, ensure_ascii=False),
        'date_of_birth': '',
    }
    for idx, (nm, dates, deg) in enumerate(insts, start=1):
        row[f'institution_{idx}_name'] = nm
        row[f'institution_{idx}_dates'] = dates
        row[f'institution_{idx}_degree'] = deg

    if verbose:
        print(f"[parsed] {base} -> key={app_key} name={name!r} uni={row['university_name']!r} "
              f"mods={len(modules)} gpa={cum_raw!r} scale_hint={scale_hint!r}")
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', dest='in_dir', help='Directory with transcript files')
    p.add_argument('--folder', dest='folder', help='Alias of --in-dir')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--glob', default='*.pdf;*.txt')
    p.add_argument('--out', default='_tmp_new_rows.csv')
    p.add_argument('--modules-out', default='_tmp_modules.csv')
    p.add_argument('--institutions-limit', type=int, default=4)
    p.add_argument('--country-hint', default='')
    p.add_argument('--ocr', action='store_true')
    p.add_argument('--ocr-threshold', type=int, default=200)
    p.add_argument('--ocr-lang', default='eng')
    p.add_argument('--debug-dump', default=None)
    p.add_argument('--hint-map', default='')
    p.add_argument('--verbose', action='store_true')

    # accept/ignore pipeline args for R compatibility
    p.add_argument('--master-in', default=None)
    p.add_argument('--master-out', default=None)
    p.add_argument('--final-out', default=None)
    p.add_argument('--merge-mode', default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    in_dir = args.in_dir or args.folder
    if not in_dir:
        print('ERROR: --in-dir/--folder is required', file=sys.stderr); sys.exit(2)

    patterns = [g.strip() for g in args.glob.split(';') if g.strip()]
    files = _collect_files(in_dir, patterns, args.recursive)

    rows: List[Dict[str, object]] = []
    modules_rows: List[Dict[str, object]] = []

    if not files:
        print('No input files found.')

    for fp in files:
        try:
            row = process_one(fp,
                institutions_limit=args.institutions_limit,
                default_country_hint=(args.country_hint or None),
                hint_map=args.hint_map,
                ocr=args.ocr, ocr_threshold=args.ocr_threshold, ocr_lang=args.ocr_lang,
                debug_dump_dir=args.debug_dump, verbose=args.verbose,
            )
            rows.append(row)
            app_key = row.get('application_key', '')
            try: mods = json.loads(row.get('module_records','[]'))
            except Exception: mods = []
            for m in mods:
                modules_rows.append({
                    'application_key': app_key,
                    'subject': m.get('subject',''),
                    'number': m.get('number',''),
                    'title': m.get('title',''),
                    'grade': m.get('grade',''),
                    'grade_scale': m.get('grade_scale',''),
                    'credits': m.get('credits',''),
                    'credit_unit': m.get('credit_unit',''),
                    'term': m.get('term',''),
                    'pf': m.get('pf',''),
                })
        except Exception as e:
            print(f"[warn] Failed to parse {fp}: {e}")

    cols = set()
    for r in rows: cols.update(r.keys())

    pref = [
        'application_key','student_name','university_name','major_most_recent',
        'degree_most_recent','degree_awarded_date','degree_classification',
        'num_semesters_with_grades','cum_gpa_most_recent','scale_hint','country_hint',
        'module_records','date_of_birth',
    ]
    for i in range(1, 10):
        nm = f'institution_{i}_name'
        if any(nm in r for r in rows):
            pref.extend([nm, f'institution_{i}_dates', f'institution_{i}_degree'])
    ordered = pref + [c for c in sorted(cols) if c not in pref]

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows: writer.writerow(r)

    m_fields = ['application_key','subject','number','title','grade','grade_scale','credits','credit_unit','term','pf']
    with open(args.modules_out, 'w', newline='', encoding='utf-8') as fm:
        mwriter = csv.DictWriter(fm, fieldnames=m_fields)
        mwriter.writeheader()
        for mr in modules_rows: mwriter.writerow(mr)

    print(f"Wrote {args.out} (rows={len(rows)}) and {args.modules_out} (rows={len(modules_rows)})")

if __name__ == '__main__':
    main()
