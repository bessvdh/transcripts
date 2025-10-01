# bulk_ingest.py  -*- coding: utf-8 -*-
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
import re
import tempfile
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# =========================
# Regex & constants
# =========================

# Term header like "Fall 2023"
TERM_PAT = re.compile(r"\b(Fall|Spring|Summer|Winter|Autumn)\s+(\d{4})\b", re.IGNORECASE)

# Strict course patterns (named groups)
COURSE_PAT_1 = re.compile(
    r"^(?P<subject>[A-Za-z]{2,6})\s*-?\s*(?P<number>\d{3,5})\s+"
    r"(?P<title>.+?)\s{2,}(?P<grade>(?:[A-F][\+\-]?|P|S|U|CR|NC|Pass|Fail|\d+(?:\.\d+)?%?))\s+"
    r"(?P<credits>\d+(?:\.\d+)?)\s*$"
)
COURSE_PAT_2 = re.compile(
    r"^(?P<subject>[A-Za-z]{2,6})\s*-?\s*(?P<number>\d{3,5})\s*[-–:]?\s*(?P<title>.+?)\s+"
    r"(?:Grade\s*[: ]\s*(?P<grade>(?:[A-F][\+\-]?|P|S|U|CR|NC|Pass|Fail|\d+(?:\.\d+)?%?)))"
    r"(?:\s+Credits?\s*[: ]\s*(?P<credits>\d+(?:\.\d+)?))?"
    r"\s*$"
)
COURSE_PAT_3 = re.compile(
    r"^(?P<subject>[A-Za-z]{2,6})\s*-?\s*(?P<number>\d{3,5})\s+"
    r"(?P<title>.+?)\s{2,}(?P<credits>\d+(?:\.\d+)?)\s+(?P<grade>(?:[A-F][\+\-]?|P|S|U|CR|NC|Pass|Fail|\d+(?:\.\d+)?%?))\s*$"
)

# Header detector for sliding window
ANY_COURSE_HEAD = re.compile(r"\b([A-Z]{2,6})\s*[-\t ]?(\d{3,5})\b")
NEXT_COURSE_HEAD = re.compile(r"\b([A-Z]{2,6})\s*[- ]?\d{3,5}\b")

# Exclusions/Noise
EXCLUDE_PATTERNS = [
    re.compile(r'\b(Date|Issued|Credential|Awarded|Transfer\s+Credit)\b', re.IGNORECASE),
    re.compile(r'\bSuite\s+\d+|Street|Avenue|Monroe|Address|Phone|Fax\b', re.IGNORECASE),
    re.compile(r'\b(Student\s+(ID|Name|Number)|University\s+of|College\s+of)\b', re.IGNORECASE),
    re.compile(r'^\d{2}-[A-Z]{3}-\d{4}$'),
    re.compile(r'^\d{4}$'),
    re.compile(r'COVD\s*019', re.IGNORECASE),
    re.compile(r'\b(Fall|Spring|Summer|Winter)\s+\d{4}\s+(as|term|semester)\b', re.IGNORECASE),
    re.compile(r'\b(Fall|Spring|Summer|Winter)\s+as\b', re.IGNORECASE),
    re.compile(r'^\d+\s*credits?\s*$', re.IGNORECASE),
    re.compile(r'History\s+of\s+\w+\s+Since\s+\d{4}', re.IGNORECASE),
    re.compile(r'\bTERM\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\b(THE|OF|AND|FOR|WITH|FROM|TO)\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\bAGES\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\bSINCE\s+\d{4}\b', re.IGNORECASE),
]

VALID_SUBJECTS = {
    'MATH', 'MTH', 'STAT', 'STS', 'STA', 'ECON', 'ECN', 'EC', 'ECO', 'APMA', 'AM',
    'PHYS', 'CHEM', 'BIOL', 'HIST', 'ENGL', 'PSYC', 'SOCI', 'PHIL', 'POLI', 'GOVT',
    'COMM', 'CS', 'ENGR', 'SPAN', 'FREN', 'GERM', 'ITAL', 'CHIN', 'JAPN', 'RELI',
    'ARTH', 'MUSC', 'THEA', 'GEOG', 'ANTH', 'NURS', 'EDUC', 'BUSN', 'ACCT', 'FINN',
    'MKTG', 'MGMT', 'WMST', 'AFAM', 'PLAP', 'PLCP', 'PLIR', 'LING', 'CLAS', 'ASTR',
    'CHBE', 'MCB', 'FSHN', 'RHET', 'FAA', 'GE', 'CPSC', 'IE', 'DSCI', 'INBU', 'FIN',
    'ITDS', 'PSCI', 'SPIA', 'ENG', 'HIS', 'PS', 'REL', 'CJ', 'RSM', 'FYS', 'PE', 'EL', 'FR', 'IS', 'KINE', 'LPPL', 'LPPP'
}
INVALID_SUBJECTS = {
    'TERM', 'THE', 'OF', 'AND', 'FOR', 'WITH', 'FROM', 'TO',
    'AGES', 'SINCE', 'DATE', 'YEAR', 'FALL', 'SPRING', 'SUMMER', 'WINTER',
    'AUG', 'USC', 'ACT', 'VA'
}

# Grade/Credit tokens
NUMERIC_GRADE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\b")
CREDITS_TOKEN = re.compile(r"(?:Credits?|Units?|Hours?)\s*[: ]\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)

GRADE_TOKEN_LOOSE = re.compile(
    r"(?i)\b(?:grade|final\s*grade|grd|mark|score)\s*[:\-]?\s*"
    r"([A-F][\+\-]?|P|S|U|CR|NC|Pass|Fail|\d{1,3}(?:\.\d+)?%?)\b"
)
LETTER_OR_PF = re.compile(r"\b([A-F][\+\-]?|P|S|U|CR|NC|Pass|Fail)\b")
NUMERIC_SCORE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)%?\b")
CREDITS_ANY = re.compile(r"(?i)\b(?:credits?|units?|credit\s*hours?|hrs?)\b\s*[:\-]?\s*(\d+(?:\.\d+)?)")
TRAILING_CRED_LIKE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:cr|hrs?)?\b", re.IGNORECASE)

# ---- Title cleanup helpers ----
TITLE_GRADE_OR_CRED = re.compile(
    r"(?i)\b(grade|final\s*grade|grd|mark|score|credits?|units?|hours?|credit\s*hours?)\b"
)
TITLE_STOP_TOKENS = re.compile(
    r"(?i)\b(major\s*:|minor\s*:|concentration\s*:|specialization\s*:|program\s*:|house\s*:|advising\s*seminars\b|intro\s*to\s*college\b|primary\s*college\b|university\s*seminar\b)"
)
TRAILING_NOISE = re.compile(r"(?:\b[A-F][\+\-]?\b|\b\d+(?:\.\d+)?\b)\s*$")

# GPA detection
CUM_GPA_PAT = re.compile(
    r"\b(?:Cumulative\s*(?:GPA|Average|CGPA)|Overall\s*GPA|CGPA|UVA\s*GPA|GPA\s*Cumulative|Total\s*GPA)\b"
    r"[:\s]*"
    r"(?P<gpa>(\d{1,2}(?:\.\d+)?(?:/\d{1,2}(?:\.\d+)?)?)|(\d{1,3}(?:\.\d+)?%))",
    re.IGNORECASE
)
STANDALONE_GPA = re.compile(r"\bGPA[:\s]*(\d{1,2}\.\d{2,3})\b", re.IGNORECASE)
TRANSCRIPT_GPA = re.compile(r"\b(\d\.\d{2,3})\s*(?:GPA|gpa)\b", re.IGNORECASE)

# Name/major/degree patterns
NAME_LINE_PAT = re.compile(r"\b(?:Student\s*Name|Name)\s*[:\-]\s*(?P<name>.+)$", re.IGNORECASE)
MAJOR_ONLY_PAT = re.compile(r"\bMajor\s*[:\-]\s*(?P<major>[^,\n]{3,120})", re.IGNORECASE)
MAJOR_LABEL = re.compile(r"\b(Major|Plan|Program|Degree\s*Plan)\s*[:\-]\s*$", re.IGNORECASE)
MAJOR_PAT = re.compile(r"\b(Major|Plan|Program|Degree\s*Plan)\s*[:\-]\s*(?P<major>[^,\n]{3,120})", re.IGNORECASE)
UNDECLARED_PAT = re.compile(r"(?i)\b(undeclared|undecided|pre[-\s]*major)\b")
DEGREE_WORDS = re.compile(r"(?i)\b(bachelor|master|ph\.?d|ba|bs|b\.?sc|m\.?a|m\.?s|msc|beng|meng)\b")

DEGREE_PAT = re.compile(r"\b(Degree|Degree\s*Awarded|Bachelor|Master|PhD|B\.?[AS]\.?|M\.?[AS]\.?)\s*[:\-]?\s*(?P<deg>[^,\n]{3,120})", re.IGNORECASE)
DEGREE_DATE_PAT = re.compile(r"\b(Awarded|Conferred|Graduation\s*Date|Graduated)\s*[:\-]\s*(?P<date>([A-Za-z]{3,9}\s+\d{4})|(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}))\b", re.IGNORECASE)
DEGREE_CLASS_PAT = re.compile(r"\b(Honou?rs?|Honor:|Class|Cum\s*Laude|Magna|Summa|Distinction|Dean'?s\s*List)\b.*", re.IGNORECASE)

# University patterns
UNIVERSITY_PAT = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+(University|College|Institute)\b", re.IGNORECASE)
COMMUNITY_COLLEGE_PAT = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+Community\s+College\b", re.IGNORECASE)

ADDRESS_TOKENS = ("street","st ","st.","road","rd ","rd.","avenue","ave","ave.","suite","ste","po box","p.o.","blvd","drive","dr ")
STUDENT_ID_TOKENS = ("student no","student id","id:","sid","uvid")

# Subject family mapping
SUBJECT_FAMILY = {
    'ECON': 'ECON', 'ECN': 'ECON', 'EC': 'ECON', 'ECO': 'ECON',
    'STAT': 'STAT', 'STS': 'STAT', 'STA': 'STAT',
    'MATH': 'MATH', 'APMA': 'MATH', 'AM': 'MATH', 'MTH': 'MATH'
}

LETTER_GRADE = set("A A+ A- B+ B B- C+ C C- D+ D D- F".split())
PF_TOKENS = {"P", "S", "U", "CR", "NC", "Pass", "Fail"}

# =========================
# Utility functions
# =========================

def _nonspace_len(s: str) -> int:
    return len(re.sub(r"\s+", "", s or ''))

def _preprocess_pdf_for_ocr(pdf_path: str, temp_dir: str) -> str:
    """Preprocess PDF to improve OCR quality."""
    preprocessed_path = os.path.join(temp_dir, 'preprocessed.pdf')
    
    # Try to use ghostscript to improve image quality
    if shutil.which('gs'):
        gs_cmd = [
            'gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pdfwrite',
            '-dColorImageResolution=300', '-dGrayImageResolution=300',
            '-dMonoImageResolution=300', '-dPDFSETTINGS=/printer',
            '-sOutputFile=' + preprocessed_path, pdf_path
        ]
        try:
            subprocess.run(gs_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return preprocessed_path
        except Exception:
            pass
    
    # If ghostscript fails, return original
    return pdf_path
  
def _ocr_with_ocrmypdf_enhanced(pdf_path: str, lang: str, verbose: bool = False) -> str:
    """Enhanced OCR with ocrmypdf using better settings."""
    if shutil.which('ocrmypdf') is None:
        return ''
    
    with tempfile.TemporaryDirectory() as td:
        # Preprocess PDF
        preprocessed_pdf = _preprocess_pdf_for_ocr(pdf_path, td)
        
        sidecar = os.path.join(td, 'sidecar.txt')
        outpdf = os.path.join(td, 'out.pdf')
        
        # Enhanced OCR command with better settings
        cmd = [
            'ocrmypdf',
            '--sidecar', sidecar,
            '--force-ocr',  # OCR even if text already exists
            '--language', lang,
            '--deskew',  # Fix skewed scans
            '--clean',  # Clean up artifacts
            '--remove-background',  # Remove background
            '--optimize', '1',  # Light optimization
            '--oversample', '300',  # Higher DPI for better accuracy
            '--skip-text',  # Skip existing text layers
            preprocessed_pdf, outpdf
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if verbose:
                print(f"[ocr] ocrmypdf succeeded for {os.path.basename(pdf_path)}")
            
            if os.path.exists(sidecar):
                with open(sidecar, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"[ocr] ocrmypdf timeout for {os.path.basename(pdf_path)}")
        except Exception as e:
            if verbose:
                print(f"[ocr] ocrmypdf failed for {os.path.basename(pdf_path)}: {e}")
    
    return ''

def _ocr_with_tesseract_enhanced(pdf_path: str, lang: str, verbose: bool = False) -> str:
    """Enhanced OCR with tesseract using better settings."""
    if shutil.which('tesseract') is None:
        return ''
    
    with tempfile.TemporaryDirectory() as td:
        # Convert PDF to images first for better OCR
        images_dir = os.path.join(td, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Use pdftoppm if available (better than direct PDF OCR)
        if shutil.which('pdftoppm'):
            try:
                # Convert PDF to PNG images at 300 DPI
                subprocess.run([
                    'pdftoppm', '-png', '-r', '300', 
                    pdf_path, os.path.join(images_dir, 'page')
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # OCR each image
                texts = []
                for img_file in sorted(os.listdir(images_dir)):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(images_dir, img_file)
                        
                        # Enhanced tesseract command
                        cmd = [
                            'tesseract', img_path, 'stdout',
                            '-l', lang,
                            '--psm', '1',  # Automatic page segmentation with OSD
                            '--oem', '3',  # Default OCR Engine Mode
                            '-c', 'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-+/\\%@#$&* \t\n',
                        ]
                        
                        try:
                            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                            text = result.stdout.decode('utf-8', errors='ignore')
                            texts.append(text)
                        except Exception:
                            continue
                
                if texts:
                    if verbose:
                        print(f"[ocr] tesseract+pdftoppm succeeded for {os.path.basename(pdf_path)}")
                    return '\n'.join(texts)
                    
            except Exception:
                pass
        
        # Fallback to direct PDF OCR
        cmd = [
            'tesseract', pdf_path, 'stdout',
            '-l', lang,
            '--psm', '1',
            '--oem', '3'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if verbose:
                print(f"[ocr] tesseract direct succeeded for {os.path.basename(pdf_path)}")
            return result.stdout.decode('utf-8', errors='ignore')
        except Exception as e:
            if verbose:
                print(f"[ocr] tesseract failed for {os.path.basename(pdf_path)}: {e}")
    
    return ''
  
def _ocr_with_python_libraries(pdf_path: str, verbose: bool = False) -> str:
    """OCR using Python libraries as fallback."""
    text_parts = []
    
    # Try with pdf2image + pytesseract if available
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=10)  # Limit pages
        
        for i, image in enumerate(images):
            try:
                # Use pytesseract with better config
                text = pytesseract.image_to_string(
                    image, 
                    config='--psm 1 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-+/\\%@#$&* \t\n'
                )
                text_parts.append(text)
            except Exception:
                continue
        
        if text_parts:
            if verbose:
                print(f"[ocr] pdf2image+pytesseract succeeded for {os.path.basename(pdf_path)}")
            return '\n'.join(text_parts)
            
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try with easyocr if available
    try:
        import easyocr
        from pdf2image import convert_from_path
        
        reader = easyocr.Reader(['en'])
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=5)  # Limit for speed
        
        for image in images:
            try:
                results = reader.readtext(image, detail=0)  # Just text, no coordinates
                text_parts.extend(results)
            except Exception:
                continue
        
        if text_parts:
            if verbose:
                print(f"[ocr] easyocr succeeded for {os.path.basename(pdf_path)}")
            return '\n'.join(text_parts)
            
    except ImportError:
        pass
    except Exception:
        pass
    
    return ''

def read_text_from_file(path: str, *, ocr: bool, ocr_threshold: int, ocr_lang: str,
                                debug_dump_dir: Optional[str], verbose: bool) -> str:
    """Enhanced text reading with better OCR capabilities."""
    ext = os.path.splitext(path)[1].lower()
    text = ''
    
    # Read existing text first
    if ext == '.txt':
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                text = f.read()
    elif ext == '.pdf':
        if fitz is not None:
            try:
                doc = fitz.open(path)
                parts = [page.get_text('text') for page in doc]
                doc.close()
                text = "\n".join(parts)
            except Exception:
                text = ''
    
    # Determine if OCR is needed
    nonspace_chars = len(re.sub(r'\s+', '', text or ''))
    needs_ocr = ocr and nonspace_chars < max(0, ocr_threshold)
    
    if needs_ocr and ext == '.pdf':
        if verbose:
            print(f"[ocr] Attempting OCR for {os.path.basename(path)} (current text: {nonspace_chars} chars)")
        
        # Try multiple OCR methods in order of preference
        ocr_results = []
        
        # Method 1: Enhanced ocrmypdf
        ocr_text1 = _ocr_with_ocrmypdf_enhanced(path, ocr_lang, verbose)
        if ocr_text1:
            ocr_results.append(('ocrmypdf', ocr_text1, len(re.sub(r'\s+', '', ocr_text1))))
        
        # Method 2: Enhanced tesseract
        if not ocr_results or len(re.sub(r'\s+', '', ocr_results[0][1])) < ocr_threshold:
            ocr_text2 = _ocr_with_tesseract_enhanced(path, ocr_lang, verbose)
            if ocr_text2:
                ocr_results.append(('tesseract', ocr_text2, len(re.sub(r'\s+', '', ocr_text2))))
        
        # Method 3: Python libraries fallback
        if not ocr_results or len(re.sub(r'\s+', '', ocr_results[0][1])) < ocr_threshold:
            ocr_text3 = _ocr_with_python_libraries(path, verbose)
            if ocr_text3:
                ocr_results.append(('python_ocr', ocr_text3, len(re.sub(r'\s+', '', ocr_text3))))
        
        # Choose the best OCR result
        if ocr_results:
            # Sort by character count (more text usually better)
            ocr_results.sort(key=lambda x: x[2], reverse=True)
            best_method, best_text, best_count = ocr_results[0]
            
            if best_count > nonspace_chars:
                text = best_text
                if verbose:
                    print(f"[ocr] Used {best_method}: {nonspace_chars} -> {best_count} chars")
            elif verbose:
                print(f"[ocr] OCR didn't improve text quality: {nonspace_chars} vs {best_count} chars")
        elif verbose:
            print(f"[ocr] All OCR methods failed for {os.path.basename(path)}")
    
    # Save debug output
    if debug_dump_dir:
        try:
            os.makedirs(debug_dump_dir, exist_ok=True)
            with open(os.path.join(debug_dump_dir, os.path.basename(path)+'.txt'), 'w',
                      encoding='utf-8', errors='ignore') as f:
                f.write(text)
        except Exception:
            pass
    
    return text
  
def application_key_from_filename(filename: str) -> str:
    # Prefer explicit APP-#### token, anywhere in the name
    m = re.search(r'\bAPP[-_ ]?(\d{4})\b', filename, re.IGNORECASE)
    if m:
        return m.group(1)
    # Else take the first 4-digit chunk in the filename
    m = re.search(r'(\d{4})', filename)
    if m:
        return m.group(1)
    # Fallback (old behavior)
    digits = re.sub(r'\D', '', filename)
    return digits[-4:].zfill(4) if digits else ''


def _looks_like_person(s: str) -> bool:
    t = re.sub(r'\s+', ' ', s or '').strip()
    if not t or any(ch.isdigit() for ch in t):
        return False
    if any(w.lower() in ('university','college','institute','polytechnic','seminar') for w in t.lower().split()):
        return False
    parts = t.split()
    if not (2 <= len(parts) <= 4):
        return False
    return sum(1 for w in parts if w[:1].isupper()) >= len(parts)-1

def _squash_repeats(s: str) -> str:
    s2 = re.sub(r'(?i)\b(University of Virginia)\b(?:\s+\1\b)+', r'\1', s or '')
    return re.sub(r'\s{2,}', ' ', s2).strip()

# =========================
# Title canonicalization
# =========================

def _clean_title(title: str) -> str:
    """Trim a course title so it doesn't swallow 'Major :', 'House:', grade/credits, or next course header."""
    t = html.unescape(str(title or ""))
    t = re.sub(r"\s{2,}", " ", t).strip(" \t-—–:|")
    if not t:
        return ""
    
    t = re.sub(r'^(?:[A-Z][a-zA-Z]+)\s+(?:UG|GR|Undergraduate|Graduate)\s+', '', t)
    # Also handle cases where campus is missing but "UG/GR" is present
    t = re.sub(r'^(?:UG|GR|Undergraduate|Graduate)\s+', '', t)

    # stop at grade/credits tokens
    t = TITLE_GRADE_OR_CRED.split(t)[0].strip(" \t-—–:|")
    # stop at admin-y tokens
    t = TITLE_STOP_TOKENS.split(t)[0].strip(" \t-—–:|")
    # stop at next course header
    m = NEXT_COURSE_HEAD.search(t)
    if m:
        t = t[:m.start()].strip(" \t-—–:|")
    # strip trailing standalone grade or number tokens repeatedly
    prev = None
    while t and t != prev:
        prev = t
        t = TRAILING_NOISE.sub("", t).strip(" \t-—–:|")
    t = re.sub(r"\s{2,}", " ", t).strip()
    # reasonable cap
    if len(t) > 120:
        t = t[:120].rsplit(" ", 1)[0]
    return t

def _canon_title(title: str) -> str:
    """Canonical form used only for de-dup keys."""
    t = _clean_title(title)
    # remove leftover subject/number echoes at the end
    t = re.sub(NEXT_COURSE_HEAD, "", t).strip()
    t = re.sub(r"\s{2,}", " ", t).strip().lower()
    return t

# =========================
# Field detectors
# =========================
# --- drop-in detect_university for bulk_ingest.py ---
U_ADMIN_NOISE = re.compile(
    r'(?i)\b('
    r'Intro\s*to\s*College|Intro\s*College|'
    r'College\s*Advising\s*Seminars?|University\s*Seminar|Primary\s*College|'
    r'Office\s*of\s*the\s*University|The\s*Goods\s*of\s*the\s*University'
    r')\b'
)
U_KEYWORD = re.compile(r'\b(University|Community\s+College|College|Institute)\b', re.IGNORECASE)

ALLCAPS_UNI = re.compile(r"\b([A-Z][A-Z ]{2,})\s+(UNIVERSITY|COLLEGE|INSTITUTE)\b")

def detect_university(lines: List[str]) -> str:
    # NEW: uppercase header catch (first ~60 lines)
    for s in lines[:60]:
        m = ALLCAPS_UNI.search(s.strip())
        if m:
            # canonicalize casing: 'EMORY UNIVERSITY' -> 'Emory University'
            head = (m.group(1).title() + ' ' + m.group(2).title()).strip()
            return _squash_repeats(head)

    # ...then fall back to your current logic (existing function body) ...
    # (keep your existing candidates/U_KEYWORD search, scoring, etc.)
    
    candidates = []
    uva_hint = False

    for l in lines[:200]:
        s = html.unescape(l or '').strip()
        if not s:
            continue
        lo = s.lower()
        if any(tok in lo for tok in STUDENT_ID_TOKENS + ADDRESS_TOKENS):
            continue
        if 'university of virginia' in lo:
            uva_hint = True

        m = UNIVERSITY_PAT.search(s) or COMMUNITY_COLLEGE_PAT.search(s)
        if m:
            cand = _squash_repeats(m.group(0))
            if not U_ADMIN_NOISE.search(cand):
                candidates.append(cand)
            continue

        if U_KEYWORD.search(s) and not U_ADMIN_NOISE.search(s):
            clean = _squash_repeats(re.sub(r'\s+', ' ', s).strip())
            if len(clean) < 100:
                candidates.append(clean)

    if uva_hint:
        return 'University of Virginia'

    if not candidates:
        return ''

    def score(name: str) -> tuple[int, int]:
        n = name.lower()
        pts = 0
        if 'university' in n: pts += 10
        if 'community college' in n: pts += 9
        if 'college' in n: pts += 6
        if 'institute' in n: pts += 4
        return (pts, -len(name))  # shorter among same type

    candidates.sort(key=score, reverse=True)
    # Filter out weak admin labels just in case
    top = next((c for c in candidates if not U_ADMIN_NOISE.search(c)), '')
    return top
  
  
def detect_student_name(lines: List[str]) -> str:
    for line in lines[:200]:
        m = NAME_LINE_PAT.search(line)
        if m:
            name = m.group('name').strip()
            name = re.sub(r'\s+(Student|ID|Number).*$','', name, flags=re.IGNORECASE)
            return name
    for line in lines[:200]:
        s = line.strip()
        if _looks_like_person(s):
            return s
    for line in lines[:200]:
        s = line.strip()
        if ',' in s and _looks_like_person(s.replace(',', ' ')):
            return s
    return ''

def _clean_major_basic(major: str) -> str:
    m = html.unescape(major or "")
    m = re.sub(r"\s{2,}", " ", m).strip(" \t-—–:|")
    return m

KNOWN_SINGLE_WORD_MAJORS = {
    'economics','physics','chemistry','biology','mathematics','statistics','history','sociology',
    'psychology','philosophy','anthropology','accounting','finance','marketing','management',
    'engineering','computer','linguistics','geography','english','music','art','nursing','education',
    'astronomy','neuroscience','biochemistry','microbiology','ecology','public','government','politics'
}

def detect_major(lines: List[str]) -> str:
    """
    Return the most recent declared major.
    Strategy:
      1) Collect all explicit 'Major:' values
      2) Fallback to Plan/Program/Degree Plan
      3) Filter 'Undeclared/Undecided/Pre-major' and degree-only labels
      4) Choose the last good candidate
    """
    candidates: List[Tuple[int, str, str]] = []  # (idx, tag, value)

    # Pass A: explicit "Major:"
    for i, raw in enumerate(lines):
        s = html.unescape(raw or '')
        m = MAJOR_ONLY_PAT.search(s)
        if m:
            val = _clean_major_basic(m.group('major'))
            if val:
                candidates.append((i, 'major', val))
        if MAJOR_LABEL.search(s) and re.search(r'(?i)\bmajor\b', s):
            for j in (1, 2):
                if i + j >= len(lines):
                    break
                nxt = _clean_major_basic(html.unescape(lines[i+j] or ''))
                if len(nxt) >= 3 and not re.match(r'^(Name|Student|Degree)\b', nxt, re.IGNORECASE):
                    candidates.append((i+j, 'major', nxt))
                    break

    # Pass B: Plan/Program/Degree Plan if nothing better
    if not any(tag == 'major' for _, tag, _ in candidates):
        for i, raw in enumerate(lines):
            s = html.unescape(raw or '')
            m = MAJOR_PAT.search(s)
            if m:
                val = _clean_major_basic(m.group('major'))
                if val:
                    candidates.append((i, 'plan', val))
            elif MAJOR_LABEL.search(s):
                for j in (1, 2):
                    if i + j >= len(lines):
                        break
                    nxt = _clean_major_basic(html.unescape(lines[i+j] or ''))
                    if len(nxt) >= 3 and not re.match(r'^(Name|Student|Degree)\b', nxt, re.IGNORECASE):
                        candidates.append((i+j, 'plan', nxt))
                        break

    if not candidates:
        return ''

    # Filter and choose latest good one
    filtered: List[Tuple[int, str]] = []
    for idx, tag, txt in candidates:
        if UNDECLARED_PAT.search(txt):
            continue
        if DEGREE_WORDS.search(txt) and len(txt.split()) <= 3:
            # "BA", "Bachelor of Science" etc. -> not a major
            continue
        if len(txt.split()) == 1 and txt.lower() not in KNOWN_SINGLE_WORD_MAJORS:
            # lone word noise (e.g., "Barcelona")
            continue
        filtered.append((idx, txt))

    chosen = (filtered[-1][1] if filtered else candidates[-1][2]).strip()
    return chosen

def detect_degree_fields(lines: List[str]) -> Tuple[str, str, str]:
    degree = ''; date = ''; klass = ''
    for line in lines[:400]:
        s = html.unescape(line or '')
        if not degree:
            m1 = DEGREE_PAT.search(s)
            if m1:
                degree = re.sub(r'\s+(Awarded|Conferred).*$', '', m1.group('deg').strip(), flags=re.IGNORECASE)
        m2 = DEGREE_DATE_PAT.search(s)
        if m2:
            date = m2.group('date').strip()
        if not klass and DEGREE_CLASS_PAT.search(s):
            klass = s.strip()[:120]
    return degree, date, klass

# Enhanced degree detection functions for bulk_ingest.py
# Replace the existing detect_degree_fields function with these improved versions

import re
from typing import Tuple, List, Optional
from datetime import datetime
import html

# Enhanced degree patterns
DEGREE_PATTERNS = [
    # Full degree names with variations
    re.compile(r'\b(Bachelor\s+of\s+(?:Arts|Science|Engineering|Business|Fine\s+Arts|Music)(?:\s+in\s+[^,\n]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+(?:Arts|Science|Business\s+Administration|Engineering|Education|Fine\s+Arts|Music|Public\s+Administration)(?:\s+in\s+[^,\n]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(Doctor\s+of\s+(?:Philosophy|Medicine|Education|Engineering|Veterinary\s+Medicine)(?:\s+in\s+[^,\n]{3,80})?)\b', re.IGNORECASE),
    
    # Abbreviated degrees
    re.compile(r'\b(B\.?[AS]\.?(?:\s+in\s+[^,\n]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(M\.?[AS]\.?(?:\s+in\s+[^,\n]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(Ph\.?D\.?(?:\s+in\s+[^,\n]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(M\.?B\.?A\.?)\b', re.IGNORECASE),
    re.compile(r'\b(B\.?Eng\.?|B\.?E\.?)\b', re.IGNORECASE),
    re.compile(r'\b(M\.?Eng\.?|M\.?E\.?)\b', re.IGNORECASE),
    
    # Common specific degrees
    re.compile(r'\b(Bachelor\s+of\s+Commerce|B\.?Com\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Bachelor\s+of\s+Laws|LL\.?B\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Juris\s+Doctor|J\.?D\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Bachelor\s+of\s+Education|B\.?Ed\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+Education|M\.?Ed\.?)\b', re.IGNORECASE),
]

# Enhanced date patterns
DATE_PATTERNS = [
    # Month Day, Year formats
    re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
    re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
    
    # Month Year formats
    re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE),
    re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{4})\b', re.IGNORECASE),
    
    # Numeric date formats
    re.compile(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b'),
    re.compile(r'\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b'),
    
    # Season Year formats
    re.compile(r'\b(Spring|Summer|Fall|Autumn|Winter)\s+(\d{4})\b', re.IGNORECASE),
]

# Context keywords that indicate degree information
DEGREE_CONTEXT_WORDS = [
    'degree', 'awarded', 'conferred', 'granted', 'earned', 'received', 
    'graduated', 'graduation', 'bachelor', 'master', 'doctor', 'diploma',
    'certificate', 'completion', 'completed'
]

DATE_CONTEXT_WORDS = [
    'awarded', 'conferred', 'granted', 'graduation', 'graduated', 'completed',
    'date', 'on', 'received', 'earned', 'degree date', 'completion date'
]

def _clean_degree_text(text: str) -> str:
    """Clean and normalize degree text."""
    text = html.unescape(text).strip()
    # Remove common prefixes/suffixes that aren't part of the degree
    text = re.sub(r'^(Degree[:\-\s]*|Program[:\-\s]*)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(with\s+honors?|cum\s+laude|magna\s+cum\s+laude|summa\s+cum\s+laude).*$', '', text, flags=re.IGNORECASE)
    return text.strip()

def _normalize_date(date_str: str) -> str:
    """Normalize date string to consistent format."""
    date_str = date_str.strip()
    
    # Month name to number mapping
    month_map = {
        'january': '01', 'jan': '01', 'february': '02', 'feb': '02',
        'march': '03', 'mar': '03', 'april': '04', 'apr': '04',
        'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07',
        'august': '08', 'aug': '08', 'september': '09', 'sep': '09', 'sept': '09',
        'october': '10', 'oct': '10', 'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    # Try to parse and normalize various date formats
    for pattern in DATE_PATTERNS:
        match = pattern.search(date_str)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if groups[0].lower() in month_map:  # Month name format
                    month = month_map[groups[0].lower()]
                    day = groups[1].zfill(2)
                    year = groups[2]
                    return f"{month}/{day}/{year}"
                elif groups[0].isdigit():  # Numeric format
                    # Handle different numeric formats
                    if len(groups[2]) == 4:  # MM/DD/YYYY or DD/MM/YYYY
                        return f"{groups[0].zfill(2)}/{groups[1].zfill(2)}/{groups[2]}"
                    else:  # YYYY/MM/DD
                        return f"{groups[1].zfill(2)}/{groups[2].zfill(2)}/{groups[0]}"
            elif len(groups) == 2:  # Month Year format
                if groups[0].lower() in month_map:
                    month = month_map[groups[0].lower()]
                    year = groups[1]
                    return f"{month}/01/{year}"  # Default to 1st of month
                elif groups[0].lower() in ['spring', 'summer', 'fall', 'autumn', 'winter']:
                    # Season to approximate month
                    season_map = {'spring': '05', 'summer': '07', 'fall': '12', 'autumn': '12', 'winter': '12'}
                    month = season_map[groups[0].lower()]
                    return f"{month}/01/{groups[1]}"
    
    return date_str  # Return original if no pattern matches

def detect_degree_fields_enhanced(lines: List[str]) -> Tuple[str, str, str]:
    """
    Enhanced degree detection with better accuracy.
    Returns (degree, date, classification)
    """
    degree_candidates = []
    date_candidates = []
    classification = ''
    
    # Search through lines with context awareness
    for i, line in enumerate(lines[:400]):  # Limit search to first 400 lines
        clean_line = html.unescape(line).strip()
        if not clean_line:
            continue
            
        lower_line = clean_line.lower()
        
        # Look for degree information
        for pattern in DEGREE_PATTERNS:
            matches = pattern.finditer(clean_line)
            for match in matches:
                degree_text = _clean_degree_text(match.group(1))
                
                # Score based on context
                context_score = 0
                
                # Check surrounding lines for context
                context_lines = []
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    context_lines.append(lines[j].lower())
                context = ' '.join(context_lines)
                
                # Higher score if degree context words are nearby
                for word in DEGREE_CONTEXT_WORDS:
                    if word in context:
                        context_score += 1
                
                # Prefer lines that explicitly mention "degree"
                if 'degree' in lower_line:
                    context_score += 3
                
                # Prefer more specific degree names
                if 'in ' in degree_text.lower():
                    context_score += 2
                
                degree_candidates.append((context_score, degree_text, i))
        
        # Look for dates with degree context
        for pattern in DATE_PATTERNS:
            matches = pattern.finditer(clean_line)
            for match in matches:
                date_text = match.group(0)
                
                # Score based on context
                context_score = 0
                
                # Check surrounding lines for context
                context_lines = []
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    context_lines.append(lines[j].lower())
                context = ' '.join(context_lines)
                
                # Higher score if date context words are nearby
                for word in DATE_CONTEXT_WORDS:
                    if word in context:
                        context_score += 1
                
                # Prefer dates near degree information
                if any(word in context for word in DEGREE_CONTEXT_WORDS):
                    context_score += 2
                
                # Prefer more recent dates (assuming current transcripts)
                try:
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
                    if year_match:
                        year = int(year_match.group(0))
                        if 2000 <= year <= 2030:  # Reasonable range for current students
                            context_score += 1
                except:
                    pass
                
                date_candidates.append((context_score, date_text, i))
        
        # Look for honors/classification
        if not classification:
            honors_patterns = [
                r'\b(summa\s+cum\s+laude)\b',
                r'\b(magna\s+cum\s+laude)\b',
                r'\b(cum\s+laude)\b',
                r'\b(with\s+(?:highest\s+)?honors?)\b',
                r'\b(with\s+distinction)\b',
                r'\b(dean\'?s\s+list)\b',
                r'\b(first\s+class\s+honours?)\b',
                r'\b(upper\s+second\s+class\s+honours?)\b',
                r'\b(lower\s+second\s+class\s+honours?)\b',
                r'\b(third\s+class\s+honours?)\b',
            ]
            
            for pattern in honors_patterns:
                match = re.search(pattern, lower_line)
                if match:
                    classification = match.group(1).title()
                    break
    
    # Select best candidates
    best_degree = ''
    if degree_candidates:
        degree_candidates.sort(key=lambda x: x[0], reverse=True)
        best_degree = degree_candidates[0][1]
    
    best_date = ''
    if date_candidates:
        date_candidates.sort(key=lambda x: x[0], reverse=True)
        best_date = _normalize_date(date_candidates[0][1])
    
    return best_degree, best_date, classification

# Additional helper function for better degree parsing
def extract_degree_from_header(lines: List[str]) -> str:
    """
    Extract degree information from document headers/titles.
    This catches cases where the degree is mentioned in the document title.
    """
    # Check first 10 lines for degree information
    for line in lines[:10]:
        clean_line = html.unescape(line).strip()
        lower_line = clean_line.lower()
        
        # Look for transcript headers that mention degrees
        if 'transcript' in lower_line and any(word in lower_line for word in ['bachelor', 'master', 'doctor']):
            for pattern in DEGREE_PATTERNS:
                match = pattern.search(clean_line)
                if match:
                    return _clean_degree_text(match.group(1))
    
    return ''

def detect_cumulative_gpa(text: str) -> Tuple[str, Optional[str]]:
    m = CUM_GPA_PAT.search(text)
    if m:
        s = m.group('gpa').strip()
        if s.endswith('%'):
            return s, 'percent_100'
        if '/' in s:
            try:
                den = float(s.split('/')[-1])
                if 4.2 <= den <= 4.3: return s, 'four_point_three'
                if 4.0 < den <= 5.0:   return s, 'five_point'
                if 5.0 < den <= 6.0:   return s, 'six_point'
                if 9.0 < den <= 10.0:  return s, 'ten_point'
                if 19.0 < den <= 20.0: return s, 'twenty_point'
                if 99.0 < den <= 100.: return s, 'percent_100'
            except Exception:
                pass
        return s, None
    m2 = STANDALONE_GPA.search(text)
    if m2:
        return m2.group(1), None
    m3 = TRANSCRIPT_GPA.search(text)
    if m3:
        return m3.group(1), None
    gpa_candidates = re.findall(r'\b([0-4]\.\d{2,3})\b', text)
    if gpa_candidates:
        return gpa_candidates[0], None
    return '', None
# Add this debug function to bulk_ingest.py for testing degree detection

def debug_degree_detection(path: str, lines: List[str], verbose: bool = True):
    """Debug function to test degree detection on a single file."""
    
    if verbose:
        print(f"\n=== DEBUGGING DEGREE DETECTION FOR {path} ===")
    
    # Test original detection
    degree_orig, date_orig, class_orig = detect_degree_fields(lines)
    
    # Test enhanced detection  
    degree_enh, date_enh, class_enh = detect_degree_fields_enhanced(lines)
    
    # Test header extraction
    degree_header = extract_degree_from_header(lines)
    
    if verbose:
        print(f"Original detection:")
        print(f"  Degree: {degree_orig!r}")
        print(f"  Date: {date_orig!r}")
        print(f"  Class: {class_orig!r}")
        
        print(f"\nEnhanced detection:")
        print(f"  Degree: {degree_enh!r}")
        print(f"  Date: {date_enh!r}")
        print(f"  Class: {class_enh!r}")
        
        print(f"\nHeader extraction:")
        print(f"  Degree: {degree_header!r}")
        
        print(f"\nRelevant lines containing degree keywords:")
        degree_keywords = ['degree', 'bachelor', 'master', 'doctor', 'awarded', 'conferred', 'graduation']
        for i, line in enumerate(lines[:100]):
            if any(keyword in line.lower() for keyword in degree_keywords):
                print(f"  Line {i:3d}: {line.strip()[:100]}...")
    
    return {
        'original': (degree_orig, date_orig, class_orig),
        'enhanced': (degree_enh, date_enh, class_enh),
        'header': degree_header
    }
    
# =========================
# Course parsing building blocks
# =========================

# find normalize_subject(...)
def normalize_subject(subj: str) -> str:
    s_raw = (subj or '').upper().strip()
    # NEW: drop campus suffixes like ECON_OX, MATH_CL, etc.
    if '_' in s_raw:
        s_raw = s_raw.split('_', 1)[0]
    # keep ampersands; remove all other non-letters (ECON&FIN -> ECON&FIN)
    s = re.sub(r"[^A-Z&]", "", s_raw)
    if s in INVALID_SUBJECTS:
        return ''
    return SUBJECT_FAMILY.get(s, s)

def is_valid_course_line(line: str, subject: str, number: int) -> bool:
    norm_subj = normalize_subject(subject)
    if not norm_subj or norm_subj in INVALID_SUBJECTS:
        return False
    for pattern in EXCLUDE_PATTERNS:
        if pattern.search(line):
            return False
    # OLD: if len(subject) > 6 or len(norm_subj) < 2: return False
    # NEW: rely on normalized length; allow up to 10 (e.g., ECONFIN, STATMATH)
    if len(norm_subj) < 2 or len(norm_subj) > 10:
        return False
    if not (100 <= number <= 99999):
        return False
    if re.search(r'^\s*\d{4}\s*$', line):
        return False
    if re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', line):
        return False
    return True

def is_term_line(line: str) -> bool:
    return bool(re.match(r'^\s*(Fall|Spring|Summer|Winter)\s+\d{4}\s*(as|term|semester)?\s*$', line, re.IGNORECASE))

def _parse_course_window(lines: List[str], i: int, subj: str, num: int, start_line: str) -> Optional[dict]:
    """Strict parser around the current line (or a joined window)."""
    if not is_valid_course_line(start_line, subj, num):
        return None

    # Title after "<SUBJ> <NUM>"
    title = ""
    mstart = re.search(rf"\b{subj}\s*[- ]?{num}\b", start_line, re.IGNORECASE)
    if mstart:
        title = _clean_title(start_line[mstart.end():])

    # Grade & Credits in a short window (this + next 2 lines)
    grade_raw = ''
    credits = None
    for j in range(i, min(i + 3, len(lines))):
        s = lines[j].strip()
        if not s:
            continue

        # Letter/PF grades
        if not grade_raw:
            for match in re.finditer(r'\b([A-F][\+\-]?|P|S|U|CR|NC|Pass|Fail)\b', s):
                candidate = match.group(1)
                start, end = match.span()
                if (start == 0 or s[start - 1].isspace()) and (end >= len(s) or s[end].isspace()):
                    grade_raw = candidate
                    break

        # Numeric grade with context
        if not grade_raw:
            for match in NUMERIC_GRADE.finditer(s):
                try:
                    val = float(match.group(1))
                except Exception:
                    continue
                if (0 <= val <= 100) or (0 <= val <= 4.3):
                    context = s[max(0, match.start() - 15):match.end() + 15].lower()
                    if any(w in context for w in ['grade', 'gpa', 'score', 'mark']) or val <= 4.3:
                        grade_raw = str(val)
                        break

        # Credits via explicit tokens
        if credits is None:
            cred = CREDITS_TOKEN.search(s)
            if cred:
                try:
                    cval = float(cred.group(1))
                    if 0 < cval <= 10:
                        credits = cval
                except Exception:
                    pass

        # Reasonable standalone numbers that look like credits
        if credits is None:
            for match in re.finditer(r'\b(0-6?)\b', s):
                try:
                    val = float(match.group(1))
                except Exception:
                    continue
                if 0.5 <= val <= 6.0:
                    # avoid years/dates
                    context = s[max(0, match.start() - 5):match.end() + 5]
                    if not re.search(r'\d{4}', context):
                        credits = val
                        break

    if credits is None or credits <= 0:
        credits = 3.0

    subj_clean = normalize_subject(subj)
    if not subj_clean:
        return None

    if not title and not grade_raw:
        return None

    pf = grade_raw.upper() in PF_TOKENS if isinstance(grade_raw, str) and grade_raw else False
    grade_scale = None
    grade_value = None if pf else grade_raw

    if grade_value is not None and isinstance(grade_value, str):
        if grade_value in LETTER_GRADE:
            grade_scale = 'us_4'
        elif grade_value.endswith('%'):
            grade_scale = 'percent_100'
            try:
                grade_value = float(grade_value[:-1])
            except Exception:
                pass
        else:
            try:
                fv = float(grade_value)
                grade_value = fv
                if 0.0 <= fv <= 4.3:    grade_scale = 'us_4'
                elif 4.3 < fv <= 5.0:   grade_scale = 'five_point'
                elif 5.0 < fv <= 6.0:   grade_scale = 'six_point'
                elif 6.0 < fv <= 10.0:  grade_scale = 'ten_point'
                elif 10.0 < fv <= 20.0: grade_scale = 'twenty_point'
                elif 20.0 < fv <= 100.: grade_scale = 'percent_100'
            except Exception:
                pass

    return {
        'subject': subj_clean,
        'number': num,
        'title': title.strip(),
        'grade': None if pf else grade_value,
        'grade_scale': grade_scale,
        'credits': credits,
        'credit_unit': 'credits',
        'pf': bool(pf),
    }

def _extract_grade_and_credits(window: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    s = " ".join(window.split())
    grade_value: Optional[str] = None
    credits_value: Optional[float] = None
    grade_scale: Optional[str] = None

    # 1) Grade tokens
    m = GRADE_TOKEN_LOOSE.search(s)
    if m:
        grade_value = m.group(1).strip()

    # 2) Letter/PF
    if grade_value is None:
        m2 = LETTER_OR_PF.search(s)
        if m2:
            grade_value = m2.group(1)

    # 3) Numeric (<= 100) as last resort
    if grade_value is None:
        for m3 in NUMERIC_SCORE.finditer(s):
            try:
                v = float(m3.group(1))
                if 0 <= v <= 100:
                    grade_value = m3.group(1) + ('%' if v > 5 else '')
                    break
            except Exception:
                pass

    # Credits
    c = CREDITS_ANY.search(s)
    if c:
        try:
            v = float(c.group(1))
            if 0 < v <= 10:
                credits_value = v
        except Exception:
            pass

    if credits_value is None:
        tail = TRAILING_CRED_LIKE.findall(s)
        for tok in reversed(tail):
            try:
                v = float(tok)
                if 0.5 <= v <= 10:
                    credits_value = v
                    break
            except Exception:
                pass

    # Scale from numeric/percent
    if grade_value:
        gv = grade_value.strip()
        if gv.endswith('%'):
            grade_scale = 'percent_100'
        else:
            try:
                fv = float(gv)
                if 0.0 <= fv <= 4.3:    grade_scale = 'us_4'
                elif 4.3 < fv <= 5.0:   grade_scale = 'five_point'
                elif 5.0 < fv <= 6.0:   grade_scale = 'six_point'
                elif 6.0 < fv <= 10.0:  grade_scale = 'ten_point'
                elif 10.0 < fv <= 20.0: grade_scale = 'twenty_point'
                elif 20.0 < fv <= 100.: grade_scale = 'percent_100'
            except Exception:
                pass

    return grade_value, credits_value, grade_scale

def _parse_course_fallback(window: str, subj_hint: str, num_hint: int) -> Optional[dict]:
    s = re.sub(r"[ \t]+", " ", window).strip()

    subj_clean = normalize_subject(subj_hint)
    if not subj_clean:
        return None

    mhead = re.search(rf"\b{subj_hint}\s*[- ]?{num_hint}\b", s)
    title = ""
    if mhead:
        title = _clean_title(s[mhead.end():])

    grade_value, credits_value, grade_scale = _extract_grade_and_credits(s)
    if credits_value is None:
        credits_value = 3.0

    if not title and not grade_value:
        return None

    pf = False
    if grade_value and grade_value.upper() in {"P","S","U","CR","NC","PASS","FAIL"}:
        pf = True

    return {
        'subject': subj_clean,
        'number': int(num_hint),
        'title': title,
        'grade': None if pf else grade_value,
        'grade_scale': grade_scale,
        'credits': float(credits_value),
        'credit_unit': 'credits',
        'pf': bool(pf),
    }

# =========================
# parse_modules
# =========================

def parse_modules(lines: List[str]) -> Tuple[List[dict], int]:
    modules: List[dict] = []
    current_term = None
    semesters = set()
    seen_keys = set()

    # Pass 1: strict patterns line-by-line
    for raw in lines:
        l = raw.strip()
        if not l:
            continue
        if is_term_line(l):
            t = TERM_PAT.search(l)
            if t:
                current_term = f"{t.group(1).title()} {t.group(2)}"
            continue
        t = TERM_PAT.search(l)
        if t:
            current_term = f"{t.group(1).title()} {t.group(2)}"
            continue

        for pat in (COURSE_PAT_1, COURSE_PAT_2, COURSE_PAT_3):
            m = pat.match(l)
            if not m:
                continue
            subj_raw = m.group('subject')
            try:
                num = int(m.group('number'))
            except Exception:
                break
            if not is_valid_course_line(l, subj_raw, num):
                break

            subj = normalize_subject(subj_raw)
            if not subj:
                break

            title = _clean_title((m.groupdict().get('title') or '').strip())
            grade_raw = (m.groupdict().get('grade') or '').strip()
            credits_raw = (m.groupdict().get('credits') or '').strip()

            credits = 3.0
            if credits_raw:
                try:
                    credits = float(credits_raw)
                    if credits > 10 or credits <= 0:
                        credits = 3.0
                except Exception:
                    credits = 3.0

            pf = grade_raw in PF_TOKENS
            grade_scale = None
            grade_value = None if pf else grade_raw

            if grade_value is not None and isinstance(grade_value, str):
                if grade_value in LETTER_GRADE:
                    grade_scale = 'us_4'
                elif grade_value.endswith('%'):
                    grade_scale = 'percent_100'
                    try:
                        grade_value = float(grade_value[:-1])
                    except Exception:
                        pass
                else:
                    try:
                        fv = float(grade_value)
                        grade_value = fv
                        if 0.0 <= fv <= 4.3:    grade_scale = 'us_4'
                        elif 4.3 < fv <= 5.0:   grade_scale = 'five_point'
                        elif 5.0 < fv <= 6.0:   grade_scale = 'six_point'
                        elif 6.0 < fv <= 10.0:  grade_scale = 'ten_point'
                        elif 10.0 < fv <= 20.0: grade_scale = 'twenty_point'
                        elif 20.0 < fv <= 100.: grade_scale = 'percent_100'
                    except Exception:
                        pass

            mod = {
                'subject': subj,
                'number': num,
                'title': title,
                'grade': None if pf else grade_value,
                'grade_scale': grade_scale,
                'credits': credits,
                'credit_unit': 'credits',
                'term': current_term,
                'pf': bool(pf),
            }

            # canonicalized title for dedupe key
            canon_title = _canon_title(mod.get('title', ''))

            key = (
                mod['subject'],
                mod['number'],
                canon_title,
                mod['term'],
                str(mod['grade']).upper()
            )
            if key in seen_keys:
                break
            seen_keys.add(key)

            modules.append(mod)
            if mod['grade'] is not None or mod['pf']:
                semesters.add(current_term or '__unknown__')
            break  # do not try other patterns on this line

    # Pass 2: multi-line window (up to 6 lines) + fallback
    if not modules:
        N = len(lines)
        for i in range(N):
            s0 = lines[i].strip()
            if not s0:
                continue
            if is_term_line(s0):
                t = TERM_PAT.search(s0)
                if t:
                    current_term = f"{t.group(1).title()} {t.group(2)}"
                continue
            t = TERM_PAT.search(s0)
            if t:
                current_term = f"{t.group(1).title()} {t.group(2)}"
                continue

            parts = [lines[j].strip() for j in range(i, min(N, i + 6))]
            window = ' '.join([p for p in parts if p])

            h = ANY_COURSE_HEAD.search(window)
            if not h:
                continue
            subj = h.group(1)
            try:
                num = int(h.group(2))
            except Exception:
                continue

            # First try strict on the window
            mod = _parse_course_window([window], 0, subj, num, window)
            if mod is None:
                # Fallback: looser extraction on the joined window
                mod = _parse_course_fallback(window, subj, num)
                if mod is None:
                    continue

            mod['term'] = current_term

            # canonicalized title for dedupe key
            canon_title = _canon_title(mod.get('title', ''))

            key = (
                mod['subject'],
                mod['number'],
                canon_title,
                mod['term'],
                str(mod['grade']).upper()
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            modules.append(mod)
            if mod['grade'] is not None or mod['pf']:
                semesters.add(current_term or '__unknown__')

    return modules, len(semesters)

# =========================
# IO helpers
# =========================

def _strip_transfer_prefix(s: str) -> str:
    return re.sub(r'^\s*(Transfer\s*Credit\s*from|Transfer)\s+', '', s or '', flags=re.IGNORECASE)

def _parse_hint_map(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not s:
        return out
    for p in [p for p in s.split(';') if p.strip()]:
        if '=' in p:
            k, v = p.split('=', 1)
            out[k.strip().lower()] = v.strip()
    return out

def _collect_files(root: str, patterns: List[str], recursive: bool) -> List[str]:
    import glob
    files: List[str] = []
    if not recursive:
        for pat in patterns:
            files.extend(glob.glob(os.path.join(root, pat)))
    else:
        for dirpath, _, _ in os.walk(root):
            for pat in patterns:
                files.extend(glob.glob(os.path.join(dirpath, pat)))
    return sorted(set(files))

# =========================
# Row builder
# =========================

# Replace these functions in your bulk_ingest.py file

def process_one(path: str, *, institutions_limit: int, default_country_hint: Optional[str],
                hint_map: Dict[str, str], ocr: bool, ocr_threshold: int, ocr_lang: str,
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
        if any(tok in lo for tok in ADDRESS_TOKENS) or any(tok in lo for tok in STUDENT_ID_TOKENS):
            uni = ''
        else:
            uni = _strip_transfer_prefix(_squash_repeats(u0))

    # UVA fallback (contextual)
    if not uni:
        for ln in lines[:80]:
            if 'university of virginia' in ln.lower():
                uni = 'University of Virginia'
                break

    major = detect_major(lines)
    
    # Use the enhanced degree detection
    degree, degree_date, degree_class = detect_degree_fields_ultra_enhanced(lines)
    
    # Fallback to header extraction if main detection failed
    if not degree:
        degree = extract_degree_from_header(lines)
    
    cum_raw, scale_hint = detect_cumulative_gpa(text)

    uni_l = (uni or '').lower()
    path_l = path.lower()
    for k, v in hint_map.items():
        if k in uni_l or k in path_l:
            if not scale_hint:
                scale_hint = v
            break

    modules, n_semesters = parse_modules(lines)

    # Enhanced institution extraction with better filtering
    insts = []
    seen_inst = set()
    for l in lines:
        if re.search(r"Transfer\s+Credit\s+from", l, re.IGNORECASE):
            nm = _strip_transfer_prefix(l).strip()
        else:
            m = UNIVERSITY_PAT.search(l) or COMMUNITY_COLLEGE_PAT.search(l)
            nm = m.group(0) if m else ''
        nm = _squash_repeats(nm)
        if not nm:
            continue
        if uni and nm.lower() == (uni or '').lower():
            continue
        # Filter out obvious non-institutions
        if any(word in nm.lower() for word in ['student', 'transcript', 'office', 'registrar']):
            continue
        if nm.lower() in seen_inst:
            continue
        seen_inst.add(nm.lower())
        insts.append((nm, '', ''))
        if len(insts) >= institutions_limit:
            break

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
        print(f"[parsed] {base} -> key={app_key}")
        print(f"  name={name!r}")
        print(f"  university={row['university_name']!r}")
        print(f"  major={major!r}")
        print(f"  degree={degree!r} (awarded: {degree_date!r})")
        if degree_class:
            print(f"  classification={degree_class!r}")
        print(f"  gpa={cum_raw!r} scale_hint={scale_hint!r}")
        print(f"  modules={len(modules)}")
        if modules:
            print(f"  Sample courses:")
            for m in modules[:3]:
                print(f"    {m['subject']} {m['number']}: {m['title'][:60]} (Grade: {m['grade']}, Credits: {m['credits']})")

    return row
  
  
import json

# More comprehensive degree patterns
COMPREHENSIVE_DEGREE_PATTERNS = [
    # Standard full degree names
    re.compile(r'\b(Bachelor\s+of\s+(?:Arts|Science|Engineering|Business|Fine\s+Arts|Music|Education|Laws|Commerce)(?:\s+in\s+[^,\n\r]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+(?:Arts|Science|Business\s+Administration|Engineering|Education|Fine\s+Arts|Music|Public\s+Administration|Public\s+Policy)(?:\s+in\s+[^,\n\r]{3,80})?)\b', re.IGNORECASE),
    re.compile(r'\b(Doctor\s+of\s+(?:Philosophy|Medicine|Education|Engineering|Veterinary\s+Medicine)(?:\s+in\s+[^,\n\r]{3,80})?)\b', re.IGNORECASE),
    
    # Common abbreviations with field
    re.compile(r'\b(B\.?A\.?\s+in\s+[^,\n\r]{3,80})\b', re.IGNORECASE),
    re.compile(r'\b(B\.?S\.?\s+in\s+[^,\n\r]{3,80})\b', re.IGNORECASE),
    re.compile(r'\b(M\.?A\.?\s+in\s+[^,\n\r]{3,80})\b', re.IGNORECASE),
    re.compile(r'\b(M\.?S\.?\s+in\s+[^,\n\r]{3,80})\b', re.IGNORECASE),
    
    # Standalone abbreviations (lower priority)
    re.compile(r'\b(Bachelor\s+of\s+Arts|B\.?A\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Bachelor\s+of\s+Science|B\.?S\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+Arts|M\.?A\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+Science|M\.?S\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Bachelor\s+of\s+Engineering|B\.?Eng\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+Business\s+Administration|M\.?B\.?A\.?)\b', re.IGNORECASE),
    
    # Other common degrees
    re.compile(r'\b(Bachelor\s+of\s+Commerce|B\.?Com\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Bachelor\s+of\s+Laws|LL\.?B\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Juris\s+Doctor|J\.?D\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Bachelor\s+of\s+Education|B\.?Ed\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Master\s+of\s+Education|M\.?Ed\.?)\b', re.IGNORECASE),
    re.compile(r'\b(Associate\s+of\s+(?:Arts|Science)|A\.?[AS]\.?)\b', re.IGNORECASE),
]

# Contextual patterns - degree mentioned with context words
CONTEXTUAL_DEGREE_PATTERNS = [
    re.compile(r'(?i)\b(?:degree|awarded|conferred|earned|received|granted|completed)\s*[:\-]?\s*([^,\n\r]{10,80}(?:bachelor|master|doctor|B\.?[AS]\.?|M\.?[AS]\.?|Ph\.?D\.?)[^,\n\r]{0,80})', re.IGNORECASE),
    re.compile(r'(?i)\b(bachelor|master|doctor)[^,\n\r]{0,80}(?:degree|awarded|conferred|earned|received|granted|completed)', re.IGNORECASE),
]

def clean_degree_text_enhanced(text: str) -> str:
    """Enhanced cleaning of degree text."""
    if not text:
        return ''
    
    # Handle JSON corruption - if we see JSON patterns, extract clean text before them
    if '"' in text and any(json_indicator in text.lower() for json_indicator in ['grade', 'subject', 'credits']):
        # Find the first quote that looks like JSON and cut there
        json_start = text.find('"')
        if json_start > 5:  # Keep some meaningful text before JSON starts
            text = text[:json_start]
    
    text = html.unescape(text).strip()
    
    # Remove common prefixes
    text = re.sub(r'^(Degree[:\-\s]*|Program[:\-\s]*|Major[:\-\s]*)', '', text, flags=re.IGNORECASE)
    
    # Remove dates that might be attached
    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', '', text)
    text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    
    # Remove honors (we handle these separately)
    text = re.sub(r'\b(?:with\s+)?(?:highest\s+)?honors?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:summa\s+)?(?:magna\s+)?cum\s+laude\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdean\'?s\s+list\b', '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_degree_comprehensive(lines: List[str]) -> str:
    """More comprehensive degree detection with multiple strategies."""
    candidates = []
    
    # Strategy 1: Look for explicit degree patterns
    for i, line in enumerate(lines[:200]):
        clean_line = html.unescape(line).strip()
        if not clean_line or len(clean_line) < 5:
            continue
            
        # Skip lines that look like course listings
        if re.match(r'^[A-Z]{2,6}\s+\d{3,5}', clean_line):
            continue
            
        # Check all comprehensive patterns
        for pattern in COMPREHENSIVE_DEGREE_PATTERNS:
            matches = pattern.finditer(clean_line)
            for match in matches:
                degree_text = clean_degree_text_enhanced(match.group(1))
                if degree_text and len(degree_text) >= 2:
                    score = 1
                    
                    # Higher score for lines with degree context
                    lower_line = clean_line.lower()
                    if any(word in lower_line for word in ['degree', 'awarded', 'conferred', 'graduated']):
                        score += 3
                    
                    # Higher score for more specific degrees (with field of study)
                    if ' in ' in degree_text.lower() or ' of ' in degree_text.lower():
                        score += 2
                    
                    # Prefer earlier lines (likely headers)
                    if i < 50:
                        score += 1
                    
                    candidates.append((score, degree_text, i))
    
    # Strategy 2: Look for contextual degree mentions
    for i, line in enumerate(lines[:200]):
        clean_line = html.unescape(line).strip()
        if not clean_line:
            continue
            
        for pattern in CONTEXTUAL_DEGREE_PATTERNS:
            match = pattern.search(clean_line)
            if match:
                degree_text = clean_degree_text_enhanced(match.group(1))
                if degree_text and len(degree_text) >= 5:
                    candidates.append((2, degree_text, i))
    
    # Strategy 3: Look in document headers/titles for degree info
    for i, line in enumerate(lines[:20]):
        clean_line = html.unescape(line).strip().lower()
        if 'transcript' in clean_line:
            # Look for degree mentions in transcript headers
            for pattern in COMPREHENSIVE_DEGREE_PATTERNS:
                match = pattern.search(line)
                if match:
                    degree_text = clean_degree_text_enhanced(match.group(1))
                    if degree_text:
                        candidates.append((4, degree_text, i))  # High score for header mentions
    
    # Strategy 4: Infer from major if no explicit degree found
    if not candidates:
        for i, line in enumerate(lines[:100]):
            clean_line = line.lower()
            if 'major' in clean_line and any(level in clean_line for level in ['bachelor', 'undergraduate', 'bs', 'ba']):
                # Try to infer degree type from context
                if any(sci_word in clean_line for sci_word in ['science', 'engineering', 'math', 'physics', 'chemistry']):
                    candidates.append((1, 'Bachelor of Science', i))
                else:
                    candidates.append((1, 'Bachelor of Arts', i))
                break
    
    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[2]), reverse=True)  # Sort by score (desc) then by line number (asc)
        return candidates[0][1]
    
    return ''

def detect_degree_fields_ultra_enhanced(lines: List[str]) -> Tuple[str, str, str]:
    """Ultra-enhanced degree detection combining all strategies."""
    
    # Get degree using comprehensive detection
    degree = detect_degree_comprehensive(lines)
    
    # Get date using enhanced detection (from previous function)
    date_candidates = []
    for i, line in enumerate(lines[:400]):
        clean_line = html.unescape(line).strip()
        if not clean_line:
            continue
            
        for pattern in DATE_PATTERNS:
            matches = pattern.finditer(clean_line)
            for match in matches:
                date_text = match.group(0)
                
                # Score based on context
                context_score = 0
                
                # Check surrounding lines for context
                context_lines = []
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    context_lines.append(lines[j].lower())
                context = ' '.join(context_lines)
                
                # Higher score if date context words are nearby
                for word in DATE_CONTEXT_WORDS:
                    if word in context:
                        context_score += 1
                
                # Prefer dates near degree information
                if degree and any(word.lower() in context for word in degree.split()):
                    context_score += 2
                
                # Prefer reasonable graduation dates
                try:
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
                    if year_match:
                        year = int(year_match.group(0))
                        if 1990 <= year <= 2030:
                            context_score += 1
                        if 2000 <= year <= 2025:  # Most likely range
                            context_score += 1
                except:
                    pass
                
                date_candidates.append((context_score, date_text, i))
    
    # Select best date
    best_date = ''
    if date_candidates:
        date_candidates.sort(key=lambda x: x[0], reverse=True)
        best_date = _normalize_date(date_candidates[0][1])
    
    # Get classification (honors)
    classification = ''
    for line in lines[:400]:
        clean_line = html.unescape(line).strip()
        lower_line = clean_line.lower()
        
        honors_patterns = [
            r'\b(summa\s+cum\s+laude)\b',
            r'\b(magna\s+cum\s+laude)\b', 
            r'\b(cum\s+laude)\b',
            r'\b(with\s+(?:highest\s+)?honors?)\b',
            r'\b(with\s+distinction)\b',
            r'\b(dean\'?s\s+list)\b',
            r'\b(first\s+class\s+honours?)\b',
            r'\b(upper\s+second\s+class\s+honours?)\b',
            r'\b(lower\s+second\s+class\s+honours?)\b',
            r'\b(third\s+class\s+honours?)\b',
        ]
        
        for pattern in honors_patterns:
            match = re.search(pattern, lower_line)
            if match:
                classification = match.group(1).title().replace("'S", "'s")
                break
        
        if classification:
            break
    
    return degree, best_date, classification

# =========================
# CLI
# =========================

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

    # pipeline args accepted/ignored
    p.add_argument('--master-in', default=None)
    p.add_argument('--master-out', default=None)
    p.add_argument('--final-out', default=None)
    p.add_argument('--merge-mode', default=None)
    p.add_argument('--force', action='store_true')

    args = p.parse_args()

    in_dir = args.in_dir or args.folder
    if not in_dir:
        print('ERROR: --in-dir/--folder is required', file=sys.stderr)
        sys.exit(2)

    patterns = [g.strip() for g in args.glob.split(';') if g.strip()]
    files = _collect_files(in_dir, patterns, args.recursive)
    hint_map = _parse_hint_map(args.hint_map)

    rows: List[Dict[str, object]] = []
    modules_rows: List[Dict[str, object]] = []

    if not files:
        print('No input files found.')

    for fp in files:
        try:
            row = process_one(
                fp,
                institutions_limit=args.institutions_limit,
                default_country_hint=(args.country_hint or None),
                hint_map=hint_map,
                ocr=args.ocr,
                ocr_threshold=args.ocr_threshold,
                ocr_lang=args.ocr_lang,
                debug_dump_dir=args.debug_dump,
                verbose=args.verbose,
            )
            rows.append(row)

            app_key = row.get('application_key', '')
            try:
                mods = json.loads(row.get('module_records', '[]'))
            except Exception:
                mods = []
            for m in mods:
                modules_rows.append({
                    'application_key': app_key,
                    'subject': m.get('subject', ''),
                    'number': m.get('number', ''),
                    'title': m.get('title', ''),
                    'grade': m.get('grade', ''),
                    'grade_scale': m.get('grade_scale', ''),
                    'credits': m.get('credits', ''),
                    'credit_unit': m.get('credit_unit', ''),
                    'term': m.get('term', ''),
                    'pf': m.get('pf', ''),
                })
        except Exception as e:
            print(f"[warn] Failed to parse {fp}: {e}")

    # Write rows CSV
    cols = set()
    for r in rows:
        cols.update(r.keys())
    pref = [
        'application_key','student_name','university_name','major_most_recent','degree_most_recent',
        'degree_awarded_date','degree_classification','num_semesters_with_grades','cum_gpa_most_recent',
        'scale_hint','country_hint','module_records','date_of_birth',
    ]
    for i in range(1, 10):
        nm = f'institution_{i}_name'
        if any(nm in r for r in rows):
            pref.extend([nm, f'institution_{i}_dates', f'institution_{i}_degree'])
    ordered = pref + [c for c in sorted(cols) if c not in pref]

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write modules CSV
    m_fields = ['application_key','subject','number','title','grade','grade_scale','credits','credit_unit','term','pf']
    with open(args.modules_out, 'w', newline='', encoding='utf-8') as fm:
        mwriter = csv.DictWriter(fm, fieldnames=m_fields)
        mwriter.writeheader()
        for mr in modules_rows:
            mwriter.writerow(mr)

    print(f"Wrote {args.out} (rows={len(rows)}) and {args.modules_out} (rows={len(modules_rows)})")

if __name__ == '__main__':
    main()
