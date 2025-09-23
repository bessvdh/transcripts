# convert_master.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pandas as pd

# Safer import: helpful error if scale_rules.py is missing/invalid
try:
    from scale_rules import apply_conversion, INSTITUTION_HINTS
except Exception as e:
    raise SystemExit(
        "ERROR: Missing or invalid scale_rules.py with apply_conversion / INSTITUTION_HINTS.\n"
        f"Detail: {e}\n"
        "Tip: place scale_rules.py next to this script, and ensure it defines:\n"
        "  - apply_conversion(raw_gpa: str, *, university_name: str, scale_hint: str|None, country_hint: str|None) -> tuple[float|None, str|None]\n"
        "  - INSTITUTION_HINTS: dict[str, str]"
    )

parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='inp', required=True)
parser.add_argument('--out', dest='out', required=True)
parser.add_argument('--force', action='store_true')
parser.add_argument('--hint', action='append', default=[])
args = parser.parse_args()

for h in args.hint:
    if '=' in h:
        k, v = h.split('=', 1)
        INSTITUTION_HINTS[k.strip().lower()] = v.strip()

df = pd.read_csv(args.inp)

for col in ['converted gpa', 'cum_gpa_most_recent', 'university_name']:
    if col not in df.columns:
        df[col] = ''
if 'scale_hint' not in df.columns:
    df['scale_hint'] = ''
if 'country_hint' not in df.columns:
    df['country_hint'] = ''

updates = 0
for i, row in df.iterrows():
    has_conv = str(row.get('converted gpa', '')).strip()
    if has_conv and not args.force:
        continue

    raw = row.get('cum_gpa_most_recent', '')
    uni = row.get('university_name', '')
    sh = row.get('scale_hint', '') or None
    ch = row.get('country_hint', '') or None

    conv, rule = apply_conversion(raw, university_name=uni, scale_hint=sh, country_hint=ch)
    if conv is not None:
        df.at[i, 'converted gpa'] = round(conv, 3)
        if rule:
            df.at[i, 'conversion_rule'] = rule
        updates += 1

if 'conversion_rule' not in df.columns:
    df['conversion_rule'] = ''

df.to_csv(args.out, index=False)
print(f"Wrote {args.out} (updated {updates} rows)")
