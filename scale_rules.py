# scale_rules.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import re, math
from typing import Optional, Tuple, Dict, Union

INSTITUTION_HINTS: Dict[str, str] = {}
COUNTRY_HINTS: Dict[str, str] = {}

ALLOWED_FRACTION_DENOMS = {4.0, 4.3, 5.0, 6.0, 10.0, 20.0, 100.0}  # reject date-like denominators

def _cap4(x: float) -> Optional[float]:
    try:
        return max(0.0, min(float(x), 4.0))
    except Exception:
        return None

def conv_linear(value: float, from_max: float) -> Optional[float]:
    try:
        v = float(value); m = float(from_max)
    except Exception:
        return None
    if m <= 0: return None
    return _cap4(v / m * 4.0)

def conv_polish_55(value: float) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return _cap4(((v - 2.0) / 3.5) * 4.0)

def conv_german_reversed_5(value: float) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if v <= 1.0: return 4.0
    if v >= 5.0: return 0.0
    return _cap4(5.0 - v)

def conv_four_point_three(value: float) -> Optional[float]: return conv_linear(value, 4.3)
def conv_twenty_point(value: float) -> Optional[float]:    return conv_linear(value, 20.0)

def _detect_by_value(val: float) -> Optional[str]:
    if val is None: return None
    try: v = float(val)
    except Exception: return None
    if 0.0 <= v <= 4.0:   return 'us_4'
    if 4.0 < v <= 5.0:    return 'five_point'
    if 5.0 < v <= 5.5:    return 'polish_55'
    if 5.5 < v <= 6.0:    return 'six_point'
    if 6.0 < v <= 10.0:   return 'ten_point'
    if 10.0 < v <= 20.0:  return 'twenty_point'
    if 20.0 < v <= 100.0: return 'percent_100'
    return None

_NUM = r'(\d+(?:\.\d+)?)'
FRACTION = re.compile(rf'^\s{_NUM}\s*/\s*{_NUM}\s*$'.replace(' ', ''))
EMBEDDED_FRAC = re.compile(rf'{_NUM}\s*/\s*{_NUM}')
EMBEDDED_NUM = re.compile(rf'{_NUM}')

def _to_float_safe(s: str) -> Optional[float]:
    if s is None: return None
    t = str(s).strip().replace(',', '.')
    if not t: return None
    try: return float(t)
    except Exception: return None

def _norm_lower_or_none(x: Union[str, float, int, None]) -> Optional[str]:
    if isinstance(x, str):
        t = x.strip().lower(); return t or None
    if isinstance(x, float) and math.isnan(x): return None
    return None

def _norm_text_lower(x: Union[str, float, int, None]) -> str:
    if isinstance(x, str): return x.strip().lower()
    if isinstance(x, float):
        if math.isnan(x): return ''
        return str(x).strip().lower()
    if x is None: return ''
    return str(x).strip().lower()

def apply_conversion(
    raw_value: object,
    *,
    university_name: str = '',
    scale_hint: Optional[Union[str, float, int]] = None,
    country_hint: Optional[Union[str, float, int]] = None
) -> Tuple[Optional[float], Optional[str]]:
    s = str(raw_value or '').strip()
    if not s: return None, None

    # exact fraction x/y
    m = FRACTION.match(s)
    if m:
        num = _to_float_safe(m.group(1)); den = _to_float_safe(m.group(2))
        if den and den > 0:
            if den not in ALLOWED_FRACTION_DENOMS:  # reject 17, 21, 28, etc.
                return None, None
            return conv_linear(num, den), f'fraction_{den:g}'

    # percentage
    if s.endswith('%'):
        v = _to_float_safe(s[:-1])
        if v is not None: return conv_linear(v, 100.0), 'percent_100'

    # plain number or embedded fraction
    val = _to_float_safe(s)
    if val is None:
        m2 = EMBEDDED_FRAC.search(s)
        if m2:
            nums = [_to_float_safe(x) for x in EMBEDDED_NUM.findall(m2.group(0))]
            if len(nums) >= 2 and nums[1] and nums[1] > 0:
                den = float(nums[1])
                if den not in ALLOWED_FRACTION_DENOMS:
                    return None, None
                return conv_linear(nums[0], den), 'embedded_fraction'
        return None, None

    lab = _norm_lower_or_none(scale_hint)
    ch  = _norm_lower_or_none(country_hint)
    uname = _norm_text_lower(university_name)
    if not lab and uname:
        for k, v in INSTITUTION_HINTS.items():
            if k in uname:
                lab = _norm_lower_or_none(v); break
    if not lab and ch:
        mapped = COUNTRY_HINTS.get(ch)
        lab = _norm_lower_or_none(mapped)

    rule = lab or _detect_by_value(val)
    if not rule: return None, None

    if rule == 'us_4':              return _cap4(val), 'us_4'
    if rule == 'ten_point':         return conv_linear(val, 10.0), 'ten_point'
    if rule == 'percent_100':       return conv_linear(val, 100.0), 'percent_100'
    if rule == 'five_point':        return conv_linear(val, 5.0), 'five_point'
    if rule == 'polish_55':         return conv_polish_55(val), 'polish_55'
    if rule == 'six_point':         return conv_linear(val, 6.0), 'six_point'
    if rule == 'german_reversed_5': return conv_german_reversed_5(val), 'german_reversed_5'
    if rule == 'four_point_three':  return conv_four_point_three(val), 'four_point_three'
    if rule == 'twenty_point':      return conv_twenty_point(val), 'twenty_point'
    return None, None
