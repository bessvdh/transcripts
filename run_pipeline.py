# run_pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]):
    print(">>>", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Optional ingest step (run bulk_ingest.py first if provided)
    p.add_argument('--in-dir', dest='in_dir', default=None)
    p.add_argument('--folder', dest='folder', default=None)     # alias
    p.add_argument('--glob', default="*.pdf;*.txt")
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--ocr', action='store_true')
    p.add_argument('--ocr-threshold', default="200")
    p.add_argument('--ocr-lang', default="eng")
    p.add_argument('--hint-map', default="")
    p.add_argument('--debug-dump', default=None)
    p.add_argument('--verbose', action='store_true')

    # Merge + postprocess steps (same as before)
    p.add_argument('--master-in', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--master-out', required=True)
    p.add_argument('--final-out', required=True)
    p.add_argument('--modules', default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--merge-log', default=None)
    p.add_argument('--merge-mode', choices=['strict', 'loose'], default='strict')

    args = p.parse_args()

    # Step 0 (optional): ingest new rows first if an input folder is given
    in_dir = args.in_dir or args.folder
    if in_dir:
        cmd0 = [sys.executable, 'bulk_ingest.py',
                '--in-dir', in_dir,
                '--glob', args.glob]
        if args.recursive:
            cmd0.append('--recursive')
        if args.ocr:
            cmd0.append('--ocr')
        cmd0 += ['--ocr-threshold', str(args.ocr_threshold),
                 '--ocr-lang', args.ocr_lang,
                 '--hint-map', args.hint_map,
                 '--out', args.new,
                 '--modules-out', args.modules or '_tmp_modules.csv']
        if args.debug_dump:
            cmd0 += ['--debug-dump', args.debug_dump]
        if args.verbose:
            cmd0.append('--verbose')
        run(cmd0)

    # Step 1: merge new rows into master
    cmd1 = [sys.executable, 'append_helper.py',
            '--master', args.master_in,
            '--new', args.new,
            '--out', args.master_out]
    if args.modules:
        cmd1 += ['--modules', args.modules]
    run(cmd1)

    # Step 2: postprocess (convert/synthesize GPA, metrics, highest ECON)
    cmd2 = [sys.executable, 'postprocess_master.py',
            '--in', args.master_out,
            '--out', args.final_out,
            '--merge-mode', args.merge_mode]
    if args.force:
        cmd2.append('--force')
    if args.merge_log:
        cmd2 += ['--merge-log', args.merge_log]
    if args.verbose:
        cmd2.append('--verbose')

    run(cmd2)

    print("Pipeline completed successfully.")
