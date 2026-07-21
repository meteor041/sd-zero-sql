#!/usr/bin/env python3
from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / 'sft' / 'filter_sft_by_length.py'
runpy.run_path(str(TARGET), run_name='__main__')
