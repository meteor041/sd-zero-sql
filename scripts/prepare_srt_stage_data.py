#!/usr/bin/env python3
from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / 'srt' / 'build_two_stage_data.py'
runpy.run_path(str(TARGET), run_name='__main__')
