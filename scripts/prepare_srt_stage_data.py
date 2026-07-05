#!/usr/bin/env python3
from pathlib import Path
import runpy

TARGET = Path('/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/scripts/srt/build_two_stage_data.py')
runpy.run_path(str(TARGET), run_name='__main__')
