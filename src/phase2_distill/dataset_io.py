import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset

DEFAULT_TRAIN_FILE = '/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_train_4k.jsonl'
DEFAULT_VALID_FILE = '/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_valid_4k.jsonl'


def load_prompt_dataset(path: str, max_samples: Optional[int] = None):
    dataset = load_dataset('json', data_files=path, split='train')
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    return dataset


def dataset_to_samples(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    dataset = load_prompt_dataset(path, max_samples)
    return [dataset[i] for i in range(len(dataset))]


def iter_sample_batches(samples: Sequence[Dict], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield list(samples[start:start + batch_size])


def write_distill_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
