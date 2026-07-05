from typing import Any, Dict, Tuple

from .constants import CANONICAL_TRACE_FIELDS, select_p_r


def reward_from_verifier(verifier_result: Dict[str, Any] | None) -> int:
    if not verifier_result:
        return 0
    return int(verifier_result.get('reward', 0))


def _coalesce_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    return bool(value)


def build_trace_record(
    sample: Dict[str, Any],
    x: str,
    y_init: str,
    verifier_init: Dict[str, Any],
    y_revised: str,
    verifier_revised: Dict[str, Any],
    *,
    raw_y_init: str | None = None,
    raw_y_revised: str | None = None,
    revision_prompt: str | None = None,
) -> Dict[str, Any]:
    init_reward = reward_from_verifier(verifier_init)
    revised_reward = reward_from_verifier(verifier_revised)
    return {
        'id': sample.get('id'),
        'db_id': sample.get('db_id'),
        'question': sample.get('question'),
        'evidence': sample.get('evidence'),
        'schema': sample.get('schema'),
        'x': x,
        'base_prompt': x,
        'gold_sql': sample.get('gold_sql'),
        'raw_y_init': raw_y_init,
        'y_init': y_init,
        'y_init_correct': init_reward == 1,
        'p_r': select_p_r(init_reward),
        'verifier_init': verifier_init,
        'revision_prompt': revision_prompt,
        'raw_y_revised': raw_y_revised,
        'y_revised': y_revised,
        'y_revised_correct': revised_reward == 1,
        'verifier_revised': verifier_revised,
        'keep': revised_reward == 1,
    }


def normalize_trace_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    x = normalized.get('x') or normalized.get('base_prompt')
    verifier_init = normalized.get('verifier_init') or {}
    verifier_revised = normalized.get('verifier_revised') or {}
    init_reward = reward_from_verifier(verifier_init)
    revised_reward = reward_from_verifier(verifier_revised)

    normalized['x'] = x
    normalized['base_prompt'] = normalized.get('base_prompt') or x
    normalized['y_init_correct'] = _coalesce_bool(normalized.get('y_init_correct'), init_reward == 1)
    normalized['p_r'] = normalized.get('p_r') or select_p_r(init_reward)
    normalized['y_revised_correct'] = _coalesce_bool(normalized.get('y_revised_correct'), revised_reward == 1)
    normalized['keep'] = _coalesce_bool(normalized.get('keep'), normalized['y_revised_correct'])
    normalized['verifier_init'] = verifier_init
    normalized['verifier_revised'] = verifier_revised
    return normalized


def validate_trace_record(record: Dict[str, Any]) -> None:
    missing = [field for field in CANONICAL_TRACE_FIELDS if field not in record]
    if missing:
        raise ValueError(f'Missing canonical trace fields: {missing}')


def dedupe_trace_key(record: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    normalized = normalize_trace_record(record)
    return (
        normalized.get('x'),
        normalized.get('y_init'),
        normalized.get('p_r'),
        normalized.get('y_revised'),
    )
