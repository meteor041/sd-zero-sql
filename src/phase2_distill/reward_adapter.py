from typing import Dict

from phase1_srt.trace_schema import reward_from_verifier
from sql_core.sql_normalizer import normalize_sql_output
from sql_core.sql_verifier import verify_sql


def compute_sql_reward(sample: Dict, student_response: str) -> Dict:
    normalized_sql = normalize_sql_output(student_response)
    verifier_result = verify_sql(sample['db_id'], normalized_sql, sample['gold_sql'])
    return {
        'normalized_sql': normalized_sql,
        'reward': reward_from_verifier(verifier_result),
        'verifier_result': verifier_result,
    }
