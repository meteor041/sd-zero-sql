P_R_CORRECT = 'Let me rephrase the above solution.'
P_R_INCORRECT = 'Wait, this response is wrong. Let me correct it.'

CANONICAL_TRACE_FIELDS = (
    'id',
    'db_id',
    'x',
    'gold_sql',
    'y_init',
    'y_init_correct',
    'p_r',
    'y_revised',
    'y_revised_correct',
    'keep',
    'verifier_init',
    'verifier_revised',
)


def select_p_r(reward: int) -> str:
    return P_R_CORRECT if int(reward) == 1 else P_R_INCORRECT
