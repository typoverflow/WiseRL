from wiserl.eval.cliff import eval_cliffwalking_rm
from wiserl.eval.offline import eval_offline, eval_offline_with_history_input
from wiserl.eval.reward_model import eval_reward_model


def eval_placeholder(*args, **kwargs):
    return {}
