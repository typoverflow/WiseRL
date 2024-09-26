from wiserl.eval.cliff import eval_cliffwalking_rm
from wiserl.eval.offline import eval_offline
from wiserl.eval.reward_model import eval_reward_model, eval_world_model, eval_world_model_and_reward_model, eval_discriminator_model, eval_discriminator_model_and_reward_model


def eval_placeholder(*args, **kwargs):
    return {}
