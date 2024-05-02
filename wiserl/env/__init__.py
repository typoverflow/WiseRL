# Register environment classes here
# Register the environments
from typing import Dict, Optional

import gym
import UtilsRL.env.wrapper
from gym.envs import register

from .base import EmptyEnv
from .cliffwalking_env import CliffWalkingEnv

# try:
#     from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

#     for env_name in ALL_V2_ENVIRONMENTS.keys():
#         ID = f"mw_{env_name}"
#         register(id=ID, entry_point="wiserl.env.metaworld_env:MetaWorldSawyerEnv", kwargs={"env_name": env_name})
#         id_parts = ID.split("-")
#         id_parts[-1] = "image-" + id_parts[-1]
#         ID = "-".join(id_parts)
#         register(id=ID, entry_point="wiserl.env.metaworld_env:get_mw_image_env", kwargs={"env_name": env_name})
# except ImportError:
#     print("Warning: Could not import MetaWorld Environments.")

try:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        ID = f"{env_name}"
        register(id=ID, entry_point="wiserl.env.metaworld_env:SawyerEnv", kwargs={"env_name": env_name})
except ImportError:
    print("Warning: Could not import MetaWorld Environments")

try:
    from wiserl.env.robomimic_env import DATASET_PATH, RobomimicEnv
    for env_name in DATASET_PATH.keys():
        ID=env_name
        register(id=ID, entry_point="wiserl.env.robomimic_env:RobomimicEnv", kwargs={"env_name": env_name})
except ImportError:
    print("Warning: Could not import RobomimicEnv")


def get_env(
    env: str,
    env_kwargs: Optional[Dict]=None,
    wrapper_class: Optional[str]=None,
    wrapper_kwargs: Optional[Dict]=None,
):
    try:
        env_kwargs = env_kwargs or {}
        env = vars()[env](**env_kwargs)
    except KeyError as e:
        env = gym.make(env, **env_kwargs)
    if wrapper_class is not None:
        wrapper_kwargs = wrapper_kwargs or {}
        env = vars(UtilsRL.env.wrapper)[wrapper_class](env, **wrapper_kwargs)
    return env
