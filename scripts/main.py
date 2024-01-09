import argparse
import functools
import os
import shutil

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

import src.algorithm
from src.env import get_env
from src.trainer.offline_trainer import OfflineTrainer

if __name__ == "__main__":
    args = parse_args(convert=False)
    name_prefix = f"{args['algorithm']}/{args['env']}/seed{args['seed']}"
    logger = CompositeLogger(
        log_dir=f"./log/{name_prefix}",
        name=args["name"],
        logger_config={
            "TensorboardLogger": {},
        },
        backup_stdout=True,
        activate=not args["debug"]
    )
    setup(args, logger)

    # process the environment
    env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    eval_env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    env = env_fn()

    # define the algorithm
    algorithm = vars(src.algorithm)[args["algorithm"]](
        env.observation_space,
        env.action_space,
        args["network"],
        args["optim"],
        args["schedulers"],
        args["processor"],
        args["checkpoint"],
        **args["algorithm_kwargs"],
        device=args["device"]
    )

    # define the trainer
    trainer = OfflineTrainer(
        algorithm=algorithm,
        env_fn=env_fn,
        eval_env_fn=eval_env_fn,
        dataset_kwargs=args["dataset"],
        dataloader_kwargs=args["dataloader"],
        **args["trainer_kwargs"],
        logger=logger
        # total_steps=args["total_steps"],
        # log_freq=args["log_freq"],
        # env_freq=args["env_freq"],
        # eval_freq=args["eval_freq"],
        # profile_freq=args["profile_freq"],
        # checkpoint_freq=args["checkpoint_freq"],
    )
    trainer.train()
