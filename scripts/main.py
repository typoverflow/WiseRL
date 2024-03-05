import argparse
import functools
import os
import shutil

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

import wandb
import wiserl.algorithm
from wiserl.env import get_env
from wiserl.trainer.offline_trainer import OfflineTrainer

if __name__ == "__main__":
    args = parse_args(convert=False)
    name_prefix = f"{args['algorithm']}/{args['name']}/{args['env']}"
    logger = CompositeLogger(
        log_dir=f"./log/{name_prefix}",
        name="seed"+str(args["seed"]),
        logger_config={
            "TensorboardLogger": {},
            "WandbLogger": {**args["wandb"], "config": args, "settings": wandb.Settings(_disable_stats=True)}
        },
        backup_stdout=True,
        activate=not args["debug"]
    )
    logger.log_config(args, type="yaml")
    setup(args, logger)

    # process the environment
    env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    eval_env_fn = functools.partial(get_env, args["env"], args["env_kwargs"], args["env_wrapper"], args["env_wrapper_kwargs"])
    env = env_fn()

    # define the algorithm

    algorithm = vars(wiserl.algorithm)[args["algorithm"].pop("class")](
        env.observation_space,
        env.action_space,
        args["network"],
        args["optim"],
        args["schedulers"],
        args["processor"],
        args["checkpoint"],
        **args["algorithm"],
        device=args["device"]
    )

    # define the trainer
    trainer = OfflineTrainer(
        algorithm=algorithm,
        env_fn=env_fn,
        eval_env_fn=eval_env_fn,
        dataset_kwargs=args["dataset"],
        eval_kwargs=args["eval"],
        dataloader_kwargs=args["dataloader"],
        **args["trainer"],
        logger=logger,
        device=args["device"]
    )
    trainer.train()
