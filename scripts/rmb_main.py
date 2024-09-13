import argparse
import functools
import os
import shutil

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

import wandb
import wiserl.algorithm
from wiserl.env import get_env
from wiserl.trainer.rmb_offline_trainer import RewardModelBasedOfflineTrainer

def post_init(args):
    # get the upper level args with string or number values, like {"env": "CartPole-v1"}
    upper_args = {"<" + k + ">": v for k, v in args.items() if isinstance(v, (str, int, float))}
    # replace args whose values are "<key>" with the value of the key
    def _replace(args):
        if type(args) is dict:
            final_args = {}
            for k, v in args.items():
                if isinstance(v, str) and v in upper_args:
                    final_args[k] = upper_args[v]
                else:
                    final_args[k] = _replace(v)
            return final_args
        elif type(args) is list:
            return [_replace(v) for v in args]
        elif type(args) is tuple:
            return tuple([_replace(v) for v in args])
        else:
            return args
    args.update(_replace(args))

if __name__ == "__main__":
    args = parse_args(convert=False, post_init=post_init)
    name_prefix = f"{args['algorithm']['class']}/{args['name']}/{args['env']}"
    logger = CompositeLogger(
        log_dir=f"./log/{name_prefix}",
        name="seed"+str(args["seed"]),
        logger_config={
            "TensorboardLogger": {},
            "WandbLogger": {**args["wandb"], "config": args, "settings": wandb.Settings(_disable_stats=True)},
            "CsvLogger": {"activate": args.get("csv", False)}
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
    trainer = RewardModelBasedOfflineTrainer(
        algorithm=algorithm,
        env_fn=env_fn,
        eval_env_fn=eval_env_fn,
        rm_dataset_kwargs=args["rm_dataset"],
        rm_dataloader_kwargs=args["rm_dataloader"],
        rm_eval_kwargs=args["rm_eval"],
        rl_dataset_kwargs=args["rl_dataset"],
        rl_dataloader_kwargs=args["rl_dataloader"],
        rl_eval_kwargs=args["rl_eval"],
        **args["trainer"],
        logger=logger,
        device=args["device"]
    )
    trainer.train()
