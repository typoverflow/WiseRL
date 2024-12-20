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
from wiserl.utils.utils import use_placeholder

# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/halfcheetah-gravity-80-150.yaml --name halfcheetah-gravity-80-150-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/halfcheetah-gravity-80-80.yaml --name halfcheetah-gravity-80-80-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/halfcheetah-gravity-150-150.yaml --name halfcheetah-gravity-150-150-bt-iql-rpl

# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-50-50.yaml --name Walker2d-v3-gravity-50-50-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-100-100.yaml --name Walker2d-v3-gravity-100-100-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-150-150.yaml --name Walker2d-v3-gravity-150-150-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-50-100.yaml --name Walker2d-v3-gravity-50-100-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-50-150.yaml --name Walker2d-v3-gravity-50-150-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-150-50.yaml --name Walker2d-v3-gravity-150-50-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-150-100.yaml --name Walker2d-v3-gravity-150-100-bt-iql-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_iql/rpl/gravity-100-100-b1-e9.yaml --name Walker2d-v3-gravity-100-100-bt-iql-rpl-b1-e9


# python scripts/rmb_main.py --config scripts/configs/bt_awac/rpl/gravity-50-50.yaml --name Walker2d-v3-gravity-50-50-bt-awac-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_awac/rpl/gravity-100-100.yaml --name Walker2d-v3-gravity-100-100-bt-awac-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_awac/rpl/gravity-150-150.yaml --name Walker2d-v3-gravity-150-150-bt-awac-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_awac/rpl/gravity-50-100.yaml --name Walker2d-v3-gravity-50-100-bt-awac-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_awac/rpl/gravity-50-150.yaml --name Walker2d-v3-gravity-50-150-bt-awac-rpl
# python scripts/rmb_main.py --config scripts/configs/bt_awac/rpl/gravity-150-50.yaml --name Walker2d-v3-gravity-150-50-bt-awac-rpl



if __name__ == "__main__":
    args = parse_args(convert=False, post_init=use_placeholder)
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
    if "eval_env" in args:
        eval_env_fn = functools.partial(get_env, args["eval_env"], args["eval_env_kwargs"], args["eval_env_wrapper"], args["eval_env_wrapper_kwargs"])
    else:
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
