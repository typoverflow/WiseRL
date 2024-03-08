import itertools
from typing import Any, Dict

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.functional import biased_bce_with_logits
from wiserl.utils.misc import make_target


class CPL_KL(Algorithm):
    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        bias: float = 1.0,
        bc_data: str = "total",
        bc_steps: int = 10000,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.bias = bias
        self.bc_data = bc_data
        self.bc_steps = bc_steps

    def setup_network(self, network_kwargs):
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]
        )
        if "encoder" in network_kwargs:
            network["encoder"] = vars(wiserl.module)[network_kwargs["encoder"].pop("class")](
                input_dim=self.observation_space.shape[0],
                output_dim=1,
                **network_kwargs["encoder"]
            )
        else:
            network["encoder"] = nn.Identity()
        self.network = nn.ModuleDict(network)
        self.ref_network = make_target(self.network)

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_kwargs.pop("class")](actor_params, **actor_kwargs)

    def setup_schedulers(self, scheduler_kwargs, is_bc=True):
        if is_bc:
            return
        else:
            return super().setup_schedulers(scheduler_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        obs = self.network.encoder(batch["obs"])
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def train_step(self, batches, step: int, total_steps: int):
        batch, *_ = batches
        obs = torch.concat([batch["obs_1"], batch["obs_2"]], dim=0)
        encoded_obs = self.network.encoder(obs)
        action = torch.concat([batch["action_1"], batch["action_2"]], dim=0)
        label = batch["label"].float()

        if step == self.bc_steps + 1:
            self.ref_network.load_state_dict(self.network.state_dict())
            self.setup_optimizers(self.optim_kwargs)
            self.setup_schedulers(self.schedulers_kwargs, is_bc=False)

        if isinstance(self.network.actor, DeterministicActor):
            logprob = - torch.square(action - self.network.actor.sample(encoded_obs)[0]).sum(dim=-1, keepdim=True)
        elif isinstance(self.network.actor, GaussianActor):
            logprob = self.network.actor.evaluate(encoded_obs, action)[0]

        if step <= self.bc_steps:
            if self.bc_data == "total":
                mask = torch.concat([torch.ones_like(label), torch.ones_like(label)], dim=0)
            elif self.bc_data == "win":
                mask = torch.concat([1-label, label], dim=0)
            elif self.bc_data == "lose":
                mask = torch.concat([label, 1-label], dim=0)
            bc_loss = -(logprob.mean(dim=1) * mask).sum() / mask.sum()

            self.optim["actor"].zero_grad()
            bc_loss.backward()
            self.optim["actor"].step()

            metrics = {
                "loss/bc_loss": bc_loss.item(),
            }
        else:
            self.ref_network.eval()
            with torch.no_grad():
                ref_encoded_obs = self.ref_network.encoder(obs)
                if isinstance(self.ref_network.actor, DeterministicActor):
                    ref_logprob = - torch.square(action - self.ref_network.actor.sample(ref_encoded_obs)[0]).sum(dim=-1, keepdim=True)
                else:
                    ref_logprob = self.ref_network.actor.evaluate(ref_encoded_obs, action)[0]
            adv = self.alpha * (logprob - ref_logprob)
            segment_adv = adv.sum(dim=1)
            adv1, adv2 = torch.chunk(segment_adv, 2, dim=0)
            cpl_loss = biased_bce_with_logits(adv1, adv2, label, bias=self.bias).mean()
            accuracy = ((adv1<adv2) == torch.round(label)).float().mean()

            self.optim["actor"].zero_grad()
            cpl_loss.backward()
            self.optim["actor"].step()
            metrics = {
                "loss/cpl_loss": cpl_loss.item(),
                "misc/accuracy": accuracy.item()
            }
        return metrics
