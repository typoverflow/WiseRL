import itertools
from typing import Any, Dict, Optional, Type, Union, Tuple

import torch
import torch.nn as nn

import wiserl.module
from wiserl.algorithm.base import Algorithm
from wiserl.module.actor import DeterministicActor, GaussianActor
from wiserl.utils.misc import make_target, sync_target


class OracleSAC(Algorithm):
    def __init__(
        self,
        *args,
        alpha: Union[float, Tuple[float, float]] = 0.2,
        auto_alpha: bool = False,
        discount: float = 0.99,
        tau: float = 0.005,
        target_freq: int = 1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._is_auto_alpha = auto_alpha
        if self._is_auto_alpha:
            target_entropy = -float(self.action_space.shape[-1])
            alpha_lr = alpha
            self._log_alpha = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=self.device), requires_grad=True)
            self._target_entropy = target_entropy
            self.alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.tensor([alpha], dtype=torch.float32, device=self.device, requires_grad=False)


        self.target_freq = target_freq
        self.discount = discount
        self.tau = tau

    def setup_network(self, network_kwargs):
        network = {}
        network["actor"] = vars(wiserl.module)[network_kwargs["actor"].pop("class")](
            input_dim=self.observation_space.shape[0],
            output_dim=self.action_space.shape[0],
            **network_kwargs["actor"]
        )
        network["critic"] = vars(wiserl.module)[network_kwargs["critic"].pop("class")](
            input_dim=self.observation_space.shape[0]+self.action_space.shape[0],
            output_dim=1,
            **network_kwargs["critic"]
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
        self.target_network = nn.ModuleDict({
            "critic": make_target(self.network.critic)
        })

    def setup_optimizers(self, optim_kwargs):
        self.optim = {}
        default_kwargs = optim_kwargs.get("default", {})

        actor_kwargs = default_kwargs.copy()
        actor_kwargs.update(optim_kwargs.get("actor", {}))
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = vars(torch.optim)[actor_kwargs.pop("class")](actor_params, **actor_kwargs)

        critic_kwargs = default_kwargs.copy()
        critic_kwargs.update(optim_kwargs.get("critic", {}))
        self.optim["critic"] = vars(torch.optim)[critic_kwargs.pop("class")](self.network.critic.parameters(), **critic_kwargs)

    def select_action(self, batch, deterministic: bool=True):
        obs = self.network.encoder(batch["obs"])
        action, *_ = self.network.actor.sample(obs, deterministic=deterministic)
        return action.squeeze().cpu().numpy()

    def actor_loss(self, encoded_obs, action, reduce=True):
        new_actions, new_logprobs, _ = self.network.actor.sample(encoded_obs)
        q_values = self.network.critic(encoded_obs, new_actions)
        if len(q_values.shape) == 2:
            q_values = q_values.unsqueeze(0)
        q_values_min = torch.min(q_values, dim=0)[0]
        q_values_std = torch.std(q_values, dim=0).mean().item()
        q_values_mean = q_values.mean().item()
        actor_loss = self._alpha * new_logprobs - q_values_min

        return actor_loss.mean() if reduce else actor_loss, {
                                                            "loss/actor_loss": actor_loss.mean() if reduce else actor_loss,\
                                                            "misc/q_values_std": q_values_std,\
                                                              "misc/q_values_min": q_values_min.mean().item(),\
                                                                "misc/q_values_mean": q_values_mean}

    def q_loss(self, encoded_obs, action, next_encoded_obs, reward, terminal, reduce=True):
        with torch.no_grad():
            self.target_network.eval()
            next_actions, next_logprobs, _ = self.network.actor.sample(next_encoded_obs)
            target_q = self.target_network.critic(next_encoded_obs, next_actions).min(0)[0]- self._alpha * next_logprobs
            target_q = reward + self.discount * (1-terminal) * target_q
        q_pred = self.network.critic(encoded_obs, action)
        q_loss = (q_pred - target_q.unsqueeze(0)).pow(2).sum(0)
        return q_loss.mean() if reduce else q_loss, {"loss/q_loss": q_loss.mean() if reduce else q_loss, "misc/q_pred":q_pred.mean() if reduce else q_pred}

    def alpha_loss(self, encoded_obs, reduce=True):
        with torch.no_grad():
            _, new_logprobs, _ = self.network.actor.sample(encoded_obs)
        alpha_loss = -(self._log_alpha * (new_logprobs + self._target_entropy)).mean()
        return alpha_loss.mean() if reduce else alpha_loss
    
    def train_step(self, batches, step:int, total_steps: int): 
        if isinstance(batches, list):
            batch, *_ = batches
        elif isinstance(batches, dict):
            batch = batches
        else:
            assert 0,f'Undefined Type:{type(batches)}'
        metrics = {}
        if "obs_1" in batch:
            obs = torch.cat([batch["obs_1"], batch["obs_2"]], dim=0)  # (B, S+1)
            action = torch.cat([batch["action_1"], batch["action_2"]], dim=0)  # (B, S+1)
            reward = torch.cat([batch["reward_1"], batch["reward_2"]], dim=0)
            terminal = torch.cat([batch["terminal_1"], batch["terminal_2"]], dim=0)

            encoded_obs = self.network.encoder(obs)

            q_loss, q_pred = self.q_loss(
                encoded_obs[:, :-1].detach(),
                action[:, :-1],
                encoded_obs[:, 1:].detach(),
                reward[:, :-1],
                terminal[:, :-1]
            )
        else:
            obs = batch["obs"]
            action = batch["action"]
            reward = batch["reward"]
            terminal = batch["terminal"].float()
            next_obs = batch["next_obs"]

            encoded_obs = self.network.encoder(obs)
            next_encoded_obs = self.network.encoder(next_obs)
            
            q_loss, q_metrics = self.q_loss(encoded_obs, action, next_encoded_obs, reward, terminal)
        
        metrics.update(q_metrics)
        self.optim["critic"].zero_grad()
        q_loss.backward()
        self.optim["critic"].step()

        # compute the loss for actor
        actor_loss, actor_metrics= self.actor_loss(encoded_obs, action)
        self.optim["actor"].zero_grad()
        actor_loss.backward()
        self.optim["actor"].step()
        metrics.update(actor_metrics)

        if self._is_auto_alpha:
            alpha_loss = self.alpha_loss(encoded_obs)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.exp().detach()
        else:
            alpha_loss = 0
        metrics["misc/alpha"] = self._alpha.item()

        if step % self.target_freq == 0:
            sync_target(self.network.critic, self.target_network.critic, tau=self.tau)

        for _, scheduler in self.schedulers.items():
            scheduler.step()

        
        return metrics
