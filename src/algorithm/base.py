import copy
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

import gym
import torch
import torch.nn as nn

import src
from src.processor import Identity, Processor
from src.utils import utils


class Algorithm(ABC):
    _save_keys: Set[str]
    _module_keys: Set[str]
    _compiled: bool

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network_kwargs: Optional[Dict] = None,
        optim_kwargs: Optional[Dict] = None,
        schedulers_kwargs: Optional[Dict[str, Dict]] = None,
        processor_kwargs: Optional[Dict] = None,
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        # Initialize the _save_keys attribute using the superclass.
        # These are used for automatically identifying keys for saving/loading.
        super().__setattr__("_save_keys", set())
        super().__setattr__("_module_keys", set())
        super().__setattr__("_compiled", False)

        # Save relevant values
        self.observation_space = observation_space
        self.action_space = action_space

        self.device = device

        # setup the networks
        self.setup_network(network_kwargs)
        self.optim = {}
        self.schedulers = {}
        self.processor = {}
        self.setup_optimizers(optim_kwargs)
        self.setup_schedulers(schedulers_kwargs)
        self.setup_processor(processor_kwargs)

        self._training = False
        if checkpoint is not None:
            self.load(checkpoint, strict=False)

    @property
    def training(self) -> bool:
        return self._training

    @property
    def save_keys(self) -> List[str]:
        return self._save_keys

    @property
    def module_keys(self) -> List[str]:
        return self._module_keys

    @property
    def compiled(self) -> bool:
        return self._compiled

    def __setattr__(self, name: str, value: Any) -> None:
        # Check to see if the value is a module etc.
        if (hasattr(self, "_save_keys") and name in self._save_keys) or (
            hasattr(self, "_module_keys") and name in self._module_keys
        ):
            pass
        elif isinstance(value, torch.nn.Parameter):
            self._save_keys.add(name)
        elif isinstance(value, torch.nn.Module):
            self._module_keys.add(name)
            if sum(p.numel() for p in value.parameters()) > 0:
                self._save_keys.add(name)  # store if we have a module with more than zero parameters.
        return super().__setattr__(name, value)

    @property
    def num_params(self):
        _num_params = 0
        for k in self.save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "parameters"):
                _num_params += sum(p.numel() for p in attr.parameters() if p.requires_grad)
            else:
                assert isinstance(attr, torch.nn.Parameter), "Can only save Modules or Parameters."
                if attr.requires_grad:
                    _num_params += attr.numel()
        return _num_params

    @property
    def nbytes(self):
        # Returns the size of all the parameters in bytes
        _bytes = 0
        for k in self.save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "parameters"):
                for p in attr.parameters():
                    _bytes += p.nelement() * p.element_size()
            if hasattr(attr, "buffers"):
                for b in attr.buffers():
                    _bytes += b.nelement() * b.element_size()
        return _bytes

    def to(self, device):
        for k in self.save_keys:
            if k == "processor" and not self.processor.supports_gpu:
                continue
            else:
                setattr(self, k, getattr(self, k).to(device))
        return self

    def compile(self, **kwargs):
        for k in self.save_keys:
            attr = getattr(self, k)
            if isinstance(attr, torch.nn.Module):
                if type(attr).forward == torch.nn.Module.forward:
                    # In this case, the forward method hasn't been overriden.
                    # Thus we assume there is a compile argument.
                    assert hasattr(attr, "compile"), (
                        "save key " + k + " is nn.Module without forward() but didn't define `compile`."
                    )
                    attr.compile(**kwargs)
                else:
                    setattr(self, k, torch.compile(attr, **kwargs))
        # indicate that we have compiled the models.
        self._compiled = True

    def train(self) -> None:
        for k in self._module_keys:
            getattr(self, k).train()
        self._training = True

    def eval(self) -> None:
        for k in self._module_keys:
            getattr(self, k).eval()
        self._training = False

    def setup_processor(self, processor_kwargs) -> None:
        pass
        # if processor_class is None:
        #     processor = Identity(self.observation_space, self.action_space)
        # else:
        #     processor = processor_class(self.observation_space, self.action_space, **processor_kwargs)

        # if processor.supports_gpu:  # move it to device if it supports GPU computation.
        #     self.processor = processor.to(self.device)
        # else:
        #     self.processor = processor

    def setup_network(self, network_kwargs) -> None:
        network = {}
        self.network = nn.ModuleDict(network).to(self.device)

    def setup_optimizers(self, optim_kwargs) -> None:
        pass

    def setup_schedulers(self, scheduler_kwargs):
        pass

    def save(self, path: str, extension: str, metadata: Optional[Dict]=None) -> None:
        """
        Saves a checkpoint of the model and the optimizers
        """
        save_dict = {}
        if len(self.optim) > 0:
            save_dict["optim"] = {k: v.state_dict() for k, v in self.optim.items()}
        if len(self.schedulers) > 0:
            save_dict["schedulers"] = {k: v.state_dict() for k, v in self.schedulers.items()}
        for k in self._save_keys:
            attr = getattr(self, k)
            if hasattr(attr, "state_dict"):
                save_dict[k] = attr.state_dict()
            else:
                assert isinstance(attr, torch.nn.Parameter), "Can only save Modules or Parameters."
                save_dict[k] = attr

        # Add the metadata
        save_dict["metadata"] = {} if metadata is None else metadata
        save_path = os.path.join(path, extension)
        if not save_path.endswith(".pt"):
            save_path += ".pt"
        torch.save(save_dict, save_path)

    def load(self, checkpoint: str, strict: bool=True) -> Dict:
        """
        Loads the model and its associated checkpoints.
        If we haven't created the optimizers and schedulers, do not load those.
        """
        print("[research] loading checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        remaining_checkpoint_keys = set(checkpoint.keys())

        # First load everything except for the optim
        for k in self.save_keys:  # Loop through keys in the Algorithm.
            if k not in checkpoint:
                if strict:
                    raise ValueError("Checkpoint did not have key " + str(k))
                else:
                    print("[research] Warning: Checkpoint did not have key", k)
                    continue

            if isinstance(getattr(self, k), torch.nn.Parameter):
                # directly set the data, this is for nn.Parameters
                getattr(self, k).data = checkpoint[k].data
            else:
                # Otherwise, load via state dict
                getattr(self, k).load_state_dict(checkpoint[k], strict=strict)
            remaining_checkpoint_keys.remove(k)

        # Now load the optimizer and its associated keys
        for k in self.optim.keys():
            if strict and k not in checkpoint["optim"]:
                raise ValueError("Strict mode was enabled, but couldn't find optimizer key")
            elif k not in checkpoint["optim"]:
                print("[research] Warning: Checkpoint did not have optimizer key", k)
                continue
            self.optim[k].load_state_dict(checkpoint["optim"][k])
        if "optim" in checkpoint:
            remaining_checkpoint_keys.remove("optim")

        # Now load the schedulers
        for k in self.schedulers.keys():
            if strict and k not in checkpoint["schedulers"]:
                raise ValueError("Strict mode was enabled, but couldn't find scheduler key")
            elif k not in checkpoint["schedulers"]:
                print("[research] Warning: Checkpoint did not have scheduler key", k)
                continue
            self.schedulers[k].load_state_dict(checkpoint["schedulers"][k])
        if "schedulers" in checkpoint:
            remaining_checkpoint_keys.remove("schedulers")

        remaining_checkpoint_keys.remove("metadata")  # Do not count metadata key, which is always addded.
        if strict and len(remaining_checkpoint_keys) > 0:
            raise ValueError("Algorithm did not have keys ", +str(remaining_checkpoint_keys))
        elif len(remaining_checkpoint_keys) > 0:
            print("[research] Warning: Checkpoint keys", remaining_checkpoint_keys, "were not loaded.")

        return checkpoint["metadata"]

    def format_batch(self, batch: Any) -> Any:
        # Convert items to tensor if they are not.
        # Checking first makes sure we do not distrub memory pinning
        if not utils.contains_tensors(batch):
            batch = utils.to_tensor(batch)
        if self.processor.supports_gpu:
            # Move to CUDA first.
            batch = utils.to_device(batch, self.device)
            batch = self.processor(batch)
        else:
            batch = self.processor(batch)
            batch = utils.to_device(batch, self.device)
        return batch

    @abstractmethod
    def train_step(self, batches: Any, step: int, total_steps: int) -> Dict:
        """
        Train the model. Should return a dict of loggable values
        """
        raise NotImplemented

    def validation_step(self, batch: Any) -> Dict:
        """
        perform a validation step. Should return a dict of loggable values.
        """
        raise NotImplementedError

    def env_step(self, env: gym.Env, step: int, total_steps: int) -> Dict:
        """
        Perform any extra training operations. This is done before the train step is called.
        A common use case for this would be stepping the environment etc.
        """
        return {}

    def validation_extras(self, path: str, step: int) -> Dict:
        """
        Perform any extra validation operations.
        A common usecase for this is saving visualizations etc.
        """
        return {}

    def _predict(self, batch: Any, **kwargs) -> Any:
        """
        Internal prediction function, can be overridden
        By default, we call torch.no_grad(). If this behavior isn't desired,
        override the _predict funciton in your algorithm.
        """
        with torch.no_grad():
            if len(kwargs) > 0:
                raise ValueError("Default predict method does not accept key word args, but they were provided.")
            pred = self.network(batch)
        return pred

    def predict(self, batch: Any, is_batched: bool = False, **kwargs) -> Any:
        is_np = not utils.contains_tensors(batch)
        if not is_batched:
            # Unsqeeuze everything
            batch = utils.unsqueeze(batch, 0)
        batch = self.format_batch(batch)
        pred = self._predict(batch, **kwargs)
        if not is_batched:
            pred = utils.get_from_batch(pred, 0)
        if is_np:
            pred = utils.to_np(pred)
        return pred
