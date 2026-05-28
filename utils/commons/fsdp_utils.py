from typing import Dict
import torch
from torch._dynamo.eval_frame import OptimizedModule
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer


def get_model_states(model: Module, *, sharded: bool = False):
    """
    Get model state dict.
    Call by all ranks.
    If full state dict, only use the result on rank 0.
    If sharded state dict, only for fsdp model.
    """
    if isinstance(model, OptimizedModule):
        model = model._orig_mod
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(model, FullyShardedDataParallel):
        configure_fsdp_states(model, sharded=sharded)
    return model.state_dict()


def get_optimizer_states(optimizer: Optimizer):
    return optimizer.state_dict()


def get_fsdp_optimizer_states(
        optimizer: Optimizer,
        model: FullyShardedDataParallel,
        *,
        sharded: bool = False,
):
    """
    Get fsdp optimizer state dict.
    Call by all ranks.
    If full state dict, only use the result on rank 0.
    If sharded state dict, only for fsdp model.
    """
    configure_fsdp_states(model, sharded=sharded)
    states = optimizer.state_dict()
    states = FullyShardedDataParallel.optim_state_dict(
        model=model,
        optim=optimizer,
        optim_state_dict=states,
    )
    return states


def set_fsdp_optimizer_states(
        states: Dict[str, torch.Tensor],
        optimizer: Optimizer,
        model: FullyShardedDataParallel,
):
    """
    Set fsdp optimizer state dict.
    Call by all ranks.
    """
    configure_fsdp_states(model, rank0_only=False)
    states = FullyShardedDataParallel.optim_state_dict_to_load(
        model=model,
        optim=optimizer,
        optim_state_dict=states,
    )
    optimizer.load_state_dict(states)


def configure_fsdp_states(
        model: FullyShardedDataParallel,
        *,
        rank0_only: bool = True,
        sharded: bool = False,
):
    """
    Configure fsdp state dict type.
    """
    if not sharded:
        FullyShardedDataParallel.set_state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=rank0_only
            ),
        )
    else:
        FullyShardedDataParallel.set_state_dict_type(
            module=model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        )
