import os

import torch
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,BackwardPrefetch, CPUOffload

from utils.commons.io import print_once

in_node_device_mesh = None
# local_shard_size = 8
def get_in_node_device_mesh():
    global in_node_device_mesh
    # global local_shard_size
    if in_node_device_mesh == None:
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = 1
        # assert world_size % local_shard_size == 0
        # in_node_device_mesh = init_device_mesh('cuda', (world_size // local_shard_size, local_shard_size))
        in_node_device_mesh = init_device_mesh('cuda', (1, world_size))
    return in_node_device_mesh


def shard_model_in_node(model: torch.nn.Module, transformer_cls, device) -> FSDP:
    transformer_class = transformer_cls
    layer_wrap = ModuleWrapPolicy([transformer_class,])
    device_mesh = get_in_node_device_mesh()
    sharding_strategy = ShardingStrategy.HYBRID_SHARD
    param_dtype = torch.bfloat16
    print_once(f'using {param_dtype} param_dtype for FSDP for model qwen...')
    model = FSDP(
        model,
        auto_wrap_policy=layer_wrap,
        device_id=device,
        sharding_strategy=sharding_strategy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float,
            buffer_dtype=param_dtype,
        ),
        sync_module_states=False,
        limit_all_gathers=True,
        use_orig_params=True,
        device_mesh=device_mesh,
    )
    # FSDP.set_state_dict_type(model, StateDictType.LOCAL_STATE_DICT)
    # why do we need this? Shilong found this would cause error when saving checkpoint? is it ralated to pytorch version?
    #TODO(shuo) why do this sync?
    torch.cuda.synchronize()
    return model