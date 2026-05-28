import contextlib
import glob
import os
import re
import subprocess
import traceback

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


@contextlib.contextmanager
def dist_load(path):
    if not dist.is_initialized() or dist.get_world_size() == 1 or os.path.realpath(path).startswith('/dev/shm'):
        yield path
    else:
        from utils.commons.hparams import hparams
        from utils.commons.trainer import LOCAL_RANK
        tmpdir = '/dev/shm'
        assert len(os.path.basename(path)) > 0
        shm_ckpt_path = f'{tmpdir}/{hparams["exp_name"]}/{os.path.basename(path)}'
        if LOCAL_RANK == 0:
            subprocess.check_call(
                f'mkdir -p {os.path.dirname(shm_ckpt_path)}; '
                f'cp -Lr {path} {shm_ckpt_path}', shell=True)
        dist.barrier()
        yield shm_ckpt_path
        dist.barrier()
        if LOCAL_RANK == 0:
            subprocess.check_call(f'rm -rf {shm_ckpt_path}', shell=True)


def torch_load_dist(path, map_location='cpu', mmap=None):
    with dist_load(path) as tmp_path:
        checkpoint = torch.load(tmp_path, map_location=map_location, mmap=mmap)
    return checkpoint


def get_last_checkpoint(work_dir, steps=None, map_location='cpu', mmap=None, return_step=False):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch_load_dist(last_ckpt_path, map_location=map_location, mmap=mmap)
    if not return_step:
        return checkpoint, last_ckpt_path
    else:
        if last_ckpt_path is not None:
            pattern = r'.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
            global_steps = int(re.findall(pattern, last_ckpt_path)[0])
        else:
            global_steps = 0
        return checkpoint, last_ckpt_path, global_steps


def get_all_ckpts(work_dir, steps=None):
    if steps is None or steps == 0:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_{steps}.ckpt'
    pattern = '.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
    all_ckpts = [x for x in glob.glob(ckpt_path_pattern) if len(re.findall(pattern, x)) > 0]
    return sorted(all_ckpts, key=lambda x: -int(re.findall(pattern, x)[0]))


def get_all_ckpt_steps(work_dir):
    ckpt_path_pattern = f'{work_dir}/model_ckpt_steps_*.ckpt'
    pattern = '.*steps_(\d+)(?:\.ckpt|_backbone\.ckpt)'
    all_ckpts = [x for x in glob.glob(ckpt_path_pattern) if len(re.findall(pattern, x)) > 0]
    steps = [int(re.findall(pattern, c)[0]) for c in all_ckpts]
    steps = sorted(steps)
    return steps


def load_ckpt(cur_model, ckpt_base_dir, model_name='model', force=True, strict=True,
              silent=False, load_opt=False, opts=None, steps=None, checkpoint=None, ckpt_path='', delete_unmatch=True, map_location='cpu', mmap=None):
    if checkpoint is None:
        if os.path.isfile(ckpt_base_dir):
            base_dir = os.path.dirname(ckpt_base_dir)
            ckpt_path = ckpt_base_dir
            checkpoint = torch_load_dist(ckpt_base_dir, map_location=map_location, mmap=mmap)
        else:
            base_dir = ckpt_base_dir
            if load_opt:
                checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
            else:
                ckpt_path = f'{ckpt_base_dir}/model_only_last.ckpt'
                if os.path.exists(ckpt_path):
                    checkpoint = torch_load_dist(ckpt_path, map_location=map_location, mmap=mmap)
                else:
                    checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir, steps)
    if checkpoint is not None:
        # ===== 新增：聚合 missing / unmatched key，避免在不同层重复打印 =====
        aggregated_missing = set()
        aggregated_unmatched = set()

        def _canonical_key(k: str) -> str:
            # 将形如 ".0." / ".1." 这类层索引归一化，避免“同一模块不同层”重复打印
            # 例如 encoder.layers.0.self_attn.q_proj.weight -> encoder.layers.<N>.self_attn.q_proj.weight
            return re.sub(r'\.\d+(\.|$)', '.<N>\\1', k)
        # ====================================================================

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict_all = {
            k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        if not isinstance(cur_model, list):
            cur_models = [cur_model]
            model_names = [model_name]
        else:
            cur_models = cur_model
            model_names = model_name
        for model_name, cur_model in zip(model_names, cur_models):
            if isinstance(cur_model, DistributedDataParallel):
                cur_model = cur_model.module
            device = next(cur_model.parameters()).device
            if '.' not in model_name:
                state_dict = state_dict_all[model_name]
            else:
                base_model_name = model_name.split('.')[0]
                rest_model_name = model_name[len(base_model_name) + 1:]
                state_dict = {
                    k[len(rest_model_name) + 1:]: v for k, v in state_dict_all[base_model_name].items()
                    if k.startswith(f'{rest_model_name}.')}
            state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            if not strict and delete_unmatch:
                try:
                    cur_model.load_state_dict(state_dict, strict=True)
                    if not silent:
                        print(f"| loaded '{model_name}' from '{ckpt_path}' with strict=True.")
                except:
                    cur_model_state_dict = cur_model.state_dict()
                    cur_model_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in
                                            cur_model_state_dict.items()}

                    state_dict, repaired, removed = repair_unmatched_state_dict(cur_model_state_dict, state_dict, silent)

            load_results = cur_model.load_state_dict(state_dict, strict=strict)
            cur_model.to(device)
            if not silent:
                print(f"| loaded '{model_name}' from '{ckpt_path}'.")
                missing_keys, unexpected_keys = load_results.missing_keys, load_results.unexpected_keys
                print(f"| Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                # 新增：记录 missing 的 key
                for k in missing_keys:
                    aggregated_missing.add(_canonical_key(k))

        # 新增：在所有 model 都 load 完之后，统一按模块去重打印一次
        if not silent:
            if aggregated_missing:
                print("| ===== Missing key names (deduplicated by module) =====")
                for k in sorted(aggregated_missing):
                    print("|   ", k)
            if aggregated_unmatched:
                print("| ===== Unmatched (size-mismatch) key names (deduplicated by module) =====")
                for k in sorted(aggregated_unmatched):
                    print("|   ", k)

        if load_opt:
            if "optimizer_states" in checkpoint:
                optimizer_states = checkpoint['optimizer_states']
            else:
                optimizer_states = torch_load_dist(ckpt_path[:-5] + '_optm.ckpt', map_location=map_location, mmap=mmap)
            assert len(opts) == len(optimizer_states)
            for optimizer, opt_state in zip(opts, optimizer_states):
                opt_state = {k.replace('_orig_mod.', ''): v for k, v in opt_state.items()}
                if optimizer is None:
                    return
                try:
                    optimizer.load_state_dict(opt_state)
                    for i, state in enumerate(optimizer.state.values()):
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                except ValueError:
                    print(f"| WARMING: optimizer {optimizer} parameters not match !!!")
        return checkpoint.get('global_step', 0)
    else:
        e_msg = f"| ckpt not found in {base_dir}."
        if force:
            assert False, e_msg
        else:
            print(e_msg)

            

def repair_unmatched_state_dict(cur_model_state_dict, state_dict, silent=False):
    repaired, removed = 0, []
    for key, old_param in list(state_dict.items()):
        if key in cur_model_state_dict:
            new_param = cur_model_state_dict[key]
            if new_param.shape != old_param.shape:
                print("| Unmatched keys:", key, "cur model:", tuple(new_param.shape), "ckpt model:", tuple(old_param.shape))
                merged = merge_unmatched_params(new_param, old_param, key)
                if merged is not None:
                    state_dict[key] = merged
                    repaired += 1
                    print(f"| Unmatched key {key} is partially loaded")
                else:
                    removed.append(key)

    for key in removed:
        del state_dict[key]
    
    if repaired > 0 and not silent:
        print(f"| Partially loaded {repaired} tensor(s) with size mismatch by copying overlapping slices.")

    return state_dict, repaired, removed


def merge_unmatched_params(new_tensor: torch.Tensor, old_tensor: torch.Tensor, key: str = ""):
    """
    Return a tensor with the same shape as new_tensor, where the overlapping
    slice is copied from old_tensor. If not possible, return None.
    """
    try:
        if new_tensor.ndim != old_tensor.ndim:
            return None
        old_tensor = old_tensor.to(dtype=new_tensor.dtype)

        slices = tuple(slice(0, min(n, o)) for n, o in zip(new_tensor.shape, old_tensor.shape))

        out = new_tensor.clone()
        out[slices] = old_tensor[slices]
        return out
    except Exception as e:
        print(f"| merge failed on '{key}': {e}")
        return None


def load_with_size_mismatch(model, state_dict, prefix=""):
    current_model_dict = model.state_dict()
    cm_keys = current_model_dict.keys()
    mismatch_keys = {k.replace(prefix, "") for k, v in state_dict.items() if k.replace(prefix, "") in cm_keys and v.size() != current_model_dict[k.replace(prefix, "")].size()}
    new_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.replace(prefix, "") in cm_keys and v.size() == current_model_dict[k.replace(prefix, "")].size()}
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"| mismatch keys: ", mismatch_keys)
    if len(missing_keys) > 0:
        print(f"| missing_keys in: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"| unexpected_keys in: {unexpected_keys}")
