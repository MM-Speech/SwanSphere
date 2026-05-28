import json
import glob
import cv2
import torch
import os
import uuid
import tqdm
import hashlib
import numpy as np
import scipy
import imageio.v3 as iio

import modules.eg3ds.dnnlib as dnnlib
from modules.eg3ds.metrics.metric_utils import FeatureStats
from utils.commons.multiprocess_utils import multiprocess_run_tqdm


def load_job(img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img) / 127.5 - 1
    return img



def cal_fid_stats_for_vid_names(vid_names, ds_name, dump=False):
    num_gpus = 1
    rank = 0 
    import sys, pickle
    sys.path.append("/mnt/bn/sa-ag-data/yezhenhui/projects/GeneFace_private/modules/eg3ds")
    # you can download this ckpt at https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
    ckpt_name = '/mnt/bn/sa-ag-data/yezhenhui/nfs/myenv/cache/useful_ckpts/inception-2015-12-05.pkl'
    with open(ckpt_name, 'rb') as f:
        detector = pickle.load(f).to("cuda")
    del sys.path[-1]

    # try to obtain cached gt feats from disk
    args = dict(ds_name=ds_name, detector_url='inception_net_v3', detector_kwargs={'return_features': True}, stats_kwargs={"capture_mean_cov": True})
    md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
    cache_tag = f'{ds_name}-inception_net_v3-{md5.hexdigest()}'
    cache_file_name = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')
    batch_size = 64
    # Check if the file exists (all processes must agree).
    cache_is_ready = os.path.isfile(cache_file_name) if rank == 0 and dump else False
    if num_gpus > 1:
        cache_is_ready = torch.as_tensor(cache_is_ready, dtype=torch.float32, device="cuda")
        torch.distributed.broadcast(tensor=cache_is_ready, src=0)
        cache_is_ready = (float(cache_is_ready.cpu()) != 0)
    if cache_is_ready:
        if rank == 0:
            print(f"load feats from disk: {cache_file_name}")
        stats = FeatureStats.load(cache_file_name)
    else:
        img_lst = []
        for vid_name in vid_names:
            frames = iio.imread(vid_name, plugin="pyav")
            img_lst.append(frames)
        all_imgs = np.concatenate(img_lst, axis=0) / 127.5 - 1
        all_imgs = torch.tensor(all_imgs).permute(0, 3, 1, 2)

        stats = FeatureStats(max_items=len(all_imgs), capture_mean_cov=True)
        
        for i in tqdm.trange(len(all_imgs)//batch_size, desc="calculating feats"):
            images = all_imgs[batch_size*i:batch_size*i+batch_size]
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            if images.dtype != torch.uint8:
                images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            features = detector(images.to("cuda"), return_features=True)
            stats.append_torch(features, num_gpus=num_gpus, rank=rank)
        # dump the gt feats to disk
        if rank == 0 and dump:
            os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
            temp_file = cache_file_name + '.' + uuid.uuid4().hex
            stats.save(temp_file)
            os.replace(temp_file, cache_file_name) # atomic
            print(f"dumped feats into disk: {cache_file_name}")
    return stats

def cal_fid_stats_for_img_names(img_names, ds_name, dump=False):
    num_gpus = 1
    rank = 0 
    import sys, pickle
    sys.path.append("/mnt/bn/sa-ag-data/yezhenhui/projects/GeneFace_private/modules/eg3ds")
    # you can download this ckpt at https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
    ckpt_name = '/mnt/bn/sa-ag-data/yezhenhui/nfs/myenv/cache/useful_ckpts/inception-2015-12-05.pkl'
    with open(ckpt_name, 'rb') as f:
        detector = pickle.load(f).to("cuda")
    del sys.path[-1]

    # try to obtain cached gt feats from disk
    args = dict(ds_name=ds_name, detector_url='inception_net_v3', detector_kwargs={'return_features': True}, stats_kwargs={"capture_mean_cov": True})
    md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
    cache_tag = f'{ds_name}-inception_net_v3-{md5.hexdigest()}'
    cache_file_name = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')
    batch_size = 64
    # Check if the file exists (all processes must agree).
    cache_is_ready = os.path.isfile(cache_file_name) if rank == 0 and dump else False
    if num_gpus > 1:
        cache_is_ready = torch.as_tensor(cache_is_ready, dtype=torch.float32, device="cuda")
        torch.distributed.broadcast(tensor=cache_is_ready, src=0)
        cache_is_ready = (float(cache_is_ready.cpu()) != 0)
    if cache_is_ready:
        if rank == 0:
            print(f"load feats from disk: {cache_file_name}")
        stats = FeatureStats.load(cache_file_name)
    else:
        img_lst = []
        if len(img_names) < 100:
            for img_name in img_names:
                img_lst.append(load_job(img_name))
        else:
            for (i, res) in multiprocess_run_tqdm(load_job, img_names, desc='loading img into disc', num_workers=min(32, max(1, len(img_names)//20))):
                img_lst.append(res)
        all_imgs = torch.stack(img_lst, dim=0).permute(0, 3, 1, 2)

        stats = FeatureStats(max_items=len(img_names), capture_mean_cov=True)
        for i in tqdm.trange(len(img_names)//batch_size, desc="calculating feats"):
            images = all_imgs[batch_size*i:batch_size*i+batch_size]
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            if images.dtype != torch.uint8:
                images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            features = detector(images.to("cuda"), return_features=True)
            stats.append_torch(features, num_gpus=num_gpus, rank=rank)
        if len(img_names) % batch_size != 0:
            images = all_imgs[batch_size*(len(img_names)//batch_size):]
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])
            if images.dtype != torch.uint8:
                images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            features = detector(images.to("cuda"), return_features=True)
            stats.append_torch(features, num_gpus=num_gpus, rank=rank)
        # dump the gt feats to disk
        if rank == 0 and dump:
            os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
            temp_file = cache_file_name + '.' + uuid.uuid4().hex
            stats.save(temp_file)
            os.replace(temp_file, cache_file_name) # atomic
            print(f"dumped feats into disk: {cache_file_name}")
    return stats

def cal_fid_given_stats(stats_gt, stats_fake):
    mu_real, sigma_real = stats_gt.get_mean_cov()
    mu_gen, sigma_gen = stats_fake.get_mean_cov()
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    fid = float(fid)
    return fid

def cal_fid(vid_name1, vid_name2):
    stats_1 = cal_fid_stats_for_vid_names([vid_name1], 'vid1')
    stats_2 = cal_fid_stats_for_vid_names([vid_name2], 'vid2')
    fid = cal_fid_given_stats(stats_1, stats_2)
    return fid
    
if __name__ == '__main__':
    # stats_1 = cal_fid_stats_for_vid_names(['icml_test_data/inputs/nerfs/may.mp4'], 'vid')
    # stats_2 = cal_fid_stats_for_vid_names(['icml_test_data/person_specific/radnerf/may.mp4'], 'vid2')
    # stats_3 = cal_fid_stats_for_vid_names(['icml_test_data/person_specific/ernerf/may.mp4'], 'vid3')
    # stats_4 = cal_fid_stats_for_vid_names(['icml_test_data/person_specific/general3d/may.mp4'], 'vid4')
    # fid1 = cal_fid_given_stats(stats_1, stats_2)
    # fid2 = cal_fid_given_stats(stats_1, stats_3)
    # fid3 = cal_fid_given_stats(stats_1, stats_4)
    # print(fid1)
    # print(fid2)
    # print(fid3)


    import glob, tqdm, numpy, traceback

    fid_lst = []
    method_name = 'fsfacev2v'
    src_name_pattern = f"icml_test_data/fewshot/cross/{method_name}/*.mp4"
    src_names = glob.glob(src_name_pattern)
    for src_name in tqdm.tqdm(src_names):
        try:
            gt_name = src_name.replace(f'icml_test_data/fewshot/cross/{method_name}', 'icml_test_data/inputs/VD_cross/video') 
            stats_1 = cal_fid_stats_for_vid_names([gt_name], 'vid')
            stats_2 = cal_fid_stats_for_vid_names([src_name], 'vid4')
            fid = cal_fid_given_stats(stats_1, stats_2)
            fid_lst.append(fid)
        except:
            pass
    print(method_name,fid_lst)
    print(method_name,numpy.mean(fid_lst))
