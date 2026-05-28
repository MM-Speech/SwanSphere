import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.eg3ds.models.dual_discriminator import filtered_resizing
from modules.eg3ds.torch_utils.ops import upfirdn2d
from deep_3drecon.deep_3drecon_models.arcface_torch.backbones.iresnet import iresnet50
from inference.os_avatar.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from tasks.eg3ds.loss_utils.arcface_loss.loss import ArcFaceLoss

csim_loss_fn = None

def read_video_to_normalized_frames(vid_name):
    frames = []
    cap = cv2.VideoCapture(vid_name)
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    frames = np.stack(frames)
    frames = torch.tensor(frames) / 127.5 - 1
    frames = frames.permute(0, 3, 1, 2)
    return frames

def load_img_or_video_to_normalized_btwc_tensor(img_or_video_name):
    if img_or_video_name[-4:] in ['.png', '.jpg']:
        img = load_img_to_normalized_512_bchw_tensor(img_or_video_name)
        img = F.interpolate(img, size=(112,112))
    elif img_or_video_name[-4:] in ['.mp4']:
        img = read_video_to_normalized_frames(img_or_video_name)
        img = F.interpolate(img, size=(112,112))
    else:
        raise NotImplementedError(f"unsupport type: {img_or_video_name[-4:]}")
    return img
    
@torch.no_grad()
def cal_csim(img_or_video_name1, img_or_video_name2):
    global csim_loss_fn
    if csim_loss_fn is None:
        csim_loss_fn = ArcFaceLoss().cuda()
    img1 = load_img_or_video_to_normalized_btwc_tensor(img_or_video_name1).cuda()
    img2 = load_img_or_video_to_normalized_btwc_tensor(img_or_video_name2).cuda()
    length = min(len(img1), len(img2))
    if length > 1:
        img1 = img1[:length]
        img2 = img2[:length]
    csim = csim_loss_fn.cal_csim(img1, img2)
    return csim.item()

if __name__ == '__main__':

    # import glob, tqdm, numpy
    # csim_lst = []
    # method_name = 'fsfacev2v'
    # src_name_pattern = f"icml_test_data/fewshot/same/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/fewshot/same/{method_name}', 'icml_test_data/inputs/VD_same/video')
    #     try:
    #         csim = cal_csim(src_name, gt_name)
    #     except:
    #         pass
    #     csim_lst.append(csim)
    # print(method_name,csim_lst)
    # print(method_name,numpy.mean(csim_lst))

    gt_name = '/mnt/bn/sa-ag-data/yezhenhui/projects/GeneFace_private/Obama_test.mp4'

    src_name = 'case_study_sd_hybrid/Obama_data_5s.mp4'
    csim = cal_csim(src_name, gt_name)
    print(src_name, ": ", csim)

    src_name = 'case_study_sd_hybrid/Obama_data_10s.mp4'
    csim = cal_csim(src_name, gt_name)
    print(src_name, ": ", csim)

    src_name = 'case_study_sd_hybrid/Obama_data_15s.mp4'
    csim = cal_csim(src_name, gt_name)
    print(src_name, ": ", csim)

    src_name = 'case_study_sd_hybrid/Obama_data_30s.mp4'
    csim = cal_csim(src_name, gt_name)
    print(src_name, ": ", csim)

    src_name = 'case_study_sd_hybrid/Obama_data_60s.mp4'
    csim = cal_csim(src_name, gt_name)
    print(src_name, ": ", csim)

    src_name = 'case_study_sd_hybrid/Obama_data_180s.mp4'
    csim = cal_csim(src_name, gt_name)
    print(src_name, ": ", csim)