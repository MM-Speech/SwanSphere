import os
import cv2
import imageio
import numpy as np
import tqdm
import torch
import math
import torch.nn.functional as F
import lpips
from utils.commons.tensor_utils import convert_to_np

lpips_fn = None
ssim_fn = None

def read_video_to_normalized_frames(vid_name):
    frames = []
    cap = cv2.VideoCapture(vid_name)
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_bgr = cv2.resize(frame_bgr, (512,512))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    frames = np.stack(frames)
    frames = torch.tensor(frames) / 127.5 - 1
    frames = frames.permute(0, 3, 1, 2)
    return frames

# structural similarity index
class SSIM:
    '''
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    '''
    def __init__(self):
        pass

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

def psnr_fn(imgs1, imgs2):
    imgs1 = convert_to_np(imgs1)
    imgs2 = convert_to_np(imgs2)
    mse_lst = []
    for i in range(len(imgs1)):
        img1 = imgs1[i]
        img2 = imgs2[i]
        mse = ((img1-img2)**2).mean()
        mse_lst.append(mse)
    mse = np.mean(np.array(mse_lst))
    psnr = 10*np.log10(1/mse)
    return psnr

def cal_all_mse_metrics(video_name1, video_name2):
    global ssim_fn
    if ssim_fn is None:
        ssim_fn = SSIM()
    img1 = read_video_to_normalized_frames(video_name1).cuda()
    img1 = F.interpolate(img1, (512,512))
    img2 = read_video_to_normalized_frames(video_name2).cuda()
    img2 = F.interpolate(img2, (512,512))

    length = min(len(img1), len(img2))
    if length > 1:
        img1 = img1[:length]
        img2 = img2[:length]
        
    ssim = ssim_fn(img1, img2).item()

    l1 = (img1 - img2).abs().mean().item()
    psnr = psnr_fn(img1, img2)

    global lpips_fn
    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net='alex')
        lpips_fn.cuda()
    lpips_loss = lpips_fn(img1,img2).mean().item()

    ret = {
        'ssim': ssim,
        'l1': l1,
        'psnr': psnr,
        'lpips': lpips_loss,
    }
    return ret

if __name__ == '__main__':
    # vid_name1 = "gf2_iclr_test_data/self_videos/-HF7vQHhp3c_1.mp4"
    # vid_name2 = "infer_out/tmp/-HF7vQHhp3c_1_-HF7vQHhp3c_1.mp4"
    # ret = cal_all_mse_metrics(vid_name1, vid_name2)
    # print(ret)

    import glob, tqdm, numpy

    # l1_lst = []
    # psnr_lst = []
    # ssim_lst = []
    # lpips_lst = []
    # method_name = 'facev2v'
    # src_name_pattern = f"icml_test_data/oneshot_VD/same/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/same/results/{method_name}', 'icml_test_data/inputs/VD_same/video')
    #     ret = cal_all_mse_metrics(src_name, gt_name)
    #     l1_lst.append(ret['l1'])
    #     psnr_lst.append(ret['psnr'])
    #     ssim_lst.append(ret['ssim'])
    #     lpips_lst.append(ret['lpips'])
    # print(method_name,"l1",numpy.mean(l1_lst))
    # print(method_name,"psnr",numpy.mean(psnr_lst))
    # print(method_name,"ssim",numpy.mean(ssim_lst))
    # print(method_name,"lpips",numpy.mean(lpips_lst))

    # l1_lst = []
    # psnr_lst = []
    # ssim_lst = []
    # lpips_lst = []
    # method_name = 'hidenerf'
    # src_name_pattern = f"icml_test_data/oneshot_VD/same/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/same/results/{method_name}', 'icml_test_data/inputs/VD_same/video')
    #     ret = cal_all_mse_metrics(src_name, gt_name)
    #     l1_lst.append(ret['l1'])
    #     psnr_lst.append(ret['psnr'])
    #     ssim_lst.append(ret['ssim'])
    #     lpips_lst.append(ret['lpips'])
    # print(method_name,"l1",numpy.mean(l1_lst))
    # print(method_name,"psnr",numpy.mean(psnr_lst))
    # print(method_name,"ssim",numpy.mean(ssim_lst))
    # print(method_name,"lpips",numpy.mean(lpips_lst))

    # l1_lst = []
    # psnr_lst = []
    # ssim_lst = []
    # lpips_lst = []
    # method_name = 'tps'
    # src_name_pattern = f"icml_test_data/oneshot_VD/same/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/same/results/{method_name}', 'icml_test_data/inputs/VD_same/video')
    #     ret = cal_all_mse_metrics(src_name, gt_name)
    #     l1_lst.append(ret['l1'])
    #     psnr_lst.append(ret['psnr'])
    #     ssim_lst.append(ret['ssim'])
    #     lpips_lst.append(ret['lpips'])
    # print(method_name,"l1",numpy.mean(l1_lst))
    # print(method_name,"psnr",numpy.mean(psnr_lst))
    # print(method_name,"ssim",numpy.mean(ssim_lst))
    # print(method_name,"lpips",numpy.mean(lpips_lst))

    # l1_lst = []
    # psnr_lst = []
    # ssim_lst = []
    # lpips_lst = []
    # method_name = 'dpe'
    # src_name_pattern = f"icml_test_data/oneshot_VD/same/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/same/results/{method_name}', 'icml_test_data/inputs/VD_same/video')
    #     ret = cal_all_mse_metrics(src_name, gt_name)
    #     l1_lst.append(ret['l1'])
    #     psnr_lst.append(ret['psnr'])
    #     ssim_lst.append(ret['ssim'])
    #     lpips_lst.append(ret['lpips'])
    # print(method_name,"l1",numpy.mean(l1_lst))
    # print(method_name,"psnr",numpy.mean(psnr_lst))
    # print(method_name,"ssim",numpy.mean(ssim_lst))
    # print(method_name,"lpips",numpy.mean(lpips_lst))


    l1_lst = []
    psnr_lst = []
    ssim_lst = []
    lpips_lst = []
    method_name = 'fsfacev2v'
    src_name_pattern = f"icml_test_data/fewshot/same/{method_name}/*.mp4"
    src_names = glob.glob(src_name_pattern)
    for src_name in tqdm.tqdm(src_names):
        gt_name = src_name.replace(f'icml_test_data/fewshot/same/{method_name}', 'icml_test_data/inputs/VD_same/video')
        ret = cal_all_mse_metrics(src_name, gt_name)
        l1_lst.append(ret['l1'])
        psnr_lst.append(ret['psnr'])
        ssim_lst.append(ret['ssim'])
        lpips_lst.append(ret['lpips'])
    print(method_name,"l1",numpy.mean(l1_lst))
    print(method_name,"l1",l1_lst)
    print(method_name,"psnr",numpy.mean(psnr_lst))
    print(method_name,"psnr",psnr_lst)
    print(method_name,"ssim",numpy.mean(ssim_lst))
    print(method_name,"ssim",ssim_lst)
    print(method_name,"lpips",numpy.mean(lpips_lst))
    print(method_name,"lpips",lpips_lst)