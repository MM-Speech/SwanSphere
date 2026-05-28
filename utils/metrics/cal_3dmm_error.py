import numpy as np
# from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
import deep_3drecon
import cv2
import torch
import mediapipe
from utils.commons.tensor_utils import convert_to_tensor
import glob, tqdm, numpy

face_reconstructor = None
mp_face_mesh = mediapipe.solutions.face_mesh


index_lm68_from_lm468 = [127,234,93,132,58,136,150,176,152,400,379,365,288,361,323,454,356,70,63,105,66,107,336,296,334,293,300,168,197,5,4,75,97,2,326,305,
                         33,160,158,133,153,144,362,385,387,263,373,380,61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178]

# landmark detection in Deep3DRecon
def lm68_2_lm5(in_lm):
    # in_lm: shape=[68,2]
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    # 将上述特殊角点的数据取出，得到5个新的角点数据，拼接起来。
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    # 将第一个角点放在了第三个位置
    lm = lm[[1,2,0,3,4],:2]
    return lm

def extract_lms_mediapipe_job(frames):
    try:
        if frames is None:
            return None
        with mp_face_mesh.FaceMesh(
                            static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) as face_mesh:
            ldms_normed = []
            frame_i = 0
            frame_ids = []
            for i in tqdm.trange(len(frames), desc="extracting mediapipe landmarks..."):
                # Convert the BGR image to RGB before processing.
                ret = face_mesh.process(frames[i])
                # Print and draw face mesh landmarks on the image.
                if not ret.multi_face_landmarks:
                    print(f"Skip Item: Caught errors when mediapipe get face_mesh, maybe No face detected in some frames!")
                    return None
                else:
                    myFaceLandmarks = []
                    lms = ret.multi_face_landmarks[0]
                    for lm in lms.landmark:
                        myFaceLandmarks.append([lm.x, lm.y, lm.z])
                    ldms_normed.append(myFaceLandmarks)
                frame_ids.append(frame_i)
                frame_i += 1
        bs, H, W, _ = frames.shape
        ldms478 = np.array(ldms_normed)[..., :2] * np.array([H, W]).reshape([1,1,2]) # [T, 478, 2]
        lm68 = ldms478[:, index_lm68_from_lm468, :] # [T, 68, 2]
        lm5_lst = [lm68_2_lm5(lm68[i]) for i in range(lm68.shape[0])]
        lm5 = np.stack(lm5_lst)
        return ldms478, lm68, lm5
    except Exception as e:
        print(e)
        return None


def get_coeff_dict(fname):
    global face_reconstructor
    if face_reconstructor is None:
        face_reconstructor = deep_3drecon.Reconstructor()
    assert fname.endswith(".mp4")

    cap = cv2.VideoCapture(fname)
    frames = []
    cnt = 0
    print(f"loading video ...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        cnt += 1

    lm478_arr, lm68_arr, lm5_arr = extract_lms_mediapipe_job(np.stack(frames))

    video_rgb = np.stack(frames) # [t, 224,224, 3]
    lm478_arr =lm478_arr.reshape([cnt, 478, 2])
    lm68_arr =lm68_arr.reshape([cnt, 68, 2])
    lm5_arr = lm5_arr.reshape([cnt, 5, 2])
    num_frames = cnt
    batch_size = 32
    iter_times = num_frames // batch_size
    last_bs = num_frames % batch_size
    coeff_lst = []
    for i_iter in tqdm.trange(iter_times, desc="start extracting 3DMM..."):
        start_idx = i_iter * batch_size
        batched_images = video_rgb[start_idx: start_idx + batch_size]
        batched_lm5 = lm5_arr[start_idx: start_idx + batch_size]
        coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
        coeff_lst.append(coeff)
    if last_bs != 0:
        batched_images = video_rgb[-last_bs:]
        batched_lm5 = lm5_arr[-last_bs:]
        coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
        coeff_lst.append(coeff)
    coeff = np.concatenate(coeff_lst,axis=0).reshape([cnt, -1])
    ret_dict = {
            'id': coeff[..., :80],  # identity, [b, t, c=80] 
            'exp': coeff[..., 80:144],  # expression, [b, t, c=80]
            'euler': coeff[..., 224:227],  # euler euler for pose, [b, t, c=3]
            'trans':  coeff[..., 254:257], # translation, [b, t, c=3]
        }
    return ret_dict

def cal_3dmm_error(vid_or_npy_name1, vid_or_npy_name2):
    coeff_dict1 = convert_to_tensor(get_coeff_dict(vid_or_npy_name1))
    coeff_dict2 = convert_to_tensor(get_coeff_dict(vid_or_npy_name2))
    length = min(len(coeff_dict1['exp']), len(coeff_dict2['exp']))
    expression_error = coeff_dict1['exp'][:length] - coeff_dict2['exp'][:length]
    aed = expression_error.abs().mean().item()
    apd_v1 = (torch.cat([coeff_dict1['euler'][:length],coeff_dict1['trans'][:length]]) - torch.cat([coeff_dict2['euler'][:length],coeff_dict2['trans'][:length]])).abs().mean()
    apd_v2 = (coeff_dict1['euler'][:length] - coeff_dict2['euler'][:length]).abs().mean()
    return aed, apd_v1, apd_v2

if __name__ == '__main__':
    # vid_name1 = "icml_test_data/inputs/nerfs/may.mp4"
    # vid_name2 = "icml_test_data/person_specific/radnerf/may.mp4"
    # ret = cal_3dmm_error(vid_name1, vid_name2)
    # print("RADNeRF 3dmm", ret)

    # vid_name2 = "icml_test_data/person_specific/ernerf/may.mp4"
    # ret = cal_3dmm_error(vid_name1, vid_name2)
    # print("ERNeRF 3dmm", ret)

    # vid_name2 = "icml_test_data/person_specific/general3d/may.mp4"
    # ret = cal_3dmm_error(vid_name1, vid_name2)
    # print("RADNeRF 3dmm", ret)


    aed_lst = []
    apd_lst = []
    apd2_lst = []
    method_name = 'fsfacev2v'
    src_name_pattern = f"icml_test_data/fewshot/cross/{method_name}/*.mp4"
    src_names = glob.glob(src_name_pattern)
    for src_name in tqdm.tqdm(src_names):
        try:
            gt_name = src_name.replace(f'icml_test_data/fewshot/cross/{method_name}', 'icml_test_data/inputs/VD_cross/video')
            aed,apd,apd2 = cal_3dmm_error(src_name, gt_name)
            aed_lst.append(aed)
            apd_lst.append(apd)
            apd2_lst.append(apd2)
        except:
            pass
    print(method_name,numpy.mean(aed_lst), numpy.mean(apd_lst), numpy.mean(apd2_lst))

    # aed_lst = []
    # apd_lst = []
    # apd2_lst = []
    # method_name = 'tps'
    # src_name_pattern = f"icml_test_data/oneshot_VD/cross/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/cross/results/{method_name}', 'icml_test_data/inputs/VD_cross/video')
    #     try:
    #         aed,apd,apd2 = cal_3dmm_error(src_name, gt_name)
    #         aed_lst.append(aed)
    #         apd_lst.append(apd)
    #         apd2_lst.append(apd2)
    #     except:
    #         pass
    # print(method_name,numpy.mean(aed_lst), numpy.mean(apd_lst), numpy.mean(apd2_lst))

    # aed_lst = []
    # apd_lst = []
    # apd2_lst = []
    # method_name = 'dpe'
    # src_name_pattern = f"icml_test_data/oneshot_VD/cross/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/cross/results/{method_name}', 'icml_test_data/inputs/VD_cross/video')
    #     try:
    #         aed,apd,apd2 = cal_3dmm_error(src_name, gt_name)
    #         aed_lst.append(aed)
    #         apd_lst.append(apd)
    #         apd2_lst.append(apd2)
    #     except:
    #         pass
    # print(method_name,numpy.mean(aed_lst), numpy.mean(apd_lst), numpy.mean(apd2_lst))

    # aed_lst = []
    # apd_lst = []
    # apd2_lst = []
    # method_name = 'general3d'
    # src_name_pattern = f"icml_test_data/oneshot_VD/cross/results/{method_name}/*.mp4"
    # src_names = glob.glob(src_name_pattern)
    # for src_name in tqdm.tqdm(src_names):
    #     gt_name = src_name.replace(f'icml_test_data/oneshot_VD/cross/results/{method_name}', 'icml_test_data/inputs/VD_cross/video')
    #     try:
    #         aed,apd,apd2 = cal_3dmm_error(src_name, gt_name)
    #         aed_lst.append(aed)
    #         apd_lst.append(apd)
    #         apd2_lst.append(apd2)
    #     except:
    #         pass
    # print(method_name,numpy.mean(aed_lst), numpy.mean(apd_lst), numpy.mean(apd2_lst))
