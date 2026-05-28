import cv2
import dlib
import numpy as np
import os
import tqdm

def read_video_to_rgb_frames(vid_name):
    frames = []
    cap = cv2.VideoCapture(vid_name)
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    frames = np.stack(frames)
    return frames

class LandmarkDetector:
    def __init__(self, lm_model_path='utils/metrics/shape_predictor_68_face_landmarks.dat'):
        self.lm_model_path = lm_model_path
        self.bbox_detector = dlib.get_frontal_face_detector()
        # lm detector, download the model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        if not os.path.exists(lm_model_path):
            os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            os.system("bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
            try:
                os.makedirs(os.path.dirname(lm_model_path), exist_ok=True)
                os.system(f"mv shape_predictor_68_face_landmarks.dat {lm_model_path}")
            except: pass
        self.lm_detector = dlib.shape_predictor(lm_model_path)
    
    def detect_lip_lm_from_image_name(self, image_name):
        img = cv2.imread(image_name)
        img = cv2.resize(img, (512,512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.bbox_detector(gray, 1)
        assert len(dets) == 1, "detect more than one face in the image!"
        face_bbox = dets[0]
        lm68_points = self.lm_detector(img, face_bbox)
        lip_points = lm68_points.parts()[48:68]
        lip_pos = np.array([(pt.x, pt.y) for pt in lip_points]) # array, uint8, [20,2]
        return lip_pos

    def detect_lip_lm_from_rgb_img(self, img):
        img = cv2.resize(img, (512,512))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = self.bbox_detector(gray, 1)
        assert len(dets) == 1, "detect more than one face in the image!"
        face_bbox = dets[0]
        lm68_points = self.lm_detector(img, face_bbox)
        lip_points = lm68_points.parts()[48:68]
        lip_pos = np.array([(pt.x, pt.y) for pt in lip_points]) # array, uint8, [20,2]
        return lip_pos

    def detect_lm68_from_image_name(self, image_name):
        img = cv2.imread(image_name)
        img = cv2.resize(img, (512,512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.bbox_detector(gray, 1)
        assert len(dets) == 1, "detect more than one face in the image!"
        face_bbox = dets[0]
        lm68_points = self.lm_detector(img, face_bbox)
        points = lm68_points.parts()
        pos = np.array([(pt.x, pt.y) for pt in points]) # array, uint8, [20,2]
        return pos

    def detect_lm68_from_rgb_img(self, img):
        img = cv2.resize(img, (512,512))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = self.bbox_detector(gray, 1)
        assert len(dets) == 1, "detect more than one face in the image!"
        face_bbox = dets[0]
        lm68_points = self.lm_detector(img, face_bbox)
        points = lm68_points.parts()
        pos = np.array([(pt.x, pt.y) for pt in points]) # array, uint8, [20,2]
        return pos

class LMDCalculator:
    def __init__(self):
        self.lm_detector = LandmarkDetector()
    
    def cal_akd(self, vid_name1, vid_name2):
        imgs1 = read_video_to_rgb_frames(vid_name1)
        imgs2 = read_video_to_rgb_frames(vid_name2)
        lms_lst1, lms_lst2 = [], []

        for i in tqdm.trange(len(imgs1)):
            lm1 = self.lm_detector.detect_lm68_from_rgb_img(imgs1[i])
            lm2 = self.lm_detector.detect_lm68_from_rgb_img(imgs2[i])
            lms_lst1.append(lm1)            
            lms_lst2.append(lm2)   
        lms1 = np.stack(lms_lst1, axis=0)        
        lms2 = np.stack(lms_lst2, axis=0)   
        return np.abs((lms1 - lms2)).mean()

    def cal_lmd(self, vid_name1, vid_name2):
        imgs1 = read_video_to_rgb_frames(vid_name1)
        imgs2 = read_video_to_rgb_frames(vid_name2)
        lms_lst1, lms_lst2 = [], []
        lip_lms_lst1, lip_lms_lst2 = [], []

        for i in tqdm.trange(min(len(imgs1), len(imgs2)), desc="extract 2d lms..."):
            lm1 = self.lm_detector.detect_lm68_from_rgb_img(imgs1[i])
            lip_lm1 = lm1[48:68]
            lm2 = self.lm_detector.detect_lm68_from_rgb_img(imgs2[i])
            lip_lm2 = lm2[48:68]
            lms_lst1.append(lm1)            
            lip_lms_lst1.append(lip_lm1)            
            lms_lst2.append(lm2)   
            lip_lms_lst2.append(lip_lm2)            
        lms1 = np.stack(lms_lst1, axis=0)
        lip_lms1 = np.stack(lip_lms_lst1, axis=0)        
        lms2 = np.stack(lms_lst2, axis=0)  
        lip_lms2 = np.stack(lip_lms_lst2, axis=0)  

        lms1 = lms1 - lms1.mean(axis=0)     
        lms2 = lms2 - lms2.mean(axis=0)     
        lip_lms1 = lip_lms1 - lip_lms1.mean(axis=0)     
        lip_lms2 = lip_lms2 - lip_lms2.mean(axis=0)     

        lip_lmd = np.abs((lip_lms1 - lip_lms2)).mean()
        lmd = np.abs((lms1 - lms2)).mean()
        vel_lms1 = lms1[1:] - lms1[:-1]
        vel_lms2 = lms2[1:] - lms2[:-1]
        lmd_vel = np.abs((vel_lms1 - vel_lms2)).mean()
        vel_lip_lms1 = lip_lms1[1:] - lip_lms1[:-1]
        vel_lip_lms2 = lip_lms2[1:] - lip_lms2[:-1]
        lip_lmd_vel = np.abs((vel_lip_lms1 - vel_lip_lms2)).mean()
        return lmd, lmd_vel, lip_lmd, lip_lmd_vel

if __name__ == '__main__':
    lmd_calculator = LMDCalculator()

    # vid_name1 = "ijcai24_test/inputs/video/3.mp4"
    # vid_name2 = "ijcai24_test_static/results/wav2lip/3.mp4"
    # lmd = lmd_calculator.cal_lmd(vid_name1, vid_name2)
    # print("Wav2Lip LMD", lmd)

    # vid_name1 = "ijcai24_test/inputs/video/3.mp4"
    # vid_name2 = "ijcai24_test/results/makeittalk/3.mp4"
    # lmd = lmd_calculator.cal_lmd(vid_name1, vid_name2)
    # print("MakeItTalk LMD", lmd)

    # vid_name1 = "ijcai24_test/inputs/pcavs_cropped/3.mp4"
    # vid_name2 = "ijcai24_test/results/pcavs/3.mp4"
    # lmd = lmd_calculator.cal_lmd(vid_name1, vid_name2)
    # print("PCAVS LMD", lmd)
    import glob, tqdm, numpy

    akd_lst = []
    method_name = 'fsfacev2v'
    src_name_pattern = f"icml_test_data/fewshot/cross/{method_name}/*.mp4"
    src_names = glob.glob(src_name_pattern)
    src_names = src_names[-3:]
    for src_name in tqdm.tqdm(src_names):
        gt_name = src_name.replace(f'icml_test_data/fewshot/cross/{method_name}', 'icml_test_data/inputs/VD_cross/video')
        try:
            akd = lmd_calculator.cal_akd(src_name, gt_name)
            akd_lst.append(akd)
        except:
            pass
    print(method_name,akd_lst)
    print(method_name,numpy.mean(akd_lst))
