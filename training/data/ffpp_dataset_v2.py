import torch
import torch.utils.data as data
import os
import random
import json

from PIL import Image, ImageOps
from torchvision import transforms as T
from albumentations import *
import numpy as np

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class FF_Dataset(data.Dataset):
    def __init__(self, list_path, data_path, num_samples=4, img_size=299, interval=4, random_sample=False,
                 augment=False, phase='train'):
        super().__init__()

        self.num_samples = num_samples
        self.data_path = data_path
        self.random_sample = random_sample
        self.interval = interval
        self.phase = phase
        self.augment = augment

        self.aug = Compose([OneOf([
                                Blur(),
                                MotionBlur(),
                                GaussianBlur(),
                                GlassBlur(),
                            ], p=0.2),
                            OneOf([
                                ColorJitter(),
                                RandomBrightnessContrast(),
                                HueSaturationValue(),
                                CLAHE(),
                                RandomGamma(),
                            ], p=0.2),
                            OneOf([
                                ChannelDropout(),
                                ToGray(),
                            ], p=0.2),
                            GaussNoise(p=0.2), CoarseDropout(p=0.2)], p=1.0)


        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            T.Resize((img_size, img_size))
        ])

        self.spatial_size = 10

        if not self.interval % 4 == 0:
            self.interval = self.interval - self.interval % 4

        with open('datasets/ffpp/bboxes_with_partial_bboxes_step_4.txt', 'r') as f:
            self.bboxes = json.load(f)

        self.video_list = []
        self.type_list = []
        self.begin_index_list = []

        with open(list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '')
                line_splited = line.split(' ')
                self.video_list.append(line_splited[0])
                self.type_list.append(line_splited[1])
                self.begin_index_list.append(line_splited[2])
        
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        manipulation = self.type_list[index]
        data_path = os.path.join(self.data_path, manipulation, self.video_list[index])
        
        labels = 0 if 'Original' == manipulation else 1
                        
        frame_list = os.listdir(data_path)
        frame_list.sort(key=lambda x:int(x[:-4]))

        if self.random_sample:
            # select the source video, randomly select the begin index in bboxes except the last 3 frames
            begin_index = random.choice(list(self.bboxes[self.video_list[index][:3]].keys())[:-3])
        else:
            # Sequentially select the begin index in valid bboxes
            begin_index = self.begin_index_list[index]

        video_bboxes = self.bboxes[self.video_list[index][:3]]
        # bbox, l_eye_bbox, r_eye_bbox, l_cheek_bbox, r_cheek_bbox, nose_bbox, mouth_bbox = self.bboxes[self.video_list[index][:3]][begin_index]

        begin_index = int(begin_index)
        bbox_list = []

        # select num_samples frame with interval to form list
        if self.num_samples * self.interval + begin_index < len(frame_list):
            clip_frames = frame_list[begin_index:(begin_index + self.num_samples * self.interval):self.interval]
            for i in range(len(clip_frames)):
                idx = begin_index + i * self.interval
                if not str(idx) in video_bboxes:
                    while not str(idx) in video_bboxes:
                        idx -= 4
                        if idx < 0:
                            idx = begin_index
                bbox_list.append(video_bboxes[str(idx)])
        else:
            clip_frames = frame_list[(len(frame_list) - self.num_samples * self.interval)::self.interval]
            for i in range(len(clip_frames)):
                idx = len(frame_list) - (len(frame_list) % 4) - self.num_samples * self.interval + i * self.interval
                if not str(idx) in video_bboxes:
                    while not str(idx) in video_bboxes:
                        idx -= 4
                        if idx < 0:
                            idx = begin_index
                bbox_list.append(video_bboxes[str(idx)])
        face_seq = []

        # crop the face and execute data augmentation
        for idx, frame in enumerate(clip_frames):
            frame_img = Image.open(os.path.join(data_path, frame))
            ori_img = Image.open(os.path.join(os.path.join(self.data_path, 'Original', self.video_list[index][:3]), frame))
            
            if frame_img.size != ori_img.size:
                pad = ori_img.size[0] - frame_img.size[0]
                left_pad = (pad+1)//2
                right_pad = pad - left_pad
                frame_img = ImageOps.expand(frame_img, (left_pad, 0, right_pad, 0), 'red')

            frame_img = frame_img.crop(bbox_list[idx][0])
            if self.phase == 'train' and self.augment:
                frame_img = self.aug(image=np.array(frame_img))['image']
            face_img = self.to_tensor(frame_img)
            del frame_img
            del ori_img
                
            face_seq.append(face_img)

        # stack the face sequence and generate the rois
        face_seq = torch.stack(face_seq, dim=0)

        rois = []
        for bbox in bbox_list:
            rois.append(bbox[1:]+ [[0.05, 0.05, 0.95, 0.95]])
        with open('check.json', 'w') as f:
            json.dump(rois, f)
        rois = torch.tensor(rois) * self.spatial_size

        # rois = torch.tensor([l_eye_bbox, r_eye_bbox, l_cheek_bbox, r_cheek_bbox, nose_bbox, mouth_bbox, [0.05, 0.05, 0.95, 0.95]]) * self.spatial_size

        # randomly reverse the sequence or horizontal flip the sequence
        if self.random_sample:
            if random.randint(1, 10) > 5:
                face_seq = torch.flip(face_seq, [0])
                rois = torch.flip(rois, [0])
            if random.randint(1, 10) > 5:
                face_seq = torch.flip(face_seq, [3])
                rois[:, 0], rois[:, 1] = rois[:, 1], rois[:, 0] # swap eyes
                rois[:, 2], rois[:, 3] = rois[:, 3], rois[:, 2] # swap cheeks
                rois[:, :, 0], rois[:, :, 2] = 1 - rois[:, :, 2], 1 - rois[:, :, 0]
                # for bbox in [r_eye_bbox, l_eye_bbox, r_cheek_bbox, l_cheek_bbox, nose_bbox, mouth_bbox,
                #              [0.05, 0.05, 0.95, 0.95]]:
                #     x1, y1, x2, y2 = bbox
                #     rois.append([1 - x2, y1, 1 - x1, y2])
                # rois = torch.tensor(rois) * self.spatial_size

        return face_seq, labels, rois