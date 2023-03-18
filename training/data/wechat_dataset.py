import torch
import torch.utils.data as data
import os
import json

from PIL import Image
from torchvision import transforms as T

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class WX_Dataset(data.Dataset):
    def __init__(self, img_size=299, interval=2, random_sample=False):
        super().__init__()

        self.random_sample = random_sample
        self.interval = interval
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            T.Resize((img_size, img_size))
        ])
        
        with open('datasets/putin/bboxes.txt', 'r') as f:
            self.bboxes = json.load(f)

        self.video_list = []
        self.type_list = []
        self.begin_index_list = []
        self.frame_list = os.listdir('wx_frames/')

    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        bbox, l_eye_bbox, r_eye_bbox, l_cheek_bbox, r_cheek_bbox, nose_bbox, mouth_bbox = self.bboxes['putin'][str(index*20)]
        clip_frames = self.frame_list[index*20::self.interval][:8]
            
        face_seq = []
        
        for frame in clip_frames:
            frame_img = Image.open(os.path.join('wx_frames', frame))

            frame_img = frame_img.crop(bbox)

            face_img = self.to_tensor(frame_img)
            del frame_img
                
            face_seq.append(face_img)
                
        face_seq = torch.stack(face_seq, dim=0)
        rois = torch.tensor([l_eye_bbox, r_eye_bbox, l_cheek_bbox, r_cheek_bbox, nose_bbox, mouth_bbox, [0.05, 0.05, 0.95, 0.95]])*10

        return face_seq, 1, rois