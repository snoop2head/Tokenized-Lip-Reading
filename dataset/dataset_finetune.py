import os
import glob
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from PIL import Image
from easydict import EasyDict
import math
from typing import Union, Tuple
from utils.face_mesh import FACE_MESH_USE
from .custom_transform import *

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

jpeg = TurboJPEG()


class RDropDataset_finetune(Dataset):
    def __init__(
      self, 
      root_dir, 
      split="train",
      label_dir= "/home/ubuntu/LIP/LRW2/_clones/learn-an-effective-lip-reading-model-without-pains/label_sorted.txt", 
      num_classes=500,
      num_video_patch=29,
      num_image_patch=29,
      special_token_length=2,
      transforms= None
    ):
    
        with open(label_dir) as myfile:
            labels = myfile.read().splitlines()
            labels = labels[:num_classes]
        
        labels = sorted(labels)
        
        # designate labels and spectrogram images
        label_map = []
        image_file_names = []
        for i, label in enumerate(labels):
            label_data = glob.glob(os.path.join(root_dir, label, split, "*.npy"))
            image_file_names.extend(label_data)
            label_map.extend([i] * len(label_data))

        # designate cropped grayscaled video
        video_file_names = []
        for (i, label) in enumerate(labels):
            files = glob.glob(os.path.join("/home/ubuntu/LIP/LRW2/_clones/learn-an-effective-lip-reading-model-without-pains/lrw_cropped_npy_gray_pkl_jpeg", label, split, "*.pkl"))
            files = sorted(files)
            video_file_names.extend(files)
        
        self.split = split
        self.transforms = transforms
        self.image_file_names = image_file_names
        self.video_file_names = video_file_names
        self.labels = label_map
        self.video_movement_limit = 8
        self.num_image_patch = num_image_patch
        self.num_video_patch = num_video_patch
        self.encoder_special_tokens = special_token_length
        self.seq_length = self.num_image_patch+self.num_video_patch+self.encoder_special_tokens
    
    def __getitem__(self, index):
        # get image-like spectrogram
        npy_path = self.image_file_names[index]
        image = np.load(npy_path)

        # selecting FACE_MESH_USE coordinates
        image = image[:, FACE_MESH_USE, :]

        # VIDEO-WISE Random Addition
        if self.split == "train":
            random_int_x = random.randint(-self.video_movement_limit, self.video_movement_limit)
            random_int_y = random.randint(-self.video_movement_limit, self.video_movement_limit)
            random_float_x = random_int_x / 256
            random_float_y = random_int_y / 256
            image[:, :, 0] = image[:, :, 0] + random_float_x # x-channel as 0
            image[:, :, 1] = image[:, :, 1] + random_float_y # y-channel as 1
        else:
            pass
        
        # flip on x-axis
        flipped_meshes = image.copy()
        flipped_meshes[:, :, 0] = 1.0 - flipped_meshes[:, :, 0]

        # get video data
        tensor = torch.load(self.video_file_names[index])
        inputs = tensor.get("video")
        inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:, :, :, 0]

        if self.split == "train":
            batch_img = RandomCrop(inputs, (96, 96), random_int_x)
        elif self.split == "val" or self.split == "test":
            batch_img = CenterCrop(inputs, (96, 96))
        video = torch.FloatTensor(batch_img[:, np.newaxis, ...])
        video = video.transpose(0, 1)
        
        # get attention mask length of the num_video + num_image patches
        attention_mask = torch.ones(self.seq_length).bool()
        
        # get label
        label = self.labels[index]
        label = torch.tensor(label)
        
        if self.transforms:
            image_transform = self.transform(image)
            flipped_meshes_transform = self.transform(flipped_meshes)
            return image_transform.float(), flipped_meshes_transform.float(), video, video.flip(3), attention_mask.bool(), label
        else:
            return image.float(), flipped_meshes.float(), video, video.flip(3), attention_mask.bool(), label

    def __len__(self):
        assert len(self.image_file_names) == len(self.video_file_names)
        return len(self.image_file_names)
    
    def set_transform(self, transform):
        """ set image transforms """
        self.transform = transform