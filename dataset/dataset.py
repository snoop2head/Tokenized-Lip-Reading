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


class RDropDataset(Dataset):
    def __init__(
      self, 
      split="train",
      label_dir= "/home/ubuntu/LIP/LRW2/_clones/learn-an-effective-lip-reading-model-without-pains/label_sorted.txt", 
      num_classes=500,
      num_video_patch=29,
      num_image_patch=29,
      special_token_length = 2,
      transforms= None
    ):
    
        with open(label_dir) as myfile:
            labels = myfile.read().splitlines()
            labels = labels[:num_classes]
        labels = sorted(labels)
        
        # designate cropped grayscaled video
        label_map = []
        video_file_names = []
        for (i, label) in enumerate(labels):
            video_files = glob.glob(os.path.join("/home/ubuntu/LIP/LRW2/_clones/learn-an-effective-lip-reading-model-without-pains/lrw_cropped_npy_gray_pkl_jpeg", label, split, "*.pkl"))
            
            video_files = sorted(video_files)
            video_file_names.extend(video_files)
            label_map.extend([i] * len(video_files))
        
        self.split = split
        self.transforms = transforms
        self.video_file_names = video_file_names
        self.labels = label_map
        self.video_movement_limit = 8
        self.num_image_patch = num_image_patch
        self.num_video_patch = num_video_patch
        self.bos_token = processor.tokenizer.bos_token_id
        self.eos_token = processor.tokenizer.eos_token_id
        self.encoder_special_tokens = special_token_length
        self.seq_length = self.num_image_patch+self.num_video_patch+self.encoder_special_tokens
    
    def __getitem__(self, index):
        # get image-like spectrogram
        video_path = self.video_file_names[index]
        video_file_name = os.path.basename(video_path)
        label_name = video_file_name.split("_")[0]
        npy_path = os.path.join(
            "/home/ubuntu/LIP/LRW2/lipread_mediapipe",
            label_name,
            self.split,
            video_file_name.replace(".pkl", ".npy")
        )
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
        tensor = torch.load(video_path)
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
        
        # get attention mask
        attention_mask = torch.randint(0, 5, (self.seq_length,)).bool()
        attention_mask[0] = True # cls
        attention_mask[self.num_image_patch+1] = True # sep
        
        # load_input_ids
        input_ids = self.load_input_ids(npy_path)
        # insert self.bos_token to the beginning
        input_ids = np.concatenate([[self.bos_token], input_ids, [self.eos_token]]) # np.concatenate is faster than np.insert or np.append
        input_ids = torch.tensor(input_ids)
        
        if self.transforms:
            image_transform = self.transform(image)
            flipped_meshes_transform = self.transform(flipped_meshes)
            return image_transform.float(), flipped_meshes_transform.float(), video, video.flip(3), attention_mask.bool(), input_ids
        else:
            return image.float(), flipped_meshes.float(), video, video.flip(3), attention_mask.bool(), input_ids

    def __len__(self):
        return len(self.video_file_names)
    
    def set_transform(self, transform):
        self.transform = transform
    
    def load_input_ids(self, file):
        basename = os.path.basename(file)
        audio_dir = "/home/ubuntu/LIP/LRW2/lipread_input_id"
        label = basename.split("_")[0]
        file_dir = os.path.join(audio_dir, label, self.split, basename)
        return np.load(file_dir)