import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import datasets.transforms_image as T
import matplotlib.pyplot as plt
import torch.utils.data as data
import random
import torch
import pickle

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

class RSVGHRDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=800, transform=None, augment=False,
                 split='train', testmode=False):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode
        
        pth_file = os.path.join(anno_path, f'rsvg_{split}.pth')
        
        if not os.path.exists(pth_file):
            raise FileNotFoundError(f"RSVG-HR {split} dataset file not found: {pth_file}")
        
        data_dict = torch.load(pth_file, map_location='cpu')
        
        if 'images' in data_dict and 'annotations' in data_dict:
            self._load_coco_format(data_dict, images_path)
        else:
            self._load_custom_format(data_dict, images_path)
    
    def _load_coco_format(self, data_dict, images_path):
        """加载COCO格式的数据"""
        images_dict = {img['id']: img for img in data_dict['images']}
        
        for anno in data_dict['annotations']:
            image_id = anno['image_id']
            img_info = images_dict.get(image_id)
            
            if not img_info:
                continue
                
            image_file = os.path.join(images_path, img_info['file_name'])
            
            bbox = anno['bbox']
            box = np.array([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ], dtype=np.float32)
            
            text = anno.get('caption', '')
            
            self.images.append((image_file, box, text))
    
    def _load_custom_format(self, data_dict, images_path):
        """加载自定义格式的数据"""
        for item in data_dict:
            if isinstance(item, tuple) and len(item) >= 3:
                image_file, box, text = item[0], item[1], item[2]
                if not os.path.isabs(image_file):
                    image_file = os.path.join(images_path, image_file)
                self.images.append((image_file, np.array(box, dtype=np.float32), text))
            elif isinstance(item, dict):
                image_file = item.get('image_path', '')
                box = item.get('bbox', [])
                text = item.get('text', '')
                
                if not os.path.isabs(image_file):
                    image_file = os.path.join(images_path, image_file)
                
                if image_file and len(box) == 4:
                    self.images.append((image_file, np.array(box, dtype=np.float32), text))
    
    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        bbox[0] = np.clip(bbox[0], 0, width - 1)
        bbox[1] = np.clip(bbox[1], 0, height - 1)
        bbox[2] = np.clip(bbox[2], 0, width - 1)
        bbox[3] = np.clip(bbox[3], 0, height - 1)
        
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox = np.array([0, 0, width-1, height-1], dtype=np.float32)
        
        target = {
            'boxes': torch.tensor([bbox], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'caption': phrase,
            'orig_size': torch.tensor([height, width], dtype=torch.int64),
            'size': torch.tensor([height, width], dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64)
        }
        
        if self.transform:
            img, target = self.transform(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, target = self.pull_item(idx)
        return img, target

def make_rsvg_hr_transforms(image_set, args):
    """创建RSVG-HR数据集的变换"""
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image_set == 'train':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    
    if image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    
    raise ValueError(f'Unknown image set {image_set}')

def build_rsvg_hr(image_set, args):
    """构建RSVG-HR数据集"""
    root = args.rsvg_hr_path
    anno_path = os.path.join(root, 'Annotations')
    images_path = os.path.join(root, 'images')
    
    if not os.path.exists(images_path):
        images_path = root
    
    transforms = make_rsvg_hr_transforms(image_set, args)
    dataset = RSVGHRDataset(
        images_path=images_path,
        anno_path=anno_path,
        imsize=args.imsize,
        transform=transforms,
        augment=(image_set == 'train'),
        split=image_set,
        testmode=(image_set == 'test')
    )
    return dataset