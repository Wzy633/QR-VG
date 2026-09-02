import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import util
from util.transforms import letterbox
import datasets.transforms_image as T
import matplotlib.pyplot as plt
import torch.utils.data as data
import random
import torch


class RSVGDataset(data.Dataset):
    def __init__(self, images_path, imsize=1024, transform= None, augment= False,
                 split='train', testmode=False, split_file_dir=None):
        self.images = []
        self.images_path = images_path
        self.imsize = imsize
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode

        if split_file_dir is None:
            if images_path:
                split_file_dir = os.path.dirname(os.path.dirname(images_path))
            else:
                split_file_dir = '../Dataset/rsvg_mm'
        
        split_file = os.path.join(split_file_dir, 'rsvg_mm_' + split + '_v2.txt')
        if not os.path.exists(split_file):
            split_file = '../Dataset/rsvg_mm/rsvg_mm_' + split + '_v2.txt'
        file = open(split_file, "r").readlines()
        Index = [index.strip('\n') for index in file]
        for anno in Index:
            anno_list = anno.split(',')
            img_name = anno_list[0]
            xmin_gt, ymin_gt, xmax_gt, ymax_gt = float(anno_list[1]), float(anno_list[2]), float(anno_list[3]), float(anno_list[4])
            text = anno_list[-1]
            image_path = images_path + '/' + img_name
            box = np.array([xmin_gt, ymin_gt, xmax_gt, ymax_gt], dtype=np.float32)
            self.images.append((image_path, box, text))

    def pull_item(self, idx):
        img_path, bbox, phrase = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        return img, phrase, bbox, img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_path  = self.pull_item(idx)
        caption = " ".join(phrase.lower().split())

        w, h = img.size
        mask = np.zeros_like(img)

        if self.testmode:
            img = np.array(img)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
        bbox = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]]).to(torch.float)
        bbox = bbox.unsqueeze(0)

        target = {}
        target["dataset_name"] = "RSVG_MM"
        target["boxes"] = bbox
        target["labels"] = torch.tensor([1])
        if caption is not None:
            target["caption"] = caption
        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.testmode:
            return img.unsqueeze(0), target, dw, dh, img_path, ratio
        else:
            return img.unsqueeze(0), target

def make_coco_transforms(image_set, cautious):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [768, 840, 920, 1024]

    max_size = 1024
    if image_set == "train":
        return T.Compose(
            [T.RandomResize(scales, max_size=max_size),
            normalize]
        )

    else:
        return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])


    raise ValueError(f"unknown {image_set}")

from pathlib import Path

def build(image_set, args):
    root = Path(args.rsvg_mm_path)
    assert root.exists(), f'provided rsvg_mm path {root} does not exist'
    input_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    img_folder = root / "images"
    dataset = RSVGDataset(str(img_folder), transform=input_transform, split=image_set, testmode=(image_set == 'test'), split_file_dir=str(root))
    return dataset

if __name__ == '__main__':
    input_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    img_folder = '../Dataset/rsvg_mm/images'
    image_set = 'train'
    dataset = RSVGDataset(img_folder, transform=input_transform, split=image_set, testmode=(image_set=='test'))
    sample_num = dataset.__len__()
    sample_num = dataset.__getitem__(0)
    print(sample_num)
