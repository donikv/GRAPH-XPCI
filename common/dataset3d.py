import os

import cv2
import numpy as np
import torch
import tqdm
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import *

from common.utils import load_image_greyscale, load_image_16bit, biopsy_names
from common.preprocessing import *
from common.visualize_utilities import show, visualize_set_program
from multiprocessing import Pool

def load_volume_16bit(paths, num_workers=4):
    with Pool(num_workers) as p:
        images = p.map(load_image_16bit, paths)
    return images

def load_volume_greyscale(paths, num_workers=4):
    with Pool(num_workers) as p:
        images = p.map(load_image_greyscale, paths)
    return images

def load3d_volume(data, imaging, size=2560, start_index = 0, end_index=-1, num_workers=-1, verbose=False):
    # Find all the images that belong to the same biopsy
    images = list(filter(lambda x: x.find(imaging) != -1, data))
    if end_index == -1:
        end_index = len(images)
    start_index, end_index = max(0, start_index), min(len(images), end_index)
    volume = np.zeros((end_index-start_index, size, size), dtype=np.uint8)
    images = images[start_index:end_index]
    imgs = load_volume_greyscale(images, num_workers)
    for i, img in (enumerate(imgs) if not verbose else tqdm(enumerate(imgs), total=len(imgs), desc=f"Loading {imaging}")):
        if img is not None:
            volume[i] = img
    # Load all the images and stack them into a 3d volume
    return volume

class Biopsies3DVolumeDataset(Dataset):
    def __init__(self, paths, root, image_size, patch_size, transform, target_transform, shared_transforms, preload, cache, num_workers=-1):
        self.paths = paths
        self.imagings = biopsy_names(paths, root)
        self.imagings = sorted(self.imagings)
        self.sorted_data = sorted(paths)
        self.patch_size = patch_size
        self.image_size = image_size

        self.images = {}
        self.targets = {}
        self.preload = preload
        self.cache = cache
        if preload:
            self.__preload__()

        self.transform = transform
        self.target_transform = target_transform
        self.shared_transforms = shared_transforms
        self.num_workers = num_workers
        self.len, self.index_to_imaging_and_coordinates = self.calculate_lenght()
        self.current_volume_info = (None, None, None)

    
    def calculate_lenght(self):
        l = 0
        index_to_imaging_and_coordinates = {}
        for imaging in self.imagings:
            imaging_length = len(list(filter(lambda x: x.find(imaging) != -1, self.paths)))
            num_patch_slices = imaging_length // self.patch_size
            num_patches_x = self.image_size // self.patch_size
            num_patches_per_imaging = num_patch_slices * num_patches_x * num_patches_x
            for i in range(num_patches_per_imaging):
                slice_idx = i // (num_patches_x ** 2)
                y = (i // num_patches_x) % num_patches_x
                x = i % num_patches_x
                index_to_imaging_and_coordinates[l] = (imaging, slice_idx, y, x)
                l += 1
        return l, index_to_imaging_and_coordinates
    

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        imaging, slice_idx, y, x = self.index_to_imaging_and_coordinates[index]
        start_index = slice_idx * self.patch_size
        end_index = start_index + self.patch_size
        current_imaging, current_slice_index, current_volume = self.current_volume_info
        if current_imaging != imaging or current_slice_index != slice_idx:
            current_volume = load3d_volume(self.sorted_data, imaging, self.image_size, start_index, end_index, self.num_workers)
            current_imaging = imaging
            current_slice_index = slice_idx
            self.current_volume_info = (current_imaging, current_slice_index, current_volume)
        volume = current_volume[:, y*self.patch_size:(y+1)*self.patch_size, x*self.patch_size:(x+1)*self.patch_size]
        if self.transform:
            volume = self.transform(volume)
        return volume, imaging

if __name__ == "__main__":
    import time
    l = "./data/biopsies_new.txt"
    path = '/home/donik/datasets/XPCI/dataset_biopsies_new/'
    # path = '/run/media/donik/Disk/syncthing/datasets/XPCI/dataset_biopsies/'
    data = [path + x[2:] for x in np.loadtxt(l, dtype=str)]
    # data = list(filter(lambda x: x.find("38_201-01") != -1, data))
    print(len(data))
    imagings = biopsy_names(data, path)
    print(imagings)
    dataset = Biopsies3DVolumeDataset(data, path, 2560, 256, None, None, None, False, False, 4)
    print(len(dataset))
    s = time.time()
    vlm, a = dataset[0]
    e = time.time()
    print(e-s)

    s = time.time()
    vlm, a = dataset[1]
    e = time.time()
    print(e-s)
