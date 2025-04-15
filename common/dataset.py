import os

import cv2
import numpy as np
import torch
import tqdm
import pandas as pd
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets import *

from common.utils import load_image_greyscale, load_image_16bit, biopsies_target, BIOPSY_TYPE, SPLITS
from common.preprocessing import *
from common.visualize_utilities import show, visualize_set_program
import common.datasets.single_image_classification
import common.datasets.anomaly
from multiprocessing import Pool

visualize_set_program(os.path.basename(__file__))

def add_dataset_args(parser):
    # parser.add_argument("--path", type=str, default="./data", help="path to dataset root, default: ./data")
    parser.add_argument("--list", type=str, default="./data/biopsies_list.txt", help="path to dataset file list, default: ./data/biopsies_list.txt")
    parser.add_argument("--size", type=int, default=1024, help="size of the cropped images, default: 256")
    parser.add_argument("--dataset", type=str, default='biopsies', help="which dataset to use, default: biopsies")
    parser.add_argument("--no-split", action='store_true', default=False, help="use train/test split, default: False")
    parser.add_argument("--split-holdout", action='store_true', default=False, help="use holdout tissues for test, default: False, (DEPRECATED, use --split)") #LEGACY
    parser.add_argument("--random-artifacts", action='store_true', default=False, help="randomly remove imaging artifacts, default: False -> removes artifacts with fixed parameters")
    parser.add_argument("--balance-dataset", action='store_true', default=False, help="use weighted sampling, default: False")
    parser.add_argument("--normalize", action='store_true', default=False, help="normalize images, default: False")
    parser.add_argument("--histogram-reference", default=None, help="reference image for histogram equalization, default: ./data/07_4001_3R1212.rec.8bit.tif (archive image)")
    parser.add_argument("--contrast", action='store_true', default=False, help="use contrast augmentation, default: False")
    parser.add_argument("--split", type=str, default="holdout_legacy", choices=SPLITS, help="split type, default: holdout_legacy")
    parser.add_argument("--tissue-type", type=str, default="archive", choices=BIOPSY_TYPE, help="tissue type, default: archive")
    parser.add_argument("--histogram-equalization", action='store_true', default=False, help="use histogram equalization, default: False")
    parser.add_argument("--crop-size", type=int, default=None, help="size of the cropped images, default: None")
    parser.add_argument("--rgb", action='store_true', default=False, help="convert to RGB images, default: False")
    parser.add_argument("--rotation", type=int, default=0, help="rotation angle, default: 0")
    parser.add_argument("--perspective", type=float, default=0, help="perspective distortion scale, default: 0")
    parser.add_argument("--random-scale-min", type=float, default=None, help="random scale lower bound factor, default: None")

    group = parser.add_argument_group("Csv dataset arguments")
    group.add_argument("--train-csv", nargs='+', default=["./csv/combined/train.csv"], help="path to train csv files, default: ./csv/combined/train.csv")
    group.add_argument("--test-csv", nargs='+', default=["./csv/combined/test.csv"], help="path to test csv files, default: ./csv/combined/test.csv")
    parser.add_argument("--path", nargs='+', default=["./data"], help="path to datasets root, default: ./data")
    parser.add_argument("--test-path", nargs='*', default=None, help="path to datasets root, default: ./data")
    group.add_argument("--balance-csv-combined", action='store_true', default=False, help="balance combined csv dataset, default: False")
    #parser.add_argument("--artifacts", action='store_true', default=False, help="remove imaging artifacts, default: False")
    #group.add_argument("--train-csv", type=str, default="./csv/combined/train.csv", help="path to train csv file, default: ./csv/combined/train.csv")
    #group.add_argument("--test-csv", type=str, default="./csv/combined/test.csv", help="path to test csv file, default: ./csv/combined/test.csv")

def parse_dataset_args(args):
    if args.test_path is None:
        args.test_path = args.path

    dataset = BiopsiesDataset
    if args.dataset == "biopsies":
        train, valid, test, classes = common.datasets.single_image_classification.make_dataset(args, dataset)
    elif args.dataset == "patches":
        train, valid, test, classes = common.datasets.single_image_classification.make_dataset(args, BiopsiesPatchDataset)
    elif args.dataset == "two_class":
        train, valid, test, classes = common.datasets.single_image_classification.make_dataset(args, BiopsiesTwoClassDataset)
    elif args.dataset == "csv":
        train, valid, test, classes = common.datasets.single_image_classification.make_csv_dataset(args, BiopsiesCSVDataset)
    elif args.dataset == "anomalies":
        train, valid, test, classes = common.datasets.anomaly.make_dataset(args, dataset)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return train, valid, test, classes
    
class ImageDataset(Dataset):
    def __init__(self, paths, transform, target_transform, shared_transforms, preload, cache):
        self.paths = paths
        self.images = {}
        self.targets = {}
        self.preload = preload
        self.cache = cache
        if preload:
            self.__preload__()

        self.transform = transform
        self.target_transform = target_transform
        self.shared_transforms = shared_transforms

    def load_image(self, path):
        pass

    def load_target(self, path):
        pass

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        if self.preload or path in self.images.keys():
            image = self.images[path]
            label = self.targets[path]
        else:
            image = self.load_image(path)
            label = self.load_target(path)
            if self.cache:
                self.images[path] = image
                self.targets[path] = label

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.shared_transforms:
            image, label = self.shared_transforms(image, label)

        return image, label

    def __preload__(self):
        self.images = {x:self.load_image(x) for x in tqdm.tqdm(self.paths, total=len(self.paths))}
        self.targets = {x:self.load_target(x) for x in tqdm.tqdm(self.paths, total=len(self.paths))}

class BiopsiesCSVDataset(ImageDataset):

    def __init__(self, df, transform, target_transform, shared_transforms, preload, cache, two_class=False):
        self.data = df
        # self.root_dir = root_dir
        paths = [os.path.join(rd, x) for x, rd in zip(self.data.image, self.data.root_dir)]
        if not two_class:
            self.ys = {path: target for (path, target) in zip(paths, self.data.target)}
        else:
            self.ys = {path: 0 if target < 2 else 1 for (path, target) in zip(paths, self.data.target)}
        super().__init__(paths, transform, target_transform, shared_transforms, preload, cache)

    def load_image(self, path):
        image = load_image_greyscale(path=path)
        image = image / image.max()
        image = image.astype(np.float32)
        return image

    def load_target(self, path):
        return self.ys[path]

class TestBiopsiesCSVDataset(ImageDataset):

    def __init__(self, df, root_dir, transform, preload, cache, two_class=False):
        self.data = df
        self.root_dir = root_dir
        paths = [os.path.join(self.root_dir, x) for x in self.data.image]
        if not two_class:
            self.ys = {path: (target, image) for (path, target, image) in zip(paths, self.data.target, self.data.image)}
        else:
            self.ys = {path: (0 if target < 2 else 1, image) for (path, target, image) in zip(paths, self.data.target, self.data.image)}
        super().__init__(paths, transform, None, None, preload, cache)

    def load_image(self, path):
        image = load_image_greyscale(path=path)
        image = image / image.max()
        image = image.astype(np.float32)
        return image

    def load_target(self, path):
        return self.ys[path]


class TauDataset(ImageDataset):

    def load_image(self, path):
        path = path + '.tiff'
        image = load_image_16bit(path)
        image = image / 2 ** 16 # type: ignore
        return image

    def load_target(self, path):
        path = path + '.wp'
        target = np.loadtxt(path,delimiter=',')
        target = target / np.linalg.norm(target, ord=2)

        return target
    

class BiopsiesDataset(ImageDataset):

    CLASSES = ["0R", "1R", "2R", "3R"]

    def load_image(self, path):
        image = load_image_greyscale(path=path)
        image = image / image.max()
        image = image.astype(np.float32)
        return image

    def load_target(self, path):
        return BiopsiesDataset.__load_target__(path, self.CLASSES)

    @staticmethod
    def __load_target__(path, cls=CLASSES):
        return biopsies_target(path, cls)

class BiopsiesPatchDataset(ImageDataset):

    CLASSES = ["clean", "rejection"]

    def load_image(self, path):
        image = load_image_greyscale(path=path)
        image = image / image.max()
        image = image.astype(np.float32)
        return image

    def load_target(self, path):
        return BiopsiesDataset.__load_target__(path, self.CLASSES)

    @staticmethod
    def __load_target__(path, cls=CLASSES):
        return biopsies_target(path, cls)

class BiopsiesTwoClassDataset(ImageDataset):

    CLASSES = [[0, 1],[2, 3]]

    def load_image(self, path):
        image = load_image_greyscale(path=path)
        image = image / image.max()
        image = image.astype(np.float32)
        return image

    def load_target(self, path):
        return BiopsiesTwoClassDataset.__load_target__(path, self.CLASSES)
    
    @staticmethod
    def __load_target__(path, cls=CLASSES):
        target_all = biopsies_target(path, BiopsiesDataset.CLASSES)
        for i, c in enumerate(cls):
            if target_all in c:
                return i
        raise ValueError(f"Unknown class {target_all}")

# class BiopsiesPatchesDataset(ImageDataset):

#     def __init__(self, paths, transform, target_transform, shared_transforms, preload, cache):
#         super().__init__(paths, transform, target_transform, shared_transforms, preload, cache)

#         self.current_patches = None
#         self.current_labels = None


def simple_index(ds:Dataset, input_type:str, idx:int):
    if input_type == ImageTripletDataset.POSITIVE:
        return (idx + 1) % ds.__len__()
    else:
        return (idx + 2) % ds.__len__()

class ImageTripletDataset(Dataset):

    POSITIVE = "positive"
    NEGATIVE = "negative"

    def __init__(self, base_image_dataset: ImageDataset, positive_transforms=None, negative_transforms=None, index_fn=simple_index):
        super().__init__()
        self.base = base_image_dataset
        self.__get_idx = index_fn
        self.positive_transforms = positive_transforms
        self.negative_transforms = negative_transforms


    def __len__(self):
        return len(self.base.paths)

    def __getitem__(self, idx):
        anchor, anchor_label = self.base.__getitem__(idx)
        positive, positive_label = self.base.__getitem__(self.__get_idx(self,ImageTripletDataset.POSITIVE,idx))
        negative, negative_label = self.base.__getitem__(self.__get_idx(self,ImageTripletDataset.NEGATIVE,idx))

        if self.positive_transforms:
            positive, positive_label = self.positive_transforms(positive, positive_label)
        if self.negative_transforms:
            negative, negative_label = self.negative_transforms(negative, negative_label)
        
        image = torch.stack([anchor, positive, negative], dim=0)
        label = torch.stack([anchor_label, positive_label, negative_label], dim=0)

        return image, label



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split as split
    import matplotlib.pyplot as plt
    from common.utils import biopsies_target, NEW_DATASET_REJECTIONS
    from collections import Counter

    path = '/home/donik/datasets/XPCI/dataset_biopsies/'
    list = './data/biopsies_list.txt'

    data = [path + x[2:] for x in np.loadtxt(list, dtype=str)]
    targets = [biopsies_target(x) for x in data]
    z = Counter(targets)
    print(z)

    # path_small = '/run/media/donik/Disk/XPCI/dataset_biopsies_small'

    # data_small = [path_small + '/' + x[2:] for x in np.loadtxt(list, dtype=str)]

    # for path, path_small in zip(data, data_small):
    #     image = load_image_greyscale(path)
    #     image_small = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_NEAREST)
    #     os.makedirs(os.path.dirname(path_small), exist_ok=True)
    #     cv2.imwrite(path_small, image_small)
    #     print("Saved", path_small)


    # t, v = split(data, test_size=0.2, random_state=42)

    # dataset1 = BiopsiesDataset(t, None, None, None, False, False)

    # train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1, shuffle=True, num_workers=0)
    # for data, target in train_loader:
    #     print(data.shape)
    #     print(target)
    #     plt.imshow(data[0,:,:])
    #     show()
        