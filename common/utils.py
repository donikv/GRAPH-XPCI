import cv2
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, AblationCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import os
import torch
import torchvision

from PIL import Image, ImageTk
import numpy as np
from transformers import ViTConfig, ViTForImageClassification

NEW_DATASET_REJECTION_BIOPSIES = ["38_201-01.2_2R", "49_101-55.5_3R"]

offset = 10
NEW_DATASET_REJECTIONS = [f"38_201-01.2_S1_{i:04d}.rec.8bit.tif" for i in range(1900-offset,2151+offset)]
NEW_DATASET_REJECTIONS += [f"38_201-01.2_S2_{i:04d}.rec.8bit.tif" for i in range(10,151+offset)]

NEW_DATASET_REJECTIONS += [f"49_101-55.5_HA_S2_{i:04d}.rec.8bit.tif" for i in range(250-offset,951+offset)]
NEW_DATASET_REJECTIONS += [f"49_101-55.5_HA_S2_{i:04d}.rec.8bit.tif" for i in range(1100-offset,1501+offset)]
NEW_DATASET_REJECTIONS += [f"49_101-55.5_HA_S3_{i:04d}.rec.8bit.tif" for i in range(300-offset,1001+offset)]

NEW_DATASET_BORDER_2R = [f"38_201-01.2_S1_{i-offset:04d}.rec.8bit.tif" for i in range(1900-offset,1900)]
NEW_DATASET_BORDER_2R += [f"38_201-01.2_S2_{i+offset:04d}.rec.8bit.tif" for i in range(151,151+offset)]

NEW_DATASET_BORDER_3R = [f"49_101-55.5_HA_S2_{i-offset:04d}.rec.8bit.tif" for i in range(240,250)]
NEW_DATASET_BORDER_3R += [f"49_101-55.5_HA_S2_{i+offset:04d}.rec.8bit.tif" for i in range(1501,1511)]
NEW_DATASET_BORDER_3R += [f"49_101-55.5_HA_S3_{i-offset:04d}.rec.8bit.tif" for i in range(290,300)]
NEW_DATASET_BORDER_3R += [f"49_101-55.5_HA_S3_{i+offset:04d}.rec.8bit.tif" for i in range(1001,1101)]

NEW_DATASET_BORDER = NEW_DATASET_BORDER_2R + NEW_DATASET_BORDER_3R

#NEW PROPOSED SPLITS FOR THE BIOPSIES, SEPARATED BY FRESH AND ARCHIVE
BIOPSIES_ARHIVE_TRAIN = ["34_1013_1_0R", "06_4003_0R", "08_4001_2_3R", "14_4004_2R", "04_1051_1R1AQ_B2"]
BIOPSIES_ARHIVE_HOLDOUT = ["12_4009_2R", "07_4001_3R", "33_1012_1_0R", "19_1021_1_1R1a"]
BIOPSIES_ARHIVE = BIOPSIES_ARHIVE_TRAIN + BIOPSIES_ARHIVE_HOLDOUT
BIOPSIES_ARHIVE_SPLIT_ANOMALY = ["33_1012_1_0R", "19_1021_1_1R1a", "2R", "3R"]

BIOPSIES_FRESH_UNANNOTATED = ["024_101_16", "031_101_22", "038_101_26", "016_101_10", "025_101_16.2", "033_101_21", "039_101_27", "017_101_11", "026_101_17", "034_101_23", "040_101_28", "018_101_12", "027_101_18", "035_101_24", "041_101_30", "019_101_13", "028_101_19", "036_101_25", "043_101_32", "023_101_15", "029_101_20", "037_101_25.2"]
BIOPSIES_FRESH_TRAIN = ["014_101_08", "38_201-01.2", "49_101-55.5"]
BIOPSIES_FRESH = BIOPSIES_FRESH_TRAIN + BIOPSIES_FRESH_UNANNOTATED

BIOPSIES_FRESH_HOLDOUT = ["014_101_08_HA_S2", "49_101-55.5_HA_S3", "38_201-01.2_S2"]

SPLITS = ['holdout', 'holdout_legacy', 'holdout_fresh', 'holdout_archive', 'holdout_archive_anomaly']
BIOPSY_TYPE = ['fresh', 'fresh_unannotated', 'fresh_train', 'archive', 'archive_patches', 'fresh_patches']

def biopsy_names(data, path):
    names = [x[len(path):] for x in data]
    # Create a set of unique names, where the name is the first part of the path if the path begins with 2017, othewise its the second part
    imaging = set([x.split("/")[1] if x.split("/")[0].find("2017") != -1 else x.split("/")[0] for x in names])
    # Merge names that are the same but end in B and then a number
    imaging = set([x[:-3] if x[-2] == "B" else x for x in imaging])
    return list(imaging)

CLASSES = ["0R", "1R", "2R", "3R"]
def image_path_to_target(path, cls=CLASSES):
    for i, c in enumerate(cls):
        if c in path:
            return i
    return -1

def biopsies_target(path, cls=CLASSES):
    image_name = os.path.split(path)[1]

    for r in NEW_DATASET_REJECTION_BIOPSIES:
        if r in path:
            if image_name in NEW_DATASET_REJECTIONS:
                return image_path_to_target(path, cls)
            elif image_name in NEW_DATASET_BORDER_2R: #ADDED BORDER CASE TO 1R
                return 1
            elif image_name in NEW_DATASET_BORDER_3R:
                return 2 #image_path_to_target(path, cls) #1
            else:
                return 0
    return image_path_to_target(path, cls)

def patch_target(path, cls=['clean', 'rejection']):
    for i, c in enumerate(cls):
        if c in path:
            return i


def data_for_type(data, type):
    if type not in BIOPSY_TYPE:
        raise ValueError(f"Unknown biopsy type {type}")
    
    if type == "fresh":
        accept = BIOPSIES_FRESH #TODO: CREATE A LIST OF FRESH BIOPSIES
        exclude = []
    elif type == "fresh_unannotated":
        accept = BIOPSIES_FRESH_UNANNOTATED
        exclude = []
    elif type == "fresh_train":
        accept = BIOPSIES_FRESH_TRAIN
        exclude = []
    elif type == "archive":
        accept = BIOPSIES_ARHIVE
        exclude = []
    elif type == "archive_patches":
        accept = BIOPSIES_ARHIVE
        exclude = ["1R", "2R"]
    elif type == "fresh_patches":
        accept = BIOPSIES_FRESH_TRAIN
        exclude = []

    def f(x):
        for e in exclude:
            if e in x:
                return False
        for a in accept:
            if a in x:
                return True
        return False
    
    return list(filter(f, data))

def split(data, type="holdout_legacy"):
    if type not in SPLITS:
        raise ValueError(f"Unknown split type {type}")
    
    if type == "holdout":
        holdout = BIOPSIES_ARHIVE_HOLDOUT + BIOPSIES_FRESH_HOLDOUT
    elif type == "holdout_legacy" or type == "holdout_archive":
        holdout = BIOPSIES_ARHIVE_HOLDOUT
    elif type == "holdout_fresh":
        holdout = BIOPSIES_FRESH_HOLDOUT
    elif type == "holdout_archive_anomaly":
        holdout = BIOPSIES_ARHIVE_SPLIT_ANOMALY

    def f(x):
        for h in holdout:
            if h in x:
                return True
        return False
    test_data = list(filter(f, data))
    t = list(filter(lambda x: not f(x), data))
    return t, test_data

def load_image_greyscale(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return image

def load_image_16bit(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0,0), fx=1/4, fy=1/4, interpolation=cv2.INTER_NEAREST)
    return image

def createConfusionMatrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def createClassificationReport(y_true, y_pred, output_dict=False):
    return classification_report(y_true, y_pred, output_dict=output_dict)

def createGradCamExplanation(model, target_layers, img, transform, cam=None):
    input_tensor = transform(img).unsqueeze(0)
    print(img.shape, input_tensor.shape)
    if cam is None:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor)
    img1 = cv2.resize(img, (input_tensor.shape[3], input_tensor.shape[2]))

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img1[...,np.newaxis], grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam

def visualize_vitmae(pixel_values, model):
    pixel_values = pixel_values[:8]
    # forward pass
    outputs = model(pixel_values)
    y = model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    # visualize the mask
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.in_chns)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', pixel_values).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    print(im_paste.shape, im_masked.shape, x.shape)
    img = torch.concat([x, im_masked, im_paste], dim=0)
    img = torch.einsum('nhwc->nchw', img)

    grid = torchvision.utils.make_grid(img, normalize=True).cpu().numpy().transpose((1, 2, 0))
    print(grid.shape)

    return grid

def start_end(volume, x, size, axis):
    max = volume.shape[axis]
    if x - size//2 < 0:
        return 0, size
    elif x + size//2 > volume.shape[axis]:
        return max - size, max
    else:
        return x - size//2, x + size//2

def extract_cube(volume, z, y, x, size=256):
    zs, ze = start_end(volume, z,size,0) 
    ys, ye = start_end(volume, y,size,1)
    xs, xe = start_end(volume, x,size,2)
    return volume[zs:ze, ys:ye, xs:xe]
class Sparse3DMatrix:
    def __init__(self, volume, t_l, t_h, subsample=10) -> None:
        self.volume = volume
        self.t_h = t_h
        self.t_l = t_l
        self.shape = volume.shape
        self.__create_sparse(subsample)
    
    def __create_sparse(self, subsample):
        self.d = {}
        for z in range(0, self.shape[0], subsample):
            for y in range(0, self.shape[1], subsample):
                for x in range(0, self.shape[2], subsample):
                    if self.volume[z, y, x] > self.t_l and self.volume[z, y, x] < self.t_h:
                        self.d[(z,y,x)] = self.volume[z, y, x]

    def __getitem__(self, z, x, y):
        key = (z, x, y)
        return self.d[key]
    
    def __len__(self):
        return len(self.d)
    
    def plot_points(self):
        x,y,z,c = [],[],[],[]
        for key in self.d.keys():
            z.append(key[0])
            y.append(key[1])
            x.append(key[2])
            c.append(self.volume[key])
        return x,y,z,c


def nparray_to_image(array):
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(np.uint8(array))

    # Convert the PIL image to a Tkinter-compatible photo image
    tk_image = ImageTk.PhotoImage(pil_image)

    return tk_image




# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import json


def param_groups_lrd(model: ViTForImageClassification, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers