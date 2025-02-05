import torch
import pandas as pd
import os
import numpy as np
import cv2
from skimage.morphology import disk
from skimage.filters import rank
from tqdm import tqdm
from multiprocessing import Pool
from common.preprocessing import Equalize

def get_patches(original_image, patch_dimensions):
    patches = {}
    for x,y,h,w in patch_dimensions:
        patch = original_image[:, x:x+h, y:y+w]
        patches[(x,y,h,w)] = patch
    return patches

def get_dimensions(img_name):
    _, name = os.path.split(img_name)
    name = name.split('.')[0]
    x,y,h,w = name.split('_')
    return int(x), int(y), int(h), int(w)

def process_image(name, group, args, eq, pbar=None):
    img_path = os.path.join(args.root_dir, name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    for i, row in group.iterrows():
        if pbar is not None:
            pbar.set_postfix({'patch': row.image})
        x,y,h,w = get_dimensions(row.image)
        patch = img[y:y+h, x:x+w]
        if args.equalize:
            patch = eq(patch)
            patch = (patch * 255).astype(np.uint8)
            # footprint = disk(15)
            # patch = rank.equalize(patch, footprint=footprint)
        folder = os.path.split(row.image)[0]
        folder = os.path.join(args.output_dir, folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        patch_path = os.path.join(folder, os.path.split(row.image)[1])
        cv2.imwrite(patch_path, patch)

def main(args):
    df = pd.read_csv(args.patches_csv_path)
    df = df.tail(1000)
    df2 = pd.read_csv(args.images_csv_path)

    stacked = df
    stacked1 = df2
    grouped = stacked.groupby('original_image')

    mapping = {}
    for name, group in (pbar := tqdm(grouped, total=len(grouped), desc='Getting original image path')):
        pbar.set_postfix({'name': name})
        img_path = stacked1[stacked1.image.str.contains(name)].image.values[0]
        mapping[name] = img_path

    stacked['new_original_image'] = stacked.original_image.map(lambda x: mapping[x])
    pbar = tqdm(stacked.groupby('new_original_image'), total=len(stacked.new_original_image.unique()), desc='Processing patches')
    eq = Equalize(reference="../data/07_4001_3R1212_crop.rec.8bit.tif", ks=0, size=(329,329))
    if args.num_workers > 1:
        pool = Pool(args.num_workers)
        pool.starmap(process_image, [(name, group, args, eq) for name, group in pbar])
    else:
        for name, group in pbar:
            process_image(name, group, args, eq, pbar)


    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_csv_path', type=str, default='../data/patches/train.csv')
    parser.add_argument('--images_csv_path', type=str, default='../data/images/train.csv')
    parser.add_argument('--output_dir', type=str, default='../data/patches/train')
    parser.add_argument('--root_dir', type=str, default='../data/images/train')
    parser.add_argument('--equalize', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    main(args)