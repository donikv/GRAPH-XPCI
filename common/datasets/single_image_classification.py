from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split as split
from common.preprocessing import RandomRemoveImagingArtifacts, RemoveImagingArtifacts, HistogramStretching, Equalize, ToRGBImage
from common.utils import split as split_holdout
from common.utils import data_for_type
from random import choices
from common.datasets.dataset_transforms import make_base_transforms
import pandas as pd

def balance_df(df, args, groupby='target'):
    g = df.groupby(groupby)
    ndf = pd.DataFrame(
        g.apply(lambda x: x.sample(g.size().max(), replace=True, random_state=args.seed)
        .reset_index(drop=True))
        )
    return ndf

def make_csv_dataset(args, dataset: Dataset) -> (Dataset, Dataset, Dataset, list):
    transform, transform_test = make_base_transforms(args)
    root = args.path
    csv_path = args.train_csv
    csv_path_test = args.test_csv
    train_df = pd.read_csv(csv_path)
    if args.balance_dataset:
        train_df = balance_df(train_df, args, groupby=['type', 'target'])
        # train_df = balance_df(train_df, args, groupby='target')
    test_df = pd.read_csv(csv_path_test)
    val_df, test1_df = split(test_df, test_size=0.5, random_state=args.seed)
    
    dataset1 = dataset(train_df, root, transform=transform, shared_transforms=None, target_transform=None, preload=False, cache=False)
    dataset2 = dataset(val_df, root, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)
    dataset_test = dataset(test_df, root, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)
    return dataset1, dataset2, dataset_test, sorted(train_df.target.unique())

def make_dataset(args, dataset: Dataset) -> (Dataset, Dataset, Dataset, list):
    transform, transform_test = make_base_transforms(args)
    
    data = np.loadtxt(args.list, dtype=str)
    data = data_for_type(data, type=args.tissue_type)
    print(len(data))

    label = list(map(dataset.__load_target__, data))
    data = [args.path + '/' + x[2:] for x in data]

    if args.no_split and args.test:
        dataset1 = dataset(data, transform=transform, shared_transforms=None, target_transform=None, preload=False, cache=False)
        dataset2 = dataset(data, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)
        dataset_test = dataset(data, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)
    else:
        if not args.split_holdout or args.split.find("holdout") == -1:
            t, test_data, t_y, test_y = split(data, label, test_size=0.2, random_state=args.seed) # Add y for stratified split
            t, v, train_y, valid_y = split(t, t_y, test_size=0.2, random_state=args.seed) # Add y for stratified split
        else:
            t, test_data = split_holdout(data, type=args.split)
            print(len(t), len(test_data))
            np.savetxt("./test/holdout_train.txt", t, fmt="%s")
            np.savetxt("./test/holdout_test.txt", test_data, fmt="%s")

            t_y = list(map(dataset.__load_target__, t))
            test_y = list(map(dataset.__load_target__, test_data))
            v, test_data1, vy, test_y = split(test_data, test_y, test_size=0.5, random_state=args.seed) # Add y for stratified split
        
        if args.balance_dataset:
            def balance_dataset(t, t_y): #TODO: Add balancing by downsampling 0 first
                indices_0 = [i for i, y in enumerate(t_y) if y == 0]
                #indices_0 = indices_0[::2]
                indices_1 = [i for i, y in enumerate(t_y) if y == 1]
                if "patches" in args.dataset and len(indices_1) > 0:
                    step = len(indices_0) // len(indices_1) // 2
                    print(step)
                    if step > 1:
                        indices_0 = indices_0[::step]
                indices_2 = [i for i, y in enumerate(t_y) if y == 2]
                indices_3 = [i for i, y in enumerate(t_y) if y == 3]
                print(len(indices_0), len(indices_1), len(indices_2), len(indices_3))
                max_len = max(len(indices_0), len(indices_1), len(indices_2), len(indices_3))

                if len(indices_0):
                    indices_0 = indices_0 + choices(indices_0, k=max_len - len(indices_0))
                if len(indices_1):
                    indices_1 = indices_1 + choices(indices_1, k=max_len - len(indices_1))
                if len(indices_2):
                    indices_2 = indices_2 + choices(indices_2, k=max_len - len(indices_2))
                if len(indices_3):
                    indices_3 = indices_3 + choices(indices_3, k=max_len - len(indices_3))
                    
                indices = indices_0 + indices_1 + indices_2 + indices_3
                t = [t[i] for i in indices]
                return t
            
            t = balance_dataset(t, t_y)

        dataset1 = dataset(t, transform=transform, shared_transforms=None, target_transform=None, preload=False, cache=False)
        dataset2 = dataset(v, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)
        dataset_test = dataset(test_data, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)

    return dataset1, dataset2, dataset_test, dataset.CLASSES