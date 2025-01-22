from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split as split
from common.preprocessing import RandomRemoveImagingArtifacts, RemoveImagingArtifacts, HistogramStretching, Equalize, ToRGBImage
from common.utils import split as split_holdout
from common.utils import data_for_type
from common.datasets.dataset_transforms import make_base_transforms


def make_dataset(args, dataset: Dataset) -> (Dataset, Dataset, Dataset, list):
    transform, transform_test = make_base_transforms(args)
    
    data = np.loadtxt(args.list, dtype=str)
    data = data_for_type(data, type=args.tissue_type)
    data = [args.path + '/' + x[2:] for x in data]

    data, test_data = split_holdout(data, type=args.split)
    print(len(data), len(test_data))

    t,v = split(data, test_size=0.2, random_state=args.seed)
        
    dataset1 = dataset(t, transform=transform, shared_transforms=None, target_transform=None, preload=False, cache=False)
    dataset2 = dataset(v, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)
    dataset_test = dataset(test_data, transform=transform_test, shared_transforms=None, target_transform=None, preload=False, cache=False)

    return dataset1, dataset2, dataset_test, dataset.CLASSES
