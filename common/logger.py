import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
import common.visualize_utilities as plt_util
import common.utils as utils
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, List, Union, Any, Dict
from PIL import Image
import numpy as np

class Logger():

    def __init__(self, path) -> None:
        self.path = path
    
    def log(self, message: str, add_dt=True) -> None:
        if add_dt:
            message = f"[{datetime.datetime.now()}]: {message}"
        if self.path is not None:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            with open(os.path.join(self.path, "log.txt"), "a") as f:
                f.write(f"{message}" + "\n")
        print(message)
    
    def log_dict(self, d: dict) -> None:
        add_dt = True
        for k, v in d.items():
            self.log(f"{k}: {v}", add_dt=add_dt)
            add_dt=False
    
    def log_list(self, l: list) -> None:
        for i, v in enumerate(l):
            self.log(v, add_dt=i == 0)

    def log_figure(self, fig: matplotlib.figure.Figure, name):
        path = os.path.join(self.path, f"{name}.png")
        dir = os.path.split(path)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        fig.savefig(os.path.join(self.path, f"{name}.png"), bbox_inches='tight')
        plt.close(fig)
    
    def log_image(self, img: np.ndarray, name, cmap="viridis"):
        path = os.path.join(self.path, f"{name}.png")
        dir = os.path.split(path)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)

        if len(img.shape) == 2:
            cm = plt.get_cmap(cmap)
            img = cm(img)

        if img.dtype == np.float32 or img.dtype == np.float64:
            img = Image.fromarray(np.uint8(img * 255))
        else: 
            img = Image.fromarray(img)

        img.save(path)
    
    def log_classification_report(self, targets, preds) -> None:
        classification_report = utils.createClassificationReport(targets, preds)
        self.log(f"Classification report:")
        self.log(classification_report, add_dt=False)
        


class TensorboardLogger(Logger):

    def __init__(self, path) -> None:
        super().__init__(path)
        self.writer = SummaryWriter(os.path.join(path, "tb_events"))
    
    def log_cm(self, targets: List[int], preds: List[int], epoch: int, name: str = "confusion_matrix"):
        cm = utils.createConfusionMatrix(targets, preds)
        classes = np.unique(targets)
        confusion_matrix = plt_util.plotConfusionMatrix(cm, classes=classes)
        self.log(f"Written confusion matrix: {confusion_matrix}")
        if self.writer is not None:
            self.writer.add_figure(name, confusion_matrix, epoch) 
        return confusion_matrix
    
    def log_scalar(self, tag: str, value: Union[int, float], epoch: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, epoch)
    
    def log_image(self, img: np.ndarray, name, cmap="viridis"):
        super().log_image(img, name, cmap)
        if self.writer is not None:
            dataformat = "HWC" if len(img.shape) == 3 else "HW"
            self.writer.add_image(name, img, dataformats=dataformat)