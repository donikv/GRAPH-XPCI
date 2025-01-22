from typing import Any
import torch
import cv2
import numpy as np
from skimage import exposure
from torchvision.transforms.functional import equalize

class HistogramStretching:
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            t = pic.dtype
            if t != torch.uint8:
                pic = (pic * 255).to(torch.uint8)
            pic = equalize(pic)
            if t != torch.uint8:
                pic = pic.to(t) / 255
            return pic
        return exposure.equalize_hist(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class ToRGBImage:
    def __call__(self, pic):
        if pic.shape[0] == 3:
            return pic
        return torch.tile(pic, (3, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, pic):
        return pic.to(self.device)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensorLabel:
    def __init__(self, t=torch.float32):
        self.t = t

    def __call__(self, label):
        return torch.tensor(label, dtype=self.t)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class Equalize:

    def __init__(self, reference="./data/07_4001_3R1212.rec.8bit.tif", ks=320, contrast=0.5, size=None, remove_artefacts=True) -> None:
        self.arm = RemoveImagingArtifacts() if remove_artefacts else None
        self.ks = ks
        self.contrast = contrast
        self.size = size
        self.reference = self.load(reference, self.size)
    
    def load(self, path, size) -> np.ndarray:
        i2 = load_image_greyscale(path)
        if size is not None:
            i2 = cv2.resize(i2, size)
        if len(i2.shape) == 3:
            i2 = i2[:,:,0]
        img2 = i2.astype(np.float32) / i2.max()
        if self.arm is not None:
            img2 = self.arm(img2)
        if self.ks > 0:
            img2 = exposure.equalize_adapthist(img2, self.ks, self.contrast)
        return img2
    
    def __call__(self, img) -> Any:
        if self.ks > 0:
            img = exposure.equalize_adapthist(img, self.ks, self.contrast)
        img = exposure.match_histograms(img, self.reference)
        img = img.astype(np.float32) / img.max()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

        
def load_image_greyscale(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image
        

def regression_target_transform(target):
    return torch.tensor(target).float().unsqueeze(-1)

def remove_contours(sample, kernel_size=25, low_threshold=50, high_threshold=120):
    cnts = get_contours(sample, kernel_size, low_threshold, high_threshold)
    inv_cnts = 255 - cnts
    gaus_cnts = cv2.GaussianBlur(inv_cnts, (71,71), 0) / 255
    avg = np.mean(sample)
    masked = (gaus_cnts * sample + (1-gaus_cnts) * avg).astype(np.uint8)
    return masked

def remove_lines(sample, kernel_size=31, low_threshold=50, high_threshold=80, rho = 1, theta = np.pi / 180, threshold = 15, min_line_length = 100, max_line_gap = 10, canny=True):
    cnts = get_lines(sample, kernel_size, low_threshold, high_threshold, rho, theta, threshold, min_line_length, max_line_gap, canny)
    inv_cnts = 255 - cnts
    gaus_cnts = cv2.GaussianBlur(inv_cnts, (71,71), 0) / 255
    avg = np.mean(sample)
    masked = (gaus_cnts * sample + (1-gaus_cnts) * avg).astype(np.uint8)
    return masked

def get_lines(sample, kernel_size=31, low_threshold=50, high_threshold=80, rho = 1, theta = np.pi / 180, threshold = 15, min_line_length = 100, max_line_gap = 10, canny=True):
    blur_gray = cv2.GaussianBlur(sample,(kernel_size, kernel_size),0)
    if canny:
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    else:
        edges = np.logical_or(blur_gray < low_threshold, blur_gray > high_threshold).astype(np.uint8) * 255
    line_image = np.copy(sample) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines is None:
        return line_image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),255,100)
    
    return line_image

def get_contours(sample, kernel_size=25, low_threshold=50, high_threshold=120):
    blur_gray = cv2.GaussianBlur(sample,(kernel_size, kernel_size),0)
    # print(low_threshold, high_threshold)
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = np.copy(sample) * 0  # creating a blank to draw lines on
    cv2.drawContours(contours_image, contours, -1, 255, 100)

    return contours_image

def discretize_np(sample):
    return (sample * 255).astype(np.uint8)

class RemoveImagingArtifacts:
    def __init__(self, 
                 remove_contours=True, remove_lines=True, 
                 kernel_size_cnt=25, low_threshold=50, high_threshold=120,
                 kernel_size_lines=31,
                 canny=True, # use canny edge detection instead of thresholding for lines
                 rho = 1, # distance resolution in pixels of the Hough grid
                 theta = np.pi / 180,  # angular resolution in radians of the Hough grid
                 threshold = 15, # minimum number of votes (intersections in Hough grid cell)
                 min_line_length = 100,  # minimum number of pixels making up a line
                 max_line_gap = 10):  # maximum gap in pixels between connectable line segments
        self.remove_contours = remove_contours
        self.remove_lines = remove_lines
        self.kernel_size_cnt = kernel_size_cnt
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size_lines = kernel_size_lines
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.canny = canny

    def __call__(self, sample):
        sample = discretize_np(sample)
        if self.remove_contours:
            sample = remove_contours(sample, self.kernel_size_cnt, self.low_threshold, self.high_threshold)
        if self.remove_lines:
            sample = remove_lines(sample, self.kernel_size_lines, self.low_threshold, self.high_threshold, 
                                  self.rho, self.theta, self.threshold, self.min_line_length, self.max_line_gap, self.canny)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class RandomRemoveImagingArtifacts(RemoveImagingArtifacts):
    def __init__(self, 
                 remove_contours=True, remove_lines=True, 
                 kernel_size_cnt=25, low_threshold=50, high_threshold=120,
                 kernel_size_lines=31,
                 canny=True, # use canny edge detection instead of thresholding for lines
                 rho = 1, # distance resolution in pixels of the Hough grid
                 theta = np.pi / 180,  # angular resolution in radians of the Hough grid
                 threshold = 15, # minimum number of votes (intersections in Hough grid cell)
                 min_line_length = 100,  # minimum number of pixels making up a line
                 max_line_gap = 10, # maximum gap in pixels between connectable line segments           
                 random_kernel_offsets = [-2,2,4,-4,0],
                 random_scale = 0.1
                 ):  
        super().__init__(remove_contours, remove_lines, kernel_size_cnt, low_threshold, high_threshold, kernel_size_lines, canny, rho, theta, threshold, min_line_length, max_line_gap)
        self.random_kernel_offsets = random_kernel_offsets
        self.random_scale = random_scale

    def __call__(self, sample):
        kernel_offset1 = self.random_kernel_offsets[torch.randint(0, len(self.random_kernel_offsets), (1,))]
        kernel_offset2 = self.random_kernel_offsets[torch.randint(0, len(self.random_kernel_offsets), (1,))]
        low_treshold = self.low_threshold + torch.normal(0, self.low_threshold * self.random_scale, (1,)).to(torch.int32).item()
        high_threshold = self.high_threshold + torch.normal(0, self.high_threshold * self.random_scale, (1,)).to(torch.int32).item()
        min_line_length = self.min_line_length + torch.normal(0, self.min_line_length * self.random_scale, (1,)).to(torch.int32).item()
        sample = discretize_np(sample)
        if self.remove_contours:
            sample = remove_contours(sample, self.kernel_size_cnt + kernel_offset1, low_treshold, high_threshold)
        if self.remove_lines:
            sample = remove_lines(sample, self.kernel_size_lines + kernel_offset2, low_treshold, high_threshold, 
                                  self.rho, self.theta, self.threshold, min_line_length, self.max_line_gap, self.canny)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'