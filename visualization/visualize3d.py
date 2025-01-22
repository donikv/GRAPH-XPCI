import tkinter as tk
from tkinter.ttk import *

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from common.utils import Sparse3DMatrix, extract_cube, start_end, nparray_to_image, load_image_greyscale, biopsy_names
from common.visualize_utilities import plot_3d_cube, mark_region, explore_slices, display_slice2
from common.dataset3d import load3d_volume
from skimage.exposure import adjust_sigmoid, equalize_adapthist

import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import os
# set_start_method('fork')



def load_image_greyscale(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image



class CustomTkinterWindow:
    def __init__(self, root, path, l):
        self.root = root
        self.path = path
        self.l = l
        self.setup_ui()

    def setup_ui(self):
        path = self.path
        l = self.l
        data = [path + x[2:] for x in np.loadtxt(l, dtype=str)]
        self.data = data
        self.imagings = biopsy_names(data, path)
        self.imagings = sorted(self.imagings)
        self.sorted_data = sorted(data)

        self.imaging_var = tk.StringVar(self.root)
        self.imaging_var.set(self.imagings[0])  # default value

        self.root.protocol("WM_DELETE_WINDOW", self.kill_process)

        self.loading_frame = tk.Frame(self.root)
        self.loading_frame.pack()


        self.dropdown = tk.OptionMenu(self.loading_frame, self.imaging_var, *self.imagings)
        self.dropdown.grid(row=0, column=0, padx=10)

        self.zl_entry = tk.Entry(self.loading_frame)
        self.zl_entry.grid(row=0, column=1, padx=10)
        self.zl_entry.insert(0, "0")

        self.zh_entry = tk.Entry(self.loading_frame)
        self.zh_entry.grid(row=0, column=2, padx=10)
        self.zh_entry.insert(0, "-1")

        self.load_button = tk.Button(self.loading_frame, text="Load", command=self.load_and_display_imaging)
        self.load_button.grid(row=0, column=3, padx=10)

        self.setup_entry_fields()

    def setup_entry_fields(self):
        # Add number fields
        self.entry_frame = tk.Frame(self.root)
        self.entry_frame.pack()

        self.size_label = tk.Label(self.entry_frame, text="Size")
        self.size_label.grid(row=0, column=0)
        self.size_entry = tk.Entry(self.entry_frame)
        self.size_entry.grid(row=1, column=0, padx=10)
        self.size_entry.insert(0, "800")

        self.tl_label = tk.Label(self.entry_frame, text="TL")
        self.tl_label.grid(row=0, column=1)
        self.tl_entry = tk.Entry(self.entry_frame)
        self.tl_entry.grid(row=1, column=1, padx=10)
        self.tl_entry.insert(0, "240")

        self.th_label = tk.Label(self.entry_frame, text="TH")
        self.th_label.grid(row=0, column=2)
        self.th_entry = tk.Entry(self.entry_frame)
        self.th_entry.grid(row=1, column=2, padx=10)
        self.th_entry.insert(0, "255")

        self.subsample_label = tk.Label(self.entry_frame, text="Subsample")
        self.subsample_label.grid(row=0, column=3)
        self.subsample_entry = tk.Entry(self.entry_frame)
        self.subsample_entry.grid(row=1, column=3, padx=10)
        self.subsample_entry.insert(0, "4")

        self.extract_3d_label = tk.Label(self.entry_frame, text="Extract 3D coordinates")
        self.extract_3d_label.grid(row=0, column=4)
        self.cb3d = tk.BooleanVar()
        self.extract_3d_checkbox = tk.Checkbutton(self.entry_frame, variable=self.cb3d, onvalue=True, offvalue=False)
        self.extract_3d_checkbox.grid(row=1, column=4, padx=10)

        self.extract_button = tk.Button(self.root, text="Extract Cube", command=self.extract_new_cube, state=tk.DISABLED)
        self.extract_button.pack()
    
    def setup_sliders(self, init_x=0, init_y=0, init_plane=0):
        if hasattr(self, 'slider_frame'):
            self.plane_slider.destroy()
            self.y_slider.destroy()
            self.x_slider.destroy()
            self.plane_label.destroy()
            self.y_label.destroy()
            self.x_label.destroy()
        else:
            self.slider_frame = tk.Frame(self.root)
            self.slider_frame.pack()

        self.plane_label = tk.Label(self.slider_frame, text="Plane")
        self.plane_label.grid(row=0, column=0)
        self.plane_slider = tk.Scale(self.slider_frame, from_=init_plane, to=self.cube.shape[0]-1, length=400, orient=tk.HORIZONTAL, command=self.redraw_slices)
        self.plane_slider.grid(row=1, column=0, padx=10)

        self.y_label = tk.Label(self.slider_frame, text="Y")
        self.y_label.grid(row=0, column=1)
        self.y_slider = tk.Scale(self.slider_frame, from_=init_y, to=self.cube.shape[1]-1, length=400, orient=tk.HORIZONTAL, command=self.redraw_slices)
        self.y_slider.grid(row=1, column=1, padx=10)

        self.x_label = tk.Label(self.slider_frame, text="X")
        self.x_label.grid(row=0, column=2)
        self.x_slider = tk.Scale(self.slider_frame, from_=init_x, to=self.cube.shape[2]-1, length=400, orient=tk.HORIZONTAL, command=self.redraw_slices)
        self.x_slider.grid(row=1, column=2, padx=10)

    def kill_process(self):
        exit(0)

    def load_and_display_imaging(self, *args):
        selected_imaging = self.imaging_var.get()
        self.root.title(selected_imaging)  # Set the window title to the selected imaging name
        self.x = 0
        self.y = 0
        self.plane = 0
        zl = int(self.zl_entry.get())
        zh = int(self.zh_entry.get())
        volume = load3d_volume(self.sorted_data, selected_imaging, start_index=zl, end_index=zh, num_workers=4)
        print(volume.shape)
        self.volume = volume
        self.extract_new_cube()
        self.extract_button.config(state=tk.NORMAL)


    def extract_new_cube(self, *args):
        size = int(self.size_entry.get())
        tl = int(self.tl_entry.get())
        th = int(self.th_entry.get())
        subsample = int(self.subsample_entry.get())
        extract_3d = self.cb3d.get()
        self.extract_visualization_cube(self.volume, size, tl, th, subsample, extract_3d)

        self.y = 0
        self.x = 0
        self.plane = 0

        self.setup_sliders(init_x=self.x, init_y=self.y, init_plane=self.plane)
        self.redraw()
    
    def extract_visualization_cube(self, volume, size, tl, th, subsample, extract_3d):
        cube = extract_cube(volume, volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2]//2, size=size)
        if extract_3d:
            sparse = Sparse3DMatrix(cube, tl, th, subsample=subsample)
            crds2 = sparse.plot_points()
        else:
            crds2 = None
        self.cube = cube
        self.crds2 = crds2

    def redraw_slices(self, *args):
        # Get the values from the sliders
        self.plane = self.plane_slider.get()
        self.x = self.x_slider.get()
        self.y = self.y_slider.get()
        self.redraw()
    
    def redraw(self):
        figure, ax3d, slices = display_slice2(self.cube,crds=self.crds2, s=20, y=self.y, x=self.x, plane=self.plane)

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
            matplotlib.pyplot.close(self.figure)
        
        if not hasattr(self, 'canvas_frame'):
            print("Creating canvas frame")
            self.canvas_frame = tk.Frame(self.root)
            self.canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        if hasattr(self, 'slider_frame'):
            self.canvas = FigureCanvasTkAgg(figure, self.slider_frame)
            self.canvas.get_tk_widget().grid(row=1, column=3, padx=10)
        else:
            self.canvas = FigureCanvasTkAgg(figure, self.canvas_frame)
            self.canvas.get_tk_widget().grid(row=0, column=3, padx=10)

        self.canvas.mpl_connect('button_press_event', ax3d._button_press)
        self.canvas.mpl_connect('button_release_event', ax3d._button_release)
        self.canvas.mpl_connect('motion_notify_event', ax3d._on_move)

        self.update_images(slices)
        self.figure = figure


    def update_images(self, slices):
        def modify_slice(slice, cutoff, gain):
            w,h = slice.shape[1], slice.shape[0]
            if w > 800:
                w = 800
                h = int(h * (w / slice.shape[1]))
                slice = cv2.resize(slice, (w, h))
            slice = (equalize_adapthist(slice, 100, 0.5) * 255).astype(np.uint8)
            # slice = adjust_sigmoid(slice, cutoff=cutoff, gain=gain)
            return slice

        slices = tuple(map(lambda x: modify_slice(x, 0.4, 20), slices))

        w1,h1 = slices[0].shape[1], slices[0].shape[0]
        w2,h2 = slices[1].shape[1], slices[1].shape[0]
        w3,h3 = slices[2].shape[1], slices[2].shape[0]
        
        self.img1 = nparray_to_image(slices[0])
        self.img2 = nparray_to_image(slices[1])
        self.img3 = nparray_to_image(slices[2])


        # if w1 > 800:
        #     w1 = 800
        #     h1 = int(h1 * (w1 / slices[0].shape[1]))
        #     self.img1 = nparray_to_image(cv2.resize(slices[0], (w1, h1)))
        # if w2 > 800:
        #     w2 = 800
        #     h2 = int(h2 * (w2 / slices[1].shape[1]))
        #     self.img2 = nparray_to_image(cv2.resize(slices[1], (w2, h2)))
        # if w3 > 800:
        #     w3 = 800
        #     h3 = int(h3 * (w3 / slices[2].shape[1]))
        #     self.img3 = nparray_to_image(cv2.resize(slices[2], (w3, h3)))

        if not hasattr(self, 'canvas1'):
            # print("Creating canvas")
            self.canvas1 = tk.Canvas(self.canvas_frame, width=w1, height=h1)
            self.canvas1.grid(row=0, column=0, padx=10, pady=10)
            self.canvas2 = tk.Canvas(self.canvas_frame, width=w2, height=h2)
            self.canvas2.grid(row=0, column=1, padx=10, pady=10)
            self.canvas3 = tk.Canvas(self.canvas_frame, width=w3, height=h3)
            self.canvas3.grid(row=0, column=2, padx=10, pady=10)

            self.canvas1_image = self.canvas1.create_image(0, 0, image=self.img1, anchor="nw")
            self.canvas2_image = self.canvas2.create_image(0, 0, image=self.img2, anchor="nw")
            self.canvas3_image = self.canvas3.create_image(0, 0, image=self.img3, anchor="nw")
        else:
            self.canvas1.config(width=w1, height=h1)
            self.canvas2.config(width=w2, height=h2)
            self.canvas3.config(width=w3, height=h3)
            # print("Updating canvas")
            self.canvas1.itemconfig(self.canvas1_image, image=self.img1)
            self.canvas2.itemconfig(self.canvas2_image, image=self.img2)
            self.canvas3.itemconfig(self.canvas3_image, image=self.img3)



if __name__ == "__main__":
    l = "./data/biopsies_new.txt"
    #path = '/home/donik/datasets/XPCI/dataset_biopsies/'
    path = '/run/media/donik/Disk/syncthing/datasets/XPCI/dataset_biopsies/'

    # tk_root = tk.Tk()
    root = tk.Tk()
    app = CustomTkinterWindow(root, path, l)
    root.mainloop()


    # data = [path + x[2:] for x in np.loadtxt(l, dtype=str)]
    # imagings = biopsy_names(data, path)
    # imagings = sorted(imagings)
    # sorted_data = sorted(data)
    # print(imagings, len(imagings))
    
    # volume = load3d_volume(sorted_data, imagings[3], num_workers=-1)
    # print(volume.shape)

    # size, tl, th, subsample = 800, 240, 255, 4

    # cube = extract_cube(volume, volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2]//2, size=size)
    # sparse = Sparse3DMatrix(cube, tl, th, subsample=subsample)
    # crds2 = sparse.plot_points()

    # figure, ax3d = display_slice(cube,crds=crds2, s=20, y=400, x=400)

    # canvas = FigureCanvasTkAgg(figure, tk_root)
    # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    # canvas.mpl_connect('button_press_event', ax3d._button_press)
    # canvas.mpl_connect('button_release_event', ax3d._button_release)
    # canvas.mpl_connect('motion_notify_event', ax3d._on_move)
    # toolbar = NavigationToolbar2Tk(canvas, tk_root)
    # toolbar.update()
    # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)



    # tk_root.mainloop()
