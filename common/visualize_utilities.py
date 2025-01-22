import matplotlib.pyplot as plt
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import seaborn as sn
import numpy as np
from copy import deepcopy
from common.utils import start_end


FIG_NUM = 1
FIG_SAVE_LOC = f'./tmp_plots'
program = None
date_time = None

def show():
    global FIG_NUM
    assert program is not None, "Set the program name"
    save_loc = f'{FIG_SAVE_LOC}/{date_time}/{program}'
    os.makedirs(save_loc, exist_ok=True)
    plt.savefig(f'{save_loc}/plot_{FIG_NUM}.png')
    FIG_NUM = FIG_NUM + 1

def visualize_set_program(p):
    global program, date_time
    program = p
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def plotConfusionMatrix(cf_matrix, classes=[0,1,2,3]):   
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


def plot_3d_cube(cube, t=0.5, transparency=0.6, s=50):
    fig = plt.figure(figsize=(24, 12))
    ax = Axes3D(fig)
    z,y,x = np.indices(cube.shape)
    print(x[1], y[1], z[1])
    c = cube.astype(float) / 255
    x = list(x[c > t]) + [0]
    y = list(y[c > t]) + [0]
    z = list(z[c > t]) + [0]
    c = list(c[c > t]) + [0.0]
    ax.scatter(x, y, z, c=c, s=s, alpha=transparency, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    return x,y,z,c

def mark_region(volume, z, y, x, size=256, border=10):
    y,y1 = start_end(volume, y, size, 1)
    x,x1 = start_end(volume, x, size, 2)

    img = deepcopy(volume[z, :, :])

    #Draw a rectangle around the region of interest
    img[y:y1, x:x+border] = 255
    img[y:y1, x1-border:x1] = 255
    img[y:y+border, x:x1] = 255
    img[y1-border:y1, x:x1] = 255

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap="gray")
    plt.show()


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import (
    exposure, util
)

def display(im3d, cmap='gray', step=2):
    data_montage = util.montage(im3d[::step], padding_width=4, fill=np.nan)
    _, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(data_montage, cmap=cmap)
    ax.set_axis_off()


def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)

def slice_in_3D(ax, i, j, k, shape, crds=None, s=50):
    # From https://stackoverflow.com/questions/44881885/python-draw-3d-cube
    Z = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])

    Z = Z * [shape[1], shape[2], shape[0]]
    r = [-1, 1]
    X, Y = np.meshgrid(r, r)

    # Plot vertices
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    # List sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]],
             [Z[2], Z[3], Z[7], Z[6]]]

    # Plot sides
    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors=(0, 1, 1, 0.25),
            linewidths=1,
            edgecolors="darkblue"
        )
    )

    verts = np.array([[[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 1],
                       [0, 1, 0]]])
    verts = verts * (60, shape[2], shape[0])
    verts += [k, 0, 0]

    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors="magenta",
            linewidths=1,
            edgecolors="black",
            alpha=0.5
        )
    )

    verts = np.array([[[0, 0, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 0, 0]]])
    verts = verts * (shape[1], 60, shape[0])
    verts += [0, j, 0]

    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors="cyan",
            linewidths=1,
            edgecolors="black",
            alpha=0.5
        )
    )

    verts = np.array([[[0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0]]])
    verts = verts * (shape[1], shape[2], 60)
    verts += [0, 0, i]

    ax.add_collection3d(
        Poly3DCollection(
            verts,
            facecolors="yellow",
            linewidths=1,
            edgecolors="black",
            alpha=0.5
        )
    )
    
    if crds is not None:
        #ax.scatter(crds[0], crds[1], [shape[0] - x for x in crds[2]], c=crds[3], s=50, alpha=0.4, cmap="gray")
        ax.scatter(crds[0], crds[1], crds[2], c=crds[3], s=s, alpha=0.4, cmap="gray")
    ax.set_xlabel("x")
    # ax.set_xlim(0, shape[2])
    # ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[0])
    ax.set_ylabel("y")
    ax.set_zlabel("plane")
    ax.invert_zaxis()

    # Autoscale plot axes
    scaling = np.array([getattr(ax,
                                f'get_{dim}lim')() for dim in "xyz"])
    ax.auto_scale_xyz(* [[np.min(scaling), np.max(scaling)]] * 3)


def explore_slices(data, cmap="gray", crds=None, s=50):
    from ipywidgets import interact
    N = len(data)
    _,Y,X = data.shape
    print(N,Y,X)

    @interact(plane=(0, N - 1), x=(0, X - 1), y=(0, Y - 1))
    def display_slice_int(plane=34, y=1280, x=1280):
        fig, ax = plt.subplots(figsize=(30, 45))

        ax = fig.add_subplot(321)
        ax_3D = fig.add_subplot(322, projection="3d")
        ax2 = fig.add_subplot(3,2,(3,5))
        ax3 = fig.add_subplot(3,2, (4,6))

        show_plane(ax, data[plane], title=f'Plane {plane}', cmap=cmap)
        zs,ze = start_end(data, plane, 2560, 0)
        show_plane(ax2, data[:, y, :], title=f'Row {y}', cmap=cmap)
        show_plane(ax3, data[:, :, x], title=f'Column {x}', cmap=cmap)
        slice_in_3D(ax_3D, plane, y, x, data.shape, crds=crds, s=s)

        plt.show()

    return display_slice_int

def display_slice(data, cmap="gray", crds=None, s=50, plane=34, y=1280, x=1280):
    N = len(data)
    _,Y,X = data.shape
    
    fig, ax = plt.subplots(figsize=(30, 45))

    ax = fig.add_subplot(221)
    ax_3D = fig.add_subplot(222, projection="3d")
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)

    show_plane(ax, data[plane], title=f'Plane {plane}', cmap=cmap)
    zs,ze = start_end(data, plane, 2560, 0)
    show_plane(ax2, data[:, y, :], title=f'Row {y}', cmap=cmap)
    show_plane(ax3, data[:, :, x], title=f'Column {x}', cmap=cmap)
    slice_in_3D(ax_3D, plane, y, x, data.shape, crds=crds, s=s)

    return fig, ax_3D

def display_slice2(data, cmap="gray", crds=None, s=50, plane=34, y=1280, x=1280):
    N = len(data)
    _,Y,X = data.shape
    
    fig = plt.figure(figsize=(3, 3))
    ax = Axes3D(fig)

    slice_in_3D(ax, plane, y, x, data.shape, crds=crds, s=s)
    slice_x = data[:,:,x]
    slice_y = data[:,y,:]
    slice_z = data[plane,:,:]

    return fig, ax, (slice_z, slice_y, slice_x)