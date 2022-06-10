import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import pathlib as pl
import mimetypes


mimetypes.init()


def resize_images(input_dir: pl.Path, size: tuple, output_dir: pl.Path = None, new_suffix='.jpg') -> bool:
    """Resizes all images within folder.

    Args:
        input_dir (pl.Path): Path of target folder
        output_dir (pl.Path, optional): Path of output folder,
        if not given creates dir withing `./resized`
        size (tuple): Target (x,y) dimensions
        new_suffix (str, optional): New extension for output images. Defaults to '.jpg'.

    Returns:
        bool: Return True if executed successfully.

    """
    if not output_dir:
        output_dir = input_dir / 'resized'

    safe_makedirs(output_dir)

    for f in tqdm(os.listdir(input_dir)):

        ipath = input_dir / f
        opath = output_dir / f

        ftype = mimetypes.guess_type(ipath)[0]

        if not ftype or 'image' not in ftype:
            # skips non images files
            continue

        if new_suffix:
            opath.with_suffix(new_suffix)

        iimg = Image.open(ipath)
        oimg = iimg.resize(size)
        oimg.save(opath)

    return True


def safe_makedirs(dirs: pl.Path) -> bool:
    """Creates nested directories safely.

    Args:
        dir (pl.Path): Target dir

    Returns:
        bool: Return True if executed successfully.
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    return True


def array2tuple(arr: np.array) -> tuple:
    """
    Converts array to tuple
    """
    return tuple(map(lambda x: tuple(x)[0], arr))


def get_arange_matrix(shape: tuple) -> np.array:
    """
    Generates arange in matrix format (given shape)
    """
    return np.arange(np.prod(shape)).reshape(shape)


def get_ij(ix: tuple, subplots_size=(2, 3)) -> tuple:
    """
    Gets indexes i,j from tuple ix on subplot grid
    """
    arange_matrix = get_arange_matrix(subplots_size)
    return array2tuple(np.where(ix == arange_matrix))


def turn_off_axes(axes: np.array) -> None:
    """
    Turn offs all subplot axes
    """
    [axi.set_axis_off() for axi in axes.ravel()]


def get_figsize(subplots_size: tuple, scaling_factor: int = 2) -> tuple:
    """Gets figsize from subplots grid size

    Args:
        subplots_size (tuple): Subplots grid size
        scaling_factor (int, optional): Scaling factor of figsize w.r.t. subplots_size. Defaults to 2.

    Returns:
        tuple: Figsize dimension
    """
    x, y = tuple(map(lambda x: scaling_factor*x, subplots_size))
    return y, x


def render_samples(spath: pl.Path, samples2render: int = 10, subplots_size=(5, 2), title=None) -> figure.Figure:
    """
    Displays in IPython notebook the images produced by the model.

    It renders the numpy.array output by image_sample.py

    Args:
        spath (pl.Path): Path of generated .npz file
        samples2render (int, optional): Amount of samples to render. Defaults to 10.
        subplots_size (tuple, optional): Grid size of the output plot. Defaults to (5,2).
        title (_type_, optional): Plot title, if None renders nothing. Defaults to None.

    Returns:
        figure.Figure: Plot with rendered samples
    """
    assert samples2render <= np.prod(
        subplots_size), f'`samples2render` ({samples2render}) does not fit `subplots_size` ({subplots_size}): {samples2render} > {np.prod(subplots_size)}'

    image_arrays = np.load(spath)["arr_0"]
    samples2render = min(image_arrays.shape[0], samples2render)

    images = []
    for i in range(samples2render):
        img_array = image_arrays[i]
        img = Image.fromarray(img_array)
        images.append(img)

    f, axarr = plt.subplots(
        nrows=subplots_size[0],
        ncols=subplots_size[1],
        figsize=get_figsize(subplots_size),
        squeeze=False
    )

    for ix, img in enumerate(images):
        if ix > np.prod(subplots_size):
            break
        i, j = get_ij(ix, subplots_size)
        axarr[i, j].imshow(img)
        axarr[i, j].set_title("Sample = %s" % ix)
        turn_off_axes(axarr)

    if title:
        plt.suptitle(title, y=1.20)

    return f
