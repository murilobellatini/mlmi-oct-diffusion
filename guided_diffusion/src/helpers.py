from tqdm import tqdm
from PIL import Image
import os
import pathlib as pl
import mimetypes


mimetypes.init()


def resize_images(input_dir:pl.Path, size:tuple, output_dir:pl.Path=None, new_suffix='.jpg') -> bool:
    """Resizes all images within folder.

    Args:
        input_dir (pl.Path): Path of target folder
        output_dir (pl.Path, optional): Path of output folder,
        if not given creates dir withing `./resized`
        size (tuple): Target (x,y) dimensions
        new_suffix (str, optional): _description_. Defaults to '.jpg'.

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

def safe_makedirs(dirs:pl.Path) -> bool:
    """Creates nested directories safely.

    Args:
        dir (pl.Path): Target dir

    Returns:
        bool: Return True if executed successfully.
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    return True