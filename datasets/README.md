# Downloading datasets

This directory includes instructions and scripts for downloading ImageNet and LSUN bedrooms for use in this codebase.

## Class-conditional ImageNet

For our class-conditional models, we use the official ILSVRC2012 dataset with manual center cropping and downsampling. To obtain this dataset, navigate to [this page on image-net.org](http://www.image-net.org/challenges/LSVRC/2012/downloads) and sign in (or create an account if you do not already have one). Then click on the link reading "Training images (Task 1 & 2)". This is a 138GB tar file containing 1000 sub-tar files, one per class.

Once the file is downloaded, extract it and look inside. You should see 1000 `.tar` files. You need to extract each of these, which may be impractical to do by hand on your operating system. To automate the process on a Unix-based system, you can `cd` into the directory and run this short shell script:

```
for file in *.tar; do tar xf "$file"; rm "$file"; done
```

This will extract and remove each tar file in turn.

Once all of the images have been extracted, the resulting directory should be usable as a data directory (the `--data_dir` argument for the training script). The filenames should all start with WNID (class ids) followed by underscores, like `n01440764_2708.JPEG`. Conveniently (but not by accident) this is how the automated data-loader expects to discover class labels.

## LSUN bedroom

To download and pre-process LSUN bedroom, clone [fyu/lsun](https://github.com/fyu/lsun) on GitHub and run their download script `python3 download.py bedroom`. The result will be an "lmdb" database named like `bedroom_train_lmdb`. You can pass this to our [lsun_bedroom.py](lsun_bedroom.py) script like so:

```
python lsun_bedroom.py bedroom_train_lmdb lsun_train_output_dir
```

This creates a directory called `lsun_train_output_dir`. This directory can be passed to the training scripts via the `--data_dir` argument.

## Kaggle OCT Image datast

The script kaggle_oct.py allows for the download of the dataset via CLI command inside the script. The dataset can be found [here](https://www.kaggle.com/datasets/paultimothymooney/kermany2018). The script also checks for duplicates via MD5 hashing and removes them from the train, val, test folders into a separate duplicates folder within the dataset. The requirements in requirements.txt should be satisfied. Default dataset folder is '/data/raw/kaggle' but can be changed by flag --dataset_dir. dataset_name, format, download and move_duplicates are also set by default.

'''
python kaggle_oct.py --dataset_dir --dataset_name --data_format -- download --move_duplicates
'''

Creates folder in 'data/raw/kaggle' and extracts dataset here. Duplicates are also moved to separate folder by default.
