import os
import shutil
from tqdm import tqdm
from hashlib import md5


def prepare_kaggle_dataset(
    dataset_dir="/data/raw/kaggle",
    dataset_name="paultimothymooney/kermany2018",
    data_format=".jpeg",
    download=True,
    move_duplicates=True,
):
    """
    Downloads kaggle dataset of OCT images to dataset_dir, if download flag is set to True. Removes duplicates to ensure unique samples in dataset.
    Input:
        -dataset_dir: Type String - Directory where to place downloaded dataset. Default is '/data/raw/kaggle'
        -dataset_name: Type String - Name of dataset to download from kaggle. Default is OCT Image dataset from 2017: 'paultimothymooney/kermany2018'
        -data_format: Type String - Format of data items inside dataset to check duplicates. Default is '.jpeg' format
        -download: Type Boolean - Flag to trigger dataset download. Default is 'True'
        -move_duplicates: Type Boolean - Flag to print list of found duplicates to command line. Default is 'True'
    Output:
        Prints list of found duplicates
    """
    data_dir = os.path.abspath(os.getcwd() + dataset_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if download:
        try:
            print("Downloading dataset...")
            os.open(
                "kaggle datasets download -d {} -p {} --unzip".format(
                    dataset_name, data_dir
                )
            )
        except:
            print("Dataset was not sucessfully downloaded.")

    dataset_dir = data_dir + "/kermany2018/OCT2017"
    duplicate_dir = dataset_dir + "/duplicates"
    if not os.path.exists(duplicate_dir) and move_duplicates:
        os.mkdir(duplicate_dir)
    # list of duplicates
    duplicates = []
    duplicates.append(("Duplicate", "Original file"))
    hash_keys = dict()
    # walk throught dataset
    print("Checking for duplicates")
    for root, dir, files in tqdm(os.walk(dataset_dir)):
        if not root == duplicate_dir:
            for file in files:
                filepath = os.path.join(root, file)
                # check if file has required format
                if file.endswith(data_format):
                    filehash = file_hash(filepath)
                    if filehash not in hash_keys:
                        hash_keys[filehash] = file
                    else:
                        # creat list of duplicates
                        duplicates.append((file, hash_keys[filehash]))
                        # duplicates will be moved to a separate directory and named
                        if move_duplicates:
                            i = 0
                            while os.path.exists(os.path.join(duplicate_dir, file)):
                                i += 1
                                index = file.find("_copy_")
                                if index == -1:
                                    new_file = file + "_copy_{}".format(i)
                                else:
                                    new_file = file[:index] + "_copy_{}".format(i)
                                os.rename(
                                    os.path.join(root, file),
                                    os.path.join(root, new_file),
                                )
                                file = new_file
                            shutil.move(os.path.join(root, file), duplicate_dir)
    return duplicates


def file_hash(filepath):
    with open(filepath, "rb") as f:
        return md5(f.read()).hexdigest()


def main(print_duplicates=True):
    print("Preparing kaggle OCT image dataset")
    duplicates = prepare_kaggle_dataset(download=True)
    num_dupl = len(duplicates)
    print("Processing finished. {} duplicates found.".format(num_dupl))
    if num_dupl > 0 and print_duplicates:
        print("Found duplicates:")
        row = "{element1} | {element2}".format
        for tupel in duplicates:
            print(row(element1=tupel[0].ljust(20), element2=tupel[1].ljust(20)))


if __name__ == "__main__":
    main(print_duplicates=True)
