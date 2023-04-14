import argparse
from os import path, makedirs
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil


def main(image_source,
         partition_file,
         destination):
    tasks = ["train", "val", "test"]
    data_separation = np.array(pd.read_csv(partition_file))
    for image_id, task in data_separation:
        save_folder = path.join(destination, tasks[task])
        if not path.exists(save_folder):
            makedirs(save_folder)
        shutil.copy(path.join(image_source, image_id), save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Store the image paths")
    parser.add_argument("--image_source", "-s", help="path to CelebA image folder.")
    parser.add_argument("--partition_file", "-f", help="path to partition file.", type=str)
    parser.add_argument("--destination", "-d", help="dataset folder.", type=str, default="./dataset")

    args = parser.parse_args()

    main(args.image_source,
         args.partition_file,
         args.destination)
