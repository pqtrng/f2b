import glob
import os
import pathlib
import shutil

import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import tqdm

from src.config import Config


def plot_idx(idx, dataframe):
    """Plot a single image at index.

    Args:
        idx (Int): Index of image
        dataframe (DataFrame): Dataframe
    """
    original_image = cv2.imread(dataframe.iloc[idx].path)
    reverse_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.imshow(reverse_color)


def plot_batch_images(batch_size, dataframe):
    """Plot 16 images in the batch, along with the corresponding labels.

    Args:
        batch_size (Int): [description]
        dataframe (Dataframe): [description]
    """

    fig = plt.figure(figsize=(20, batch_size))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(4, batch_size // 4, idx + 1, xticks=[], yticks=[])
        plot_idx(idx + 505, dataframe)
        if "height" in dataframe.columns and "weight" in dataframe.columns:
            ax.set_title(
                "H:{:.1f}    W:{:.1f}    BMI:{:.2f}".format(
                    dataframe.iloc[idx + 505].height,
                    dataframe.iloc[idx + 505].weight,
                    dataframe.iloc[idx + 505].BMI,
                )
            )
        else:
            ax.set_title("BMI:{:.2f}".format(dataframe.iloc[idx].BMI))


def checking_dir(dir_name):
    """Checking if a directory is existed, if not create one.

    Args:
        dir_name (str): parent directory
        folder (str, optional): Name of folder. Defaults to "data".

    Returns:
        dir (str): checked directory
    """
    if not os.path.exists(dir_name):
        # print(f"{dir_name} is not existed. Creating it!")
        os.makedirs(dir_name)

    return dir_name


def create_output_path(output_filepath, dataset_dir_name):
    """Create output directory.

    Args:
        output_filepath (str): output directory
        dataset_dir_name (str): name of dataset

    Returns:
        str: path of created directories
    """
    train_output_filepath = os.path.join(output_filepath, dataset_dir_name + "_train")
    valid_output_filepath = os.path.join(output_filepath, dataset_dir_name + "_valid")

    train_images_output_filepath = os.path.join(
        train_output_filepath, Config.default_images_directory_name
    )
    valid_images_output_filepath = os.path.join(
        valid_output_filepath, Config.default_images_directory_name
    )

    for dir_name in [
        train_output_filepath,
        valid_output_filepath,
        train_images_output_filepath,
        valid_images_output_filepath,
    ]:
        _ = checking_dir(dir_name=dir_name)

    return (
        train_output_filepath,
        train_images_output_filepath,
        valid_output_filepath,
        valid_images_output_filepath,
    )


def get_subfolder_name(dir_name):
    """Get the sub-directory inside the given directory.

    Args:
        dir_name (str): given directory to check

    Returns:
        str: path of sub-directory if exist else raise ValueError
    """
    os.chdir(dir_name)
    sub_dirs = [d for d in pathlib.Path(dir_name).iterdir() if d.is_dir()]
    if len(sub_dirs) == 1:
        return pathlib.Path(sub_dirs[0]).absolute()
    else:
        raise Exception(f"There are more than one sub-directories in {dir_name}!")


def get_annotation_file(annotation_file_path):
    """Get the annotation file of dataset.

    Args:
        annotation_file_path (str): Path to the dataset

    Returns:
        str: absolute path to the file
    """

    os.chdir(annotation_file_path)
    if len(glob.glob("*.csv")) == 1:
        return pathlib.Path(glob.glob("*.csv")[0]).absolute()
    elif len(glob.glob("*.csv")) == 0:
        return None
    else:
        raise Exception("There are more than 1 annotation file in path!")


def get_all_files_in_dir(dir_name, extension=None, must_sort=False):
    """Find all file in a directory.

    Args:
        dir_name (str): Working directory
        extension (string): extension of file

    Returns:
        all_files (list): a list names of files in working directory
    """
    import glob

    os.chdir(dir_name)
    file_name = "*" + extension if extension else "*"
    all_files = glob.glob(file_name)
    return natsort.natsorted(all_files) if must_sort else all_files


def get_images_name(
    dirname, column_name=[Config.default_path_column_name, Config.x_col]
):
    """Get all images in a directory and return.

    Args:
        dirname (str): name of directory to check
        column_name (list, optional): Name of columns to create output dataframe. Defaults to [Config.default_path_column_name, Config.x_col].

    Returns:
        dataframe: An dataframe with information of all image in directory
    """
    all_files = get_all_files_in_dir(dir_name=dirname, must_sort=True)
    # print(f"Total {len(all_files)} photos.")
    file_paths = [pathlib.Path(file_name).absolute() for file_name in all_files]
    image_path_name_df = pd.DataFrame(
        list(zip(file_paths, all_files)), columns=column_name
    )
    return image_path_name_df


def create_dataframe(annotation_file_path, images_dir_name):
    """Create dataframe with images directory and annotation file.

    Args:
        annotation_file_path (str): path to the annotation file
        images_dirname (str): name of image directory

    Returns:
        dataframe: Dataframe from given infomation
    """
    annotation_file = get_annotation_file(annotation_file_path)
    annotation_dataframe = pd.read_csv(annotation_file)

    # Get rid of Unnamed colum
    annotation_dataframe = annotation_dataframe.loc[
        :, ~annotation_dataframe.columns.str.contains("^Unnamed")
    ]
    image_df = get_images_name(os.path.join(annotation_file_path, images_dir_name))
    full_df = image_df.merge(annotation_dataframe, left_index=True, right_index=True)

    # Rename all columns to lower case
    full_df.columns = full_df.columns.str.lower()
    # print(f"Full dataframe has shape: {full_df.shape}.")
    print(full_df.head())
    return full_df


def split_dataframe(dataframe, first_dest_path, second_dest_path):
    """Split a dataframe into 2 set.

    Args:
        dataframe (dataframe): Dataframe to split
        first_dest_path (str): Path to save first part
        second_dest_path (str): Path to save second part

    Returns:
        dataframe: 2 new dataframes
    """
    train_df_male = dataframe[dataframe.image.str.contains("^m")].sample(
        frac=Config.train_test_split,
        random_state=Config.seed,
    )

    train_df_female = dataframe[dataframe.image.str.contains("^f")].sample(
        frac=Config.train_test_split,
        random_state=Config.seed,
    )

    first_dataframe = pd.concat([train_df_female, train_df_male])
    second_dataframe = dataframe.drop(first_dataframe.index)

    print(
        f"Splitting dataframe into \n\tfirst_set: {len(first_dataframe)} files. \n\t\tNumber of males: {len(first_dataframe[first_dataframe.image.str.contains('^m')])} files.\n\t\tNumber of females: {len(first_dataframe[first_dataframe.image.str.contains('^f')])} files.\n\tsecond_set: {len(second_dataframe)} files.\n\t\tNumber of males: {len(second_dataframe[second_dataframe.image.str.contains('^m')])} files. \n\t\tNumber of females: {len(second_dataframe[second_dataframe.image.str.contains('^f')])} files."
    )

    first_dataframe.to_csv(
        os.path.join(first_dest_path, Config.default_annotation_file_name),
        index=False,
        header=True,
    )
    second_dataframe.to_csv(
        os.path.join(second_dest_path, Config.default_annotation_file_name),
        index=False,
        header=True,
    )

    return first_dataframe, second_dataframe


def copy_image_from_dataframe(
    destination, dataframe, column_name=Config.default_path_column_name
):
    """Copy images with information from dataframe to destination.

    Args:
        destination (str): Destination to copy
        dataframe (dataframe): Dataframe contains images information
        column_name (str, optional): Name of column contains image's path. Defaults to Config.default_path_column_name.
    """
    for file_name in tqdm.tqdm(dataframe[column_name], total=len(dataframe.index)):
        shutil.copy(src=file_name, dst=destination)
