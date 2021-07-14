import os

import click
import helper

from src.config import Config


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.argument("dataset_dir_name", type=click.Path())
def split_dataset(
    input_filepath="interim", output_filepath="processed", dataset_dir_name="original"
):
    """Spliting data set for training and testing
        1. Create required folder to training, validating, testing
        2. Split dataframe for each set into *_train, *_valid
        3. Split images for each set into *_train/images, *_valid/images
    Args:
        input_filepath (str): name of input directory
        output_filepath (str): name of destination directory
        dataset_dir_name (str): name of dataset must be splitted.
    """
    print(f"\n\tProcessing {dataset_dir_name}\n")
    (
        train_filepath,
        train_images_filepath,
        valid_filepath,
        valid_images_filepath,
    ) = helper.create_output_path(
        output_filepath=os.path.join(Config.data_path, output_filepath),
        dataset_dir_name=dataset_dir_name,
    )

    full_dataset_dir_name = os.path.join(
        Config.data_path, input_filepath, dataset_dir_name
    )

    full_images_dir_name = helper.get_subfolder_name(full_dataset_dir_name)

    dataframe = helper.create_dataframe(
        annotation_file_path=full_dataset_dir_name,
        images_dir_name=full_images_dir_name,
    )

    train_dataframe, valid_dataframe = helper.split_dataframe(
        dataframe=dataframe,
        first_dest_path=train_filepath,
        second_dest_path=valid_filepath,
    )

    helper.copy_image_from_dataframe(
        destination=train_images_filepath, dataframe=train_dataframe
    )

    helper.copy_image_from_dataframe(
        destination=valid_images_filepath, dataframe=valid_dataframe
    )

    print("Splitting completed!")


if __name__ == "__main__":
    split_dataset()
