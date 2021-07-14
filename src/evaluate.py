import json
import os

import click
import generate
import helper
import print_dict
import tensorflow as tf
import tensorflow_addons

from src.config import Config


@click.command()
@click.argument("training_type")
@click.argument("dataset")
@click.argument("output_network_type")
def evaluate_model(training_type, dataset, output_network_type):
    """Evalute model.

    Args:
        training_type (str): type of training
        dataset (str): name of dataset
        output_network_type (str): type of output network
    """
    dir_name = os.path.join(
        Config.trained_model_path, training_type, dataset, output_network_type
    )

    file_name = helper.get_all_files_in_dir(dir_name=dir_name, extension=".h5")[0]

    model = tf.keras.models.load_model(
        filepath=os.path.join(dir_name, file_name),
        custom_objects={"Addons>SGDW": tensorflow_addons.optimizers.SGDW},
    )

    result = model.evaluate(
        x=generate.get_generator(dataset=dataset, purpose="valid"),
        batch_size=Config.batch_size,
        workers=Config.num_of_workers,
        verbose=1,
        return_dict=True,
    )

    print_dict.pd(arg=result)

    print("\n*********************************************\n")
    with open(os.path.join(dir_name, f"result_{output_network_type}.json"), "w") as f:
        json.dump(result, f, indent=4)
    print("\n*********************************************\n")


if __name__ == "__main__":
    evaluate_model()
