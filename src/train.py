import click
import generate
import helper
import network
import set_train

from src.config import Config


@click.command()
@click.argument("training_type")
@click.argument("dataset")
@click.argument("output_network_type")
@click.argument("smoke_test")
def train(training_type, dataset, output_network_type, smoke_test="False"):
    """Train model with given information.

    Args:
        training_type (str): Type of training
        dataset (str): name of dataset
        output_network_type (str): type of output dataset
        smoke_test (str, optional): Whether to test code or train. Defaults to "False".
    """
    smoke_test = True if smoke_test == "True" else False

    model = network.get_model(
        base_network_type="resnet_50",
        output_network_type=output_network_type,
        training_type=training_type,
    )

    train_generator = generate.get_generator(
        dataset=dataset,
        purpose="train",
        seed=Config.seed,
        augment_func=helper.keras_augment_func,
    )

    valid_generator = generate.get_generator(
        dataset=dataset, purpose="valid", seed=Config.seed
    )

    set_train.set_train(
        model=model,
        train_data=train_generator,
        valid_data=valid_generator,
        dataset=dataset,
        output_network_type=output_network_type,
        training_type=training_type,
        smoke_test=smoke_test,
        seed=Config.seed,
    )


if __name__ == "__main__":
    train()
