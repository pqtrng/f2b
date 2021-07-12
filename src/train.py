import click
import generate
import helper
import network
import train_func

from src.config import Config


@click.command()
@click.argument("training_type")
@click.argument("dataset")
@click.argument("network_type")
@click.argument("smoke_test")
def train(training_type, dataset, network_type, smoke_test="False"):
    smoke_test = True if smoke_test == "True" else False

    model = network.get_model(
        output_network=network.create_output_network(network_type=network_type),
        base_network=network.create_base_network(network_name="resnet_50"),
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

    train_func.set_training(
        model=model,
        train_data=train_generator,
        valid_data=valid_generator,
        dataset=dataset,
        network_type=network_type,
        smoke_test=smoke_test,
        seed=Config.seed,
    )


if __name__ == "__main__":
    train()
