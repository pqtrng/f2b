import helper
import tensorflow as tf


def create_current_output_network():
    """Create output network as current defined.

    Returns:
        model: Output model
    """
    model = tf.keras.layers.Dense(1)
    return model


def create_output_network(network_type):
    """Create output network base on given network type.

    Args:
        network_type (str): name of network type

    Raises:
        ValueError: Raise when unknown network type are given

    Returns:
        model: Output network
    """
    if network_type == "current":
        return create_current_output_network()
    else:
        raise ValueError("Unknown network output type!")


def create_base_network(network_name="resnet_50"):
    """Create base network.

    Args:
        network_name (str, optional): Name of base network. Defaults to "resnet_50".

    Raises:
        ValueError: Raise if  unknown base network type

    Returns:
        model: Base network
    """
    if network_name == "resnet_50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            pooling="avg",
            weights="imagenet",
        )
    else:
        raise ValueError("Unknown base network name.")

    return base_model, len(base_model.layers)


def combine_model(output_network, base_network):
    """Combine base network and output network to create a complete model.

    Args:
        output_network (model): prediction part
        base_network (model): feature extraction part

    Returns:
        model: Complete model
    """
    prediction = output_network(base_network.output)
    model = tf.keras.models.Model(inputs=base_network.input, outputs=prediction)
    return model


def get_model(
    base_network_type="resnet_50", output_network_type="current", training_type="top"
):
    """Create a model from type of network and training.

    Args:
        base_network_type (str, optional): Type of base network. Defaults to "resnet_50".
        output_network_type (str, optional): Type of output network. Defaults to "current".
        training_type (str, optional): Training type, can be 'all' or 'top'. Defaults to "top".

    Returns:
        model: a complete model to predict BMI
    """
    output_network = create_output_network(network_type=output_network_type)
    base_network, num_of_layers = create_base_network(network_name=base_network_type)
    model = combine_model(output_network=output_network, base_network=base_network)
    helper.set_training_type_for_model(
        model=model,
        training_type=training_type,
        num_of_untrained_layers=num_of_layers,
    )

    return model


if __name__ == "__main__":
    model = get_model(
        base_network_type="resnet_50",
        output_network_type="current",
        training_type="top",
    )
    model.summary()
