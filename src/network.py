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
    return base_model


def get_model(output_network, base_network):
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


if __name__ == "__main__":
    base_model = create_base_network(network_name="resnet_50")
    output_model = create_output_network(network_type="current")
    model = get_model(output_network=output_model, base_network=base_model)
    model.summary()
