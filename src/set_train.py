import helper
import optimizer
import tensorflow as tf
import training_callbacks
from config import Config


def set_train(
    model,
    train_data,
    valid_data,
    dataset,
    output_network_type,
    training_type,
    smoke_test,
    seed,
):
    if smoke_test:
        print("\n\n--------------------------")
        print("Running smoke test")
        print("--------------------------\n\n")

    # Compile model
    helper.compile_model(
        model=model,
        loss=tf.keras.losses.Huber(),
        optimizer=optimizer.get_optimizer(type="sgdw"),
        metrics=Config.metric,
    )

    # Set training, logging path
    checkpoint_path, tensorboard_log_dir, model_path = helper.create_train_log_path(
        training_type=training_type,
        dataset=dataset,
        output_network_type=output_network_type,
    )

    training_callbacks.get_callbacks()


if __name__ == "__main__":
    pass
