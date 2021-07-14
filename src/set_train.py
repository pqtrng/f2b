import helper
import optimizer
import print_dict
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

    # Create callbacks
    callbacks = training_callbacks.get_callbacks(
        model=model,
        checkpoint_path=checkpoint_path,
        tensorboard_log_dir=tensorboard_log_dir,
        training_type=training_type,
        dataset=dataset,
        metric=Config.metric,
        smoke_test=smoke_test,
        seed=seed,
    )

    # Train
    history = model.fit(
        x=train_data,
        batch_size=Config.batch_size,
        epochs=2 if smoke_test else Config.epochs,
        verbose=1,
        callbacks=callbacks,
        steps_per_epoch=train_data.n // train_data.batch_size,
        validation_data=valid_data,
        workers=Config.num_of_workers,
    )

    # Save best model
    time_slot, metric = helper.save_trained_model(
        training_type=training_type,
        dataset=dataset,
        output_network_type=output_network_type,
    )

    # Save logs
    helper.save_training_log(
        time_slot=time_slot,
        training_type=training_type,
        dataset=dataset,
        output_network_type=output_network_type,
        metric=metric,
        history=history,
        save_path=model_path,
    )

    if smoke_test:
        print()
        print("\n\n--------------------------")
        print("Smoke test sample result:")
        print_dict.pd(arg=history.history)
        print("--------------------------\n\n")
    else:
        helper.clean_up_dir(path_to_dir=checkpoint_path)


if __name__ == "__main__":
    pass
