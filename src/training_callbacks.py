import os

import tensorflow as tf
from config import Config
from scheduler import FBLearningRateScheduler


def get_early_stopping_callback(
    metric=Config.metric, patience=Config.patience, min_delta=Config.min_delta
):
    """Return Early Stop Callback.

    Returns:
        Callback: Early stopping callback
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_" + metric,
        mode="min",
        verbose=2,
        patience=patience,
        min_delta=min_delta,
    )


def get_model_checkpoint_callback(
    checkpoint_path, training_type, dataset, metric, seed
):
    """Create model checkpoint callback.

    Args:
        checkpoint_path (str): Path to save model checkpoint
        training_type (str): type of string
        dataset (str): name of dataset
        metric (str): target training metric
        seed (int): random number for reproduce experiment

    Returns:
        Callback: Model Checkpoint callback
    """
    name_of_checkpoint = f"type:{training_type}-data:{dataset}-seed:{seed}-"
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_path,
            name_of_checkpoint
            + "epoch:{epoch:04d}-metric:{val_mean_absolute_error:.6f}.h5",
        ),
        monitor="val_" + metric,
        mode="min",
        verbose=1,
        save_best_only=True,
    )


def get_learning_rate_callback(
    model,
    initial_learning_rate=Config.initial_learning_rate,
    gamma=Config.gamma,
    power=Config.power,
    verbose=True,
):
    """Create learning rate scheduler callback.

    Args:
        model (Keras Model): Model will be trained with this callback
        initial_learning_rate (float, optional): Initial value for learning rate. Defaults to Config.initial_learning_rate.
        gamma (float, optional): Initial value for gamma. Defaults to Config.gamma.
        power (float, optional): Initial value for power. Defaults to Config.power.
        verbose (bool, optional): Should show detail or not. Defaults to True.

    Returns:
        Callback: Learning Rate Scheduler Callback
    """
    return FBLearningRateScheduler(
        model=model,
        initial_learning_rate=initial_learning_rate,
        gamma=gamma,
        power=power,
        verbose=verbose,
    )


def get_tensorboard_callback(log_dir):
    """Create tensorboard callback.

    Args:
        log_dir (str): path for Tensorboard logs

    Returns:
        Callback: Tensorboard callback
    """
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def get_callbacks(
    model,
    checkpoint_path,
    tensorboard_log_dir,
    training_type="top",
    dataset="original",
    metric=Config.metric,
    smoke_test=True,
    seed=Config.seed,
):
    """Create all callbacks for training.

    Args:
        model (Keras Model): Model for training
        checkpoint_path (str): Path for saving checkpoint model
        tensorboard_log_dir (str): Path for Tensorboard logging
        training_type (str, optional): Type of training. Defaults to "top".
        dataset (str, optional): Name of dataset. Defaults to "original".
        smoke_test (bool, optional): Whether for test running code. Defaults to True.
        seed (int, optional): Random number for re-producing experiment. Defaults to Config.seed.

    Returns:
        list: a list of all callbacks
    """
    training_callbacks = []
    training_callbacks.append(get_early_stopping_callback())
    training_callbacks.append(
        get_model_checkpoint_callback(
            checkpoint_path=checkpoint_path,
            training_type=training_type,
            dataset=dataset,
            metric=metric,
            seed=seed,
        )
    )

    training_callbacks.append(
        get_learning_rate_callback(
            model=model,
            initial_learning_rate=Config.initial_learning_rate,
            gamma=Config.gamma,
            power=Config.power,
            verbose=smoke_test,
        )
    )

    training_callbacks.append(get_tensorboard_callback(log_dir=tensorboard_log_dir))

    return training_callbacks
