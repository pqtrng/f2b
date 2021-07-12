import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class FBLearningRateScheduler(Callback):
    def __init__(self, model, initial_learning_rate, gamma, power, verbose=False):
        super(FBLearningRateScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.gamma = gamma
        self.power = power
        self.model = model
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(
            self.model.optimizer.learning_rate, self.initial_learning_rate
        )

    def on_train_batch_end(self, batch, logs=None):
        learning_rate = self.initial_learning_rate * tf.pow(
            ((batch + 1) * self.gamma + 1), -self.power
        )
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
