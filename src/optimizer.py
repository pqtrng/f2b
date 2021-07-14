import print_dict
import tensorflow_addons as tfa
from config import Config


def get_optimizer(type="sgdw"):
    if type == "sgdw":
        return tfa.optimizers.SGDW(
            learning_rate=Config.initial_learning_rate,
            weight_decay=Config.weight_decay,
            momentum=Config.momentum,
        )
    else:
        raise ValueError("Unknown Optimizer type!")


if __name__ == "__main__":
    optim = get_optimizer(type="sgdw")
    print_dict.pd(optim.get_config())
