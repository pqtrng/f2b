import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_idx(idx, dataframe):
    """Plot a single image at index.

    Args:
        idx (Int): Index of image
        dataframe (DataFrame): Dataframe
    """
    original_image = cv2.imread(dataframe.iloc[idx].path)
    reverse_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.imshow(reverse_color)


def plot_batch_images(batch_size, dataframe):
    """Plot 16 images in the batch, along with the corresponding labels.

    Args:
        batch_size (Int): [description]
        dataframe (Dataframe): [description]
    """

    fig = plt.figure(figsize=(20, batch_size))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(4, batch_size // 4, idx + 1, xticks=[], yticks=[])
        plot_idx(idx + 505, dataframe)
        if "height" in dataframe.columns and "weight" in dataframe.columns:
            ax.set_title(
                "H:{:.1f}    W:{:.1f}    BMI:{:.2f}".format(
                    dataframe.iloc[idx + 505].height,
                    dataframe.iloc[idx + 505].weight,
                    dataframe.iloc[idx + 505].BMI,
                )
            )
        else:
            ax.set_title("BMI:{:.2f}".format(dataframe.iloc[idx].BMI))
