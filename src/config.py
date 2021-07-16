import os
import pathlib

TEST_RATIO = 0.22
VALID_RATIO = 0.05


class Config:
    """All configurations for project."""

    # Root directory
    root = pathlib.Path(__file__).absolute().parent.parent

    # Path to data
    data_path = os.path.join(root, "data")
    raw_data_path = os.path.join(data_path, "raw")
    interim_data_path = os.path.join(data_path, "interim")
    external_data_path = os.path.join(data_path, "external")
    processed_data_path = os.path.join(data_path, "processed")

    # Path to source code
    src_path = os.path.join(root, "src")

    # Path to model
    model_path = os.path.join(root, "models")
    pretrained_model_path = os.path.join(model_path, "pretrained")
    trained_model_path = os.path.join(model_path, "trained")
    checkpoint_model_path = os.path.join(model_path, "checkpoint")

    # Path to logs file
    log_path = os.path.join(root, "logs")
    history_path = os.path.join(log_path, "history")

    # Path to report
    report_path = os.path.join(root, "report")
    figure_path = os.path.join(report_path, "figures")

    # Other variables
    batch_size = 16
    initial_learning_rate = 1e-5
    epochs = 500
    weight_decay = 5e-4
    gamma = 1e-3
    power = 0.75
    momentum = 0.9
    useful_columns = ["height", "weight", "image-src", "bmi"]
    margin = 0.1
    image_default_size = 224
    train_test_split = 1 - TEST_RATIO  # 0.95
    train_val_split = (
        train_test_split - VALID_RATIO
    ) / train_test_split  # Check in the generator
    seed = 23  # random.randint(a=1, b=100)
    patience = 50
    min_delta = 1e-5
    metric = "mean_absolute_error"
    rescale = 1.0 / 255
    num_of_workers = 4
    x_col = "image_name"
    y_col = "bmi"
    class_mode = "raw"
    color_mode = "rgb"
    n_splits = 5
    n_repeats = 5
    default_annotation_file_name = "annotation.csv"
    default_images_directory_name = "images"
    default_path_column_name = "path"
    default_image_column_name = "name"
    number_of_frames = 24


if __name__ == "__main__":
    print(Config.root)
