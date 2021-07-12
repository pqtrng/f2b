import helper


def get_generator(dataset, purpose, seed=None, augment_func=helper.keras_augment_func):
    """Create data generator.

    Args:
        dataset (str): name of dataset to train on
        purpose (str): purpose of this dataset
        seed (int, optional): Random seed to reproduce experiment. Defaults to None.
        augment_func (function, optional): augment function for input data. Defaults to keras_augment_func.

    Returns:
        [type]: [description]
    """
    annotation_dataframe, images_dir_name = helper.get_dataset_info(
        dataset=dataset, purpose=purpose
    )
    processor = helper.get_image_processor(purpose=purpose, augment_func=augment_func)
    generator = helper.create_generator(
        dataframe=annotation_dataframe,
        img_dir=images_dir_name,
        purpose=purpose,
        processor=processor,
        seed=seed,
    )
    return generator


if __name__ == "__main__":
    dataset = "original"
    train_generator = get_generator(
        dataset=dataset, purpose="train", augment_func=helper.keras_augment_func
    )
    valid_generator = get_generator(dataset=dataset, purpose="valid")
    # test_generator = get_generator(dataset=dataset, purpose="test")
    x = train_generator.next()
    print(
        f"'{dataset}' has: \n\t{train_generator.n} entries in trainset.\n\t{valid_generator.n} entries in validset."  # \n\t{test_generator.n} entries in testset."
    )
    print(f"Each batch in {dataset} has size of: {x[0].shape}")
