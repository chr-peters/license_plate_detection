from data_reader import get_image_paths_from_directory, make_dataset_from_image_paths
from visualization_tools import show_image
from preprocessing import bounding_box_in_pixel
import settings
import tensorflow as tf
from tensorflow.keras import losses, metrics
import numpy as np
import datetime
import models


if __name__ == "__main__":
    image_directory_eu = settings.DATA_DIR / "eu_cars+lps"
    image_directory_br = settings.DATA_DIR / "br_cars+lps"
    image_directory_us = settings.DATA_DIR / "us_cars+lps"
    image_directory_ro = settings.DATA_DIR / "ro_cars+lps"

    # make a list with all the image paths
    all_image_paths = np.array(
        [
            *get_image_paths_from_directory(image_directory_eu),
            *get_image_paths_from_directory(image_directory_br),
            *get_image_paths_from_directory(image_directory_us),
            *get_image_paths_from_directory(image_directory_ro),
        ]
    )

    num_images = len(all_image_paths)

    print(f"NUM_IMAGES = {num_images}")

    # randomly split the list into training and test set
    TRAINING_SET_SIZE = int(num_images * 0.8)
    train_indices = np.random.choice(num_images, TRAINING_SET_SIZE, replace=False)

    image_path_list_train = all_image_paths[train_indices]
    image_path_list_test = np.delete(all_image_paths, train_indices)

    # read the images from the paths to create the training set
    TARGET_IMG_HEIGHT = 448
    TARGET_IMG_WIDTH = 448
    dataset_train = make_dataset_from_image_paths(
        image_path_list_train,
        target_img_height=TARGET_IMG_HEIGHT,
        target_img_width=TARGET_IMG_WIDTH,
    )

    # set the batch size
    BATCH_SIZE = 32
    dataset_train = dataset_train.batch(BATCH_SIZE)

    # do the same for the test set
    dataset_test = make_dataset_from_image_paths(
        image_path_list_test,
        target_img_height=TARGET_IMG_HEIGHT,
        target_img_width=TARGET_IMG_WIDTH,
    )
    dataset_test = dataset_test.batch(BATCH_SIZE)

    # get the model
    model = models.tiny_yolo_inspired(
        input_height=TARGET_IMG_HEIGHT, input_width=TARGET_IMG_WIDTH, num_channels=3
    )

    print(model.summary())

    model.compile(optimizer="adam", loss=losses.MSE)

    # add logs and callback for tensorboard
    log_dir = settings.LOG_DIR / (
        "training_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    model.fit(
        dataset_train,
        epochs=100,
        validation_data=dataset_test,
        callbacks=[tensorboard_callback],
    )

    # visualize the test set predictions
    example_list = list(dataset_test.as_numpy_iterator())
    for cur_example in example_list:
        cur_image_batch = cur_example[0]
        cur_prediction_batch = model.predict(cur_image_batch)
        for cur_image, cur_prediction in zip(cur_image_batch, cur_prediction_batch):
            print(cur_prediction)
            show_image(
                cur_image.astype(int),
                bounding_box=bounding_box_in_pixel(
                    cur_prediction,
                    img_height=TARGET_IMG_HEIGHT,
                    img_width=TARGET_IMG_WIDTH,
                ),
            )
