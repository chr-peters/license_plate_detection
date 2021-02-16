"""
This file contains the code that was used to generate
the plots shown in the presentation.
"""
import tensorflow as tf
import cv2
from visualization_tools import show_image
import preprocessing
import data_reader
import settings

if __name__ == "__main__":
    # first, show how an image is scaled to 400x400 (this is the neural net input)
    # image_path = settings.DATA_DIR / "eu_cars+lps" / "1T43213_car_eu.jpg"
    image_path = settings.DATA_DIR / "no_labels" / "audi.jpg"

    # read the image using tensorflow
    image_tensor = data_reader.read_image_as_tensor(image_path)

    # show how it is resized to match 400x400
    image_tensor_resized = tf.image.resize(
        image_tensor, size=(400, 400), antialias=True
    )
    show_image(image_tensor_resized.numpy().astype(int))

    # next, read the bounding box
    # bounding_box = data_reader.get_bounding_box_from_xml_path(
    #     data_reader.get_bounding_box_xml_path_from_image_path(image_path)
    # )

    # # scale it first, so that it matches the 400x400 dimension
    # bounding_box_scaled = preprocessing.scale_bounding_box(
    #     bounding_box,
    #     img_height=image_tensor.shape[0],
    #     img_width=image_tensor.shape[1],
    #     target_img_height=400,
    #     target_img_width=400,
    # )

    # # get the binary mask and show it
    # bounding_box_mask = preprocessing.bounding_box_to_binary_mask(
    #     bounding_box_scaled, img_height=400, img_width=400
    # )
    # show_image(bounding_box_mask)

    # Now it's time to make a prediction.
    # To do so, we first load the model.
    model = tf.keras.models.load_model(settings.MODELS_DIR / "masks_full_data.tf")

    # make a prediction (we first have to put the image into a batch for the model to work properly)
    image_tensor_batch = tf.reshape(image_tensor_resized, shape=(1, 400, 400, 3))
    cur_prediction = model(image_tensor_batch)[0].numpy()
    show_image(cur_prediction[:, :, 0])

    # use thresholding and contour finding to extract the surrounding rectangle:
    # (the contour finding part is implemented in the mask_to_bounding_box function inside
    #  of the preprocessing module)
    ret, thresh = cv2.threshold(
        (cur_prediction * 255).astype("uint8"), int(255 * 0.5), 255, 0
    )
    show_image(thresh)

    predicted_bounding_box = preprocessing.mask_to_bounding_box(cur_prediction)

    # show the rectangle that was found as well as the original image with bounding box
    show_image(thresh, predicted_bounding_box)

    show_image(image_tensor_resized.numpy().astype(int), predicted_bounding_box)
