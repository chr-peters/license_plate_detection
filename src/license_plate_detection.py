import click
from pathlib import Path
from prediction_pipeline import make_prediction
from license_plate_extraction.data_reader import read_image_as_tensor
from license_plate_extraction.preprocessing import bounding_box_in_pixel
from license_plate_extraction.visualization_tools import show_image


@click.command()
@click.option(
    "--visualize",
    default=False,
    is_flag=True,
    help="Visualize the prediction using matplotlib.",
)
@click.argument("path", type=click.Path(exists=True, resolve_path=True))
def main(visualize, path):
    """
    Print the license plate of a car detected in an image.

    PATH refers to the location of a JPEG image or a directory of JPEG images.
    If PATH is a directory, a file will be created that contains all the predicted license plates.
    """
    path = Path(path)

    if path.is_file():
        bounding_box, prediction = make_prediction(path)
        click.echo(prediction)

        if visualize:
            image = read_image_as_tensor(path)
            bounding_box_pixel = bounding_box_in_pixel(
                bounding_box, img_height=image.shape[0], img_width=image.shape[1]
            )

            show_image(
                image,
                bounding_box_pixel,
                prediction,
            )

    elif path.is_dir():
        pass
    else:
        click.echo("PATH must be either a file or a directory!", err="True")


if __name__ == "__main__":
    main()
