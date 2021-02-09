import os

# suppress tensorflows verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import click
from pathlib import Path
from prediction_pipeline import make_prediction
from license_plate_extraction.data_reader import (
    read_image_as_tensor,
    get_image_paths_from_directory,
)
from license_plate_extraction.visualization_tools import show_image


@click.command()
@click.option(
    "--visualize",
    default=False,
    is_flag=True,
    help="Visualize the prediction using matplotlib.",
)
@click.argument("path", type=click.Path(exists=True, resolve_path=True))
@click.argument("outfile", type=click.File("w"), required=False)
def main(visualize, path, outfile):
    """
    Print the license plate and it's bounding box of a car detected in an image.

    The bounding box format is [x_min, y_min, width, height] in pixel values.

    PATH refers to the location of a JPEG image or a directory of JPEG images.
    If PATH is a directory, a file will be created that contains all the predicted license plates.
    """
    path = Path(path)

    if path.is_file():
        bounding_box, prediction = make_prediction(path)
        click.echo(f"Bounding box: {bounding_box}")
        click.echo(f"Predicted plate number: {prediction}")

        if visualize:
            image = read_image_as_tensor(path)
            show_image(
                image,
                bounding_box,
                prediction,
            )

    elif path.is_dir():
        if outfile is None:
            click.echo(
                "Error: If PATH is a directory, OUTFILE must be specified! See license_plate_detection.py --help",
                err=True,
            )
            return

        image_path_list = get_image_paths_from_directory(path)
        for cur_image_path in image_path_list:
            cur_bounding_box, cur_prediction = make_prediction(cur_image_path)

            cur_image_name = cur_image_path.stem
            cur_line = f"{cur_image_name}:{cur_prediction}"

            click.echo(f"{cur_image_name}:{cur_prediction}")
            outfile.write(cur_line + "\n")
    else:
        click.echo("PATH must be either a file or a directory!", err=True)


if __name__ == "__main__":
    main()
