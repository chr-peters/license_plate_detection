import click
from pathlib import Path


@click.command()
@click.argument("path", type=click.Path(exists=True, resolve_path=True))
def main(path):
    """
    Print the license plate of a car detected in an image.

    PATH refers to the location of a JPEG image or a directory of JPEG images.
    If PATH is a directory, a file will be created that contains all the predicted license plates.
    """
    path = Path(path)
    click.echo(path)


if __name__ == "__main__":
    main()
