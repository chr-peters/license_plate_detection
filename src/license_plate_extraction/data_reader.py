from license_plate_extraction import settings
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path


def get_bounding_box_from_xml_path(path: Path) -> np.ndarray:
    """
    Returns np.array([x_min, y_min, x_max, y_max])
    """
    with open(path, mode="r") as file:
        bs = BeautifulSoup(file, "xml")

    x_min = float(bs.bndbox.xmin.string)
    x_max = float(bs.bndbox.xmax.string)
    y_min = float(bs.bndbox.ymin.string)
    y_max = float(bs.bndbox.ymax.string)

    return np.array([x_min, y_min, x_max, y_max])
