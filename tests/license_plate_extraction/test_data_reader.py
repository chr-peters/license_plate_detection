from license_plate_extraction import data_reader, settings
import numpy as np


def test_read_eu_bounding_box():
    xml_path = settings.DATA_DIR / "eu_cars+lps" / "1T43213_car_eu.xml"

    result = data_reader.get_bounding_box_from_xml_path(xml_path)

    assert np.isclose(result[0], 213) == True
    assert np.isclose(result[1], 214) == True
    assert np.isclose(result[2], 314) == True
    assert np.isclose(result[3], 237) == True


def test_read_br_bounding_box():
    xml_path = settings.DATA_DIR / "br_cars+lps" / "AYO9034_car_br.xml"

    result = data_reader.get_bounding_box_from_xml_path(xml_path)

    assert np.isclose(result[0], 528) == True
    assert np.isclose(result[1], 412) == True
    assert np.isclose(result[2], 690) == True
    assert np.isclose(result[3], 464) == True


def test_read_ro_bounding_box():
    xml_path = settings.DATA_DIR / "ro_cars+lps" / "1912BG14_car_eu.xml"

    result = data_reader.get_bounding_box_from_xml_path(xml_path)

    assert np.isclose(result[0], 1400) == True
    assert np.isclose(result[1], 1144) == True
    assert np.isclose(result[2], 1875) == True
    assert np.isclose(result[3], 1293) == True


def test_read_us_bounding_box():
    xml_path = settings.DATA_DIR / "us_cars+lps" / "0SG719_car_us.xml"

    result = data_reader.get_bounding_box_from_xml_path(xml_path)

    assert np.isclose(result[0], 911) == True
    assert np.isclose(result[1], 136) == True
    assert np.isclose(result[2], 973) == True
    assert np.isclose(result[3], 167) == True


def test_eu_bounding_box_path_from_image_path():
    image_path = settings.DATA_DIR / "eu_cars+lps" / "1T43213_car_eu.jpg"

    bounding_box_path = data_reader.get_bounding_box_xml_path_from_image_path(
        image_path
    )

    assert bounding_box_path == settings.DATA_DIR / "eu_cars+lps" / "1T43213_car_eu.xml"


def test_br_bounding_box_path_from_image_path():
    image_path = settings.DATA_DIR / "br_cars+lps" / "AYO9034_car_br.jpg"

    bounding_box_path = data_reader.get_bounding_box_xml_path_from_image_path(
        image_path
    )

    assert bounding_box_path == settings.DATA_DIR / "br_cars+lps" / "AYO9034_car_br.xml"