# Automatic License Plate Detection in Python
*Developed by Anne-Sophie Bollmann, Susanne Klöcker, Pia von Kolken and Christian Peters*

We used Tensorflow, OpenCV and Tesseract to build an automatic license plate detection system in python.

## Usage

Open a command prompt, navigate to the `src` directory and type the following:
```sh
python license_plate_detection.py --visualize [path to .jpg image]
```

You will see something like this:

![The license plate of a car has been successfully recognized.](https://raw.githubusercontent.com/cxan96/license_plate_detection/main/demo.png)

## Installation

This project requires python 3.8.7.

1. Clone this repository to your local machine:
    ```sh
    git clone https://github.com/cxan96/license_plate_detection.git
    ```

2. Navigate into the cloned repository and create a virtual environment:
    ```sh
    cd license_plate_detection/
    python -m venv venv
    ./venv/scripts/activate
    ```

3. Install the project requirements:
    ```sh
    pip install -r requirements.txt
    ```

### Notes on Tesseract

This project conatins a full Tesseract installation for Windows 10.
If you are using Linux or Mac, you have to install Tesseract yourself and
set the following line in the file `src/character_recognition/ocr_pipeline.py`:
```python
pytesseract.pytesseract.tesseract_cmd = "[path to tesseract.exe]"
```