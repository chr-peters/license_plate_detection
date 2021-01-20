from setuptools import setup, find_packages

setup(
    name="license_plate_detection",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
