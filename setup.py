from setuptools import setup, find_packages

with open("requirements.txt") as file:
    required_packages = file.read().splitlines()

setup(
    packages=find_packages(include=["bluemist*"]),
    install_requires=required_packages
)
