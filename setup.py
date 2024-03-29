from setuptools import setup, find_packages

with open("requirements.txt") as file:
    required_packages = file.read().splitlines()

with open("requirements-optional.txt") as file:
    optional_packages = file.read().splitlines()

with open("requirements-llm.txt") as file:
    llm_packages = file.read().splitlines()

setup(
    packages=find_packages(include=["bluemist*"]),
    install_requires=required_packages,
    extras_require={
        "complete": optional_packages + llm_packages
    }
)
