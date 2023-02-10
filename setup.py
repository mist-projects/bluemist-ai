from setuptools import setup, find_packages

with open("requirements.txt") as file:
    required_packages = file.read().splitlines()

setup(
    name='bluemist-ai',
    long_description='Low code python library',
    version='0.1.1.15',
    license='MIT',
    author="Shashank Agrawal",
    author_email='shashanka89@gmail.com',
    packages=find_packages(include=["bluemist*"]),
    include_package_data=True,
    url='https://github.com/shashanka89/bluemist-ai',
    install_requires=required_packages
)
