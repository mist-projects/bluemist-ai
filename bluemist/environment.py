"""
Initialize Bluemist-AI's environment
"""

__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"

import logging
import os
import platform
import shutil
import sysconfig
from logging import config

from termcolor import colored

os.environ["BLUEMIST_PATH"] = os.path.realpath(os.path.dirname(__file__))
BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")

os.chdir(BLUEMIST_PATH)

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")
logging.captureWarnings(True)
logger.info('BLUEMIST_PATH {}'.format(BLUEMIST_PATH))

gpu_support = False


def initialize(
        log_level='DEBUG',
        enable_gpu_support=False,
        cleanup_resources=True
):
    """
    log_level : {'CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG'}, default='INFO'
        control logging level for bluemist.log

    cleanup_resources : {True, False}, default=True
        cleanup artifacts from previous runs
    """

    global gpu_support
    gpu_support = enable_gpu_support

    if gpu_support:
        gpu_brand = check_gpu_brand()
        print("gpu_brand :"+ gpu_brand)
        if gpu_brand == "Intel":
            from sklearnex import patch_sklearn;
            patch_sklearn()
            print("GPU support is enabled !!")

    if log_level.upper() in ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG']:
        logger.setLevel(logging.getLevelName(log_level))

    logger.handlers[0].doRollover()

    banner = """
██████╗ ██╗     ██╗   ██╗███████╗███╗   ███╗██╗███████╗████████╗     █████╗ ██╗
██╔══██╗██║     ██║   ██║██╔════╝████╗ ████║██║██╔════╝╚══██╔══╝    ██╔══██╗██║
██████╔╝██║     ██║   ██║█████╗  ██╔████╔██║██║███████╗   ██║       ███████║██║
██╔══██╗██║     ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚════██║   ██║       ██╔══██║██║
██████╔╝███████╗╚██████╔╝███████╗██║ ╚═╝ ██║██║███████║   ██║       ██║  ██║██║                                                                        
                                (version 0.1.1)
    """

    print(colored(banner, 'blue'))

    print('Bluemist path :: {}'.format(BLUEMIST_PATH))
    print('System platform :: {}, {}, {}, {}, {}'.format(os.name, platform.system(), platform.release(),
                                                         sysconfig.get_platform(),
                                                         platform.architecture()))
    logger.info('System platform :: {}, {}, {}, {}, {}'.format(os.name, platform.system(), platform.release(),
                                                               sysconfig.get_platform(),
                                                               platform.architecture()))

    logger.debug('Printing environment variables...')
    for key, value in os.environ.items():
        logger.debug(f'{key}={value}')

    # cleaning and building artifacts directory
    if bool(cleanup_resources):
        directories = ['artifacts/data', 'artifacts/eda', 'artifacts/experiments', 'artifacts/models',
                       'artifacts/preprocessor', 'mlruns']
        for directory in directories:
            directory_path = BLUEMIST_PATH + '/' + directory
            if os.path.exists(directory_path):
                logger.debug('Removing directory :: {}'.format(directory_path))
                shutil.rmtree(directory_path)
            if not os.path.exists(directory_path):
                logger.debug('Creating directory :: {}'.format(directory_path))
                os.mkdir(directory_path)

        with open(BLUEMIST_PATH + '/' + 'artifacts/api/predict.py', 'w') as f:
            logger.debug('Clearing file content of {}'.format(BLUEMIST_PATH + '/' + 'artifacts/api/predict.py'))
            f.truncate()


def check_gpu_brand():
    import subprocess

    # Check for NVIDIA GPU
    try:
        subprocess.check_output(['nvidia-smi', '--help'])
        return "NVIDIA"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check for Intel GPU
    try:
        subprocess.check_output(['intel_gpu_top', '-h'])
        return "Intel"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return "Unknown GPU brand"
