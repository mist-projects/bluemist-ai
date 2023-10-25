"""
Initialize Bluemist-AI's environment
"""

# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: Feb 10, 2023
# Last modified: June 22, 2023

import logging
import os
import platform
import shutil
import sysconfig
import traceback
from logging import config

from termcolor import colored

from bluemist.utils.constants import CPU_BRAND_INTEL, GPU_BRAND_INTEL, GPU_BRAND_NVIDIA

os.environ["BLUEMIST_PATH"] = os.path.realpath(os.path.dirname(__file__))
BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")

os.chdir(BLUEMIST_PATH)

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")
logging.captureWarnings(True)
logger.info('BLUEMIST_PATH {}'.format(BLUEMIST_PATH))

available_gpu = None
available_cpu = None


def initialize(
        log_level='DEBUG',
        enable_acceleration_extensions=False,
        cleanup_resources=True
):
    """
    log_level : {'CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG'}, default='DEBUG'
        Controls the logging level for bluemist.log
    enable_acceleration_extensions : {True, False}, default=False
        - Enables NVIDIA GPU acceleration/Intel CPU acceleration based on the underlying GPU/CPU infrastructure
        - NVIDIA GPU acceleration is provided by RAPIDS cuML. For the list of supported algorithms, please refer  https://docs.rapids.ai/api/cuml/stable/api/#regression-and-classification
        - Intel CPU acceleration is provided by Intel® Extension for Scikit-learn. For the list of supported algorithms, please refer https://intel.github.io/scikit-learn-intelex/algorithms.html#on-cpu
    cleanup_resources : {True, False}, default=True
        Cleanup artifacts from previous runs
    """

    global available_gpu
    global available_cpu

    banner = """
    ██████╗ ██╗     ██╗   ██╗███████╗███╗   ███╗██╗███████╗████████╗     █████╗ ██╗
    ██╔══██╗██║     ██║   ██║██╔════╝████╗ ████║██║██╔════╝╚══██╔══╝    ██╔══██╗██║
    ██████╔╝██║     ██║   ██║█████╗  ██╔████╔██║██║███████╗   ██║       ███████║██║
    ██╔══██╗██║     ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚════██║   ██║       ██╔══██║██║
    ██████╔╝███████╗╚██████╔╝███████╗██║ ╚═╝ ██║██║███████║   ██║       ██║  ██║██║                                                                        
                        (version 0.1.3 - WordCraft)
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

    if log_level.upper() in ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG']:
        logger.setLevel(logging.getLevelName(log_level))

    logger.handlers[0].doRollover()

    if enable_acceleration_extensions:
        gpu_brand = check_gpu_brand()
        cpu_brand = check_cpu_brand()

        print("GPU Brand ::", gpu_brand)
        print("CPU Brand ::", cpu_brand)

        if gpu_brand == GPU_BRAND_NVIDIA:
            try:
                import cuml
                cuml_version = cuml.__version__
                available_gpu = gpu_brand
                print("cuML version", str(cuml_version))
                print("NVIDIA GPU support is available via RAPIDS cuML ", str(cuml_version))
            except Exception as e:
                print("NVIDIA GPU support is NOT available !")
                logger.error("Error: %s", str(e))
                logger.error(traceback.format_exc())

        if cpu_brand == CPU_BRAND_INTEL:
            try:
                from sklearnex import patch_sklearn
                patch_sklearn()
                available_cpu = cpu_brand
                print("CPU Acceleration enabled via Intel® Extension for Scikit-learn")
            except Exception as e:
                print("Intel CPU Acceleration is NOT available !")
                logger.error("Error: %s", str(e))
                logger.error(traceback.format_exc())

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
        return GPU_BRAND_NVIDIA
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check for Intel GPU
    try:
        subprocess.check_output(['intel_gpu_top', '-h'])
        return GPU_BRAND_INTEL
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return "Unknown GPU brand !!"


def check_cpu_brand():
    import cpuinfo

    cpu_brand = cpuinfo.get_cpu_info()['vendor_id_raw']  # get only the brand name
    return cpu_brand
