"""
Initialize Bluemist-AI's environment
"""

__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import logging
import os
import shutil
import sysconfig
from logging import config
import platform
from termcolor import colored


os.environ["BLUEMIST_PATH"] = os.path.realpath(os.path.dirname(__file__))
BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")

os.chdir(BLUEMIST_PATH)

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")
logging.captureWarnings(True)
logger.info('BLUEMIST_PATH {}'.format(BLUEMIST_PATH))


def initialize(
        log_level='DEBUG',
        cleanup_resources=True
):
    """
    log_level : {'CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG'}, default='INFO'
        control logging level for bluemist.log

    cleanup_resources : {True, False}, default=True
        cleanup artifacts from previous runs
    """

    if log_level.upper() in ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG']:
        logger.setLevel(logging.getLevelName(log_level))

    logger.handlers[1].doRollover()

    banner = """
    ██████╗ ██╗     ██╗   ██╗███████╗███╗   ███╗██╗███████╗████████╗               █████╗ ██╗
    ██╔══██╗██║     ██║   ██║██╔════╝████╗ ████║██║██╔════╝╚══██╔══╝              ██╔══██╗██║
    ██████╔╝██║     ██║   ██║█████╗  ██╔████╔██║██║███████╗   ██║       █████╗    ███████║██║
    ██╔══██╗██║     ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚════██║   ██║       ╚════╝    ██╔══██║██║
    ██████╔╝███████╗╚██████╔╝███████╗██║ ╚═╝ ██║██║███████║   ██║                 ██║  ██║██║
    ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚══════╝   ╚═╝                 ╚═╝  ╚═╝╚═╝                                                                                           
    (version 0.1.1)
    """

    print(colored(banner, 'blue'))
    logger.info('\n{}'.format(banner))

    print('Bluemist path :: {}'.format(BLUEMIST_PATH))
    print('System platform :: {}, {}, {}, {}'.format(os.name, platform.release(), sysconfig.get_platform(),
                                                     platform.architecture()))
    logger.info('System platform :: {}, {}, {}, {}'.format(os.name, platform.release(), sysconfig.get_platform(),
                                                           platform.architecture()))

    logger.debug('Printing environment variables...')
    for key, value in os.environ.items():
        logger.debug(f'{key}={value}')

    # cleaning and building artifacts directory
    if bool(cleanup_resources):
        directories = ['data', 'eda', 'experiments', 'models', 'preprocessor']
        for directory in directories:
            directory_path = BLUEMIST_PATH + '/artifacts/' + directory
            if os.path.exists(directory_path):
                logger.debug('Removing directory :: {}'.format(directory_path))
                shutil.rmtree(directory_path)
            if not os.path.exists(directory_path):
                logger.debug('Creating directory :: {}'.format(directory_path))
                os.mkdir(directory_path)

        with open(BLUEMIST_PATH + '/' + 'artifacts/api/predict.py', 'w') as f:
            logger.debug('Clearing file content of {}'.format(BLUEMIST_PATH + '/' + 'artifacts/api/predict.py'))
            f.truncate()
