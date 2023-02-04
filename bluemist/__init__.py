import logging
import os
from logging import config

from pyfiglet import Figlet
from termcolor import colored

os.environ["HOME_PATH"] = os.path.abspath('../bluemist')
HOME_PATH = os.getenv("HOME_PATH")

os.environ["ARTIFACT_PATH"] = HOME_PATH + '/' + 'artifacts'
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")

config.fileConfig(HOME_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")

logger.info('HOME_PATH {}'.format(HOME_PATH))
logger.info('ARTIFACT_PATH {}'.format(ARTIFACT_PATH))


def initialize(
        log_level='INFO',
        banner_color='blue',
        artifact_path=None
):
    """
    log_level : {'CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG'}, default='INFO'
        control logging level for bluemist.log
    banner_color : {red, green, yellow, blue, magenta, cyan, white}, default='blue'
        color of Bluemist AI banner
    artifact_path: str, default=None
        Future use. filesystem path where Bluemist AI artifacts will be stored
    """

    if log_level.upper() in ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG']:
        logger.setLevel(logging.getLevelName(log_level))

    logger.handlers[1].doRollover()

    figlet_banner = Figlet(font='small')
    figlet_version = Figlet(font='digital')

    banner = colored(figlet_banner.renderText('B l u e  m i s t - AI'), banner_color)
    version = colored(figlet_version.renderText('0.1.1'), banner_color)
    logger.info('\n{}{}'.format(banner, version))

    logger.info('Printing all environment variables')
    for key, value in os.environ.items():
        logger.info(f'{key}={value}')
