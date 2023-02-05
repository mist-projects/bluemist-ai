import logging
import os
from logging import config

from pyfiglet import Figlet
from termcolor import colored
from IPython.display import display, HTML

os.environ["BLUEMIST_PATH"] = os.path.realpath(os.path.dirname(__file__))
BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")

os.chdir(BLUEMIST_PATH)

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")

logger.info('BLUEMIST_PATH {}'.format(BLUEMIST_PATH))

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

    logger.handlers[0].doRollover()

    figlet_banner = Figlet(font='small')
    figlet_version = Figlet(font='digital')

    banner = """
    ██████╗ ██╗     ██╗   ██╗███████╗███╗   ███╗██╗███████╗████████╗               █████╗ ██╗
    ██╔══██╗██║     ██║   ██║██╔════╝████╗ ████║██║██╔════╝╚══██╔══╝              ██╔══██╗██║
    ██████╔╝██║     ██║   ██║█████╗  ██╔████╔██║██║███████╗   ██║       █████╗    ███████║██║
    ██╔══██╗██║     ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚════██║   ██║       ╚════╝    ██╔══██║██║
    ██████╔╝███████╗╚██████╔╝███████╗██║ ╚═╝ ██║██║███████║   ██║                 ██║  ██║██║
    ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚══════╝   ╚═╝                 ╚═╝  ╚═╝╚═╝                                                                                           
    """

    print(colored(banner + '\n>>>> version 0.1.1 <<<<', 'blue'))
    #logger.info('\n{}{}'.format(banner, version))

    logger.info('Printing all environment variables')
    for key, value in os.environ.items():
        logger.info(f'{key}={value}')
