"""
Performs Exploratory Data Analysis (EDA)
"""

__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import logging
import os
from logging import config
from IPython.core.display_functions import display
from pandas_profiling import ProfileReport
import dtale
import sweetviz as sv


BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")
EDA_ARTIFACTS_PATH = BLUEMIST_PATH + '/' + 'artifacts/eda'

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def perform_eda(data,
                provider='pandas-profiling',
                sample_size=10000,
                data_randomizer=2):
    """
        data: pandas dataframe
            Dataframe for exploratory data analysis
        provider : {'pandas-profiling', 'sweetviz', 'dtale'}, default='pandas-profiling'
            Library provider for exploratory data analysis
        sample_size: str, default=10000
            Number of rows to return from dataframe. ``None`` to perform eda on the complete dataset which can be slower
            if dataset has large number of rows and columns
        data_randomizer: int, default=None
            Controls the data split. Provide a value to reproduce the same split.
    """

    if data.size >= sample_size:
        data = data.sample(n=sample_size, random_state=data_randomizer)

    output_file = EDA_ARTIFACTS_PATH + '/' + provider + '.html'

    valid_providers = ['pandas-profiling', 'sweetviz', 'dtale']
    if provider in valid_providers:
        logger.info('Peforming EDA using :: {}'.format(provider))
        if provider == 'pandas-profiling':
            logger.info('Output file :: {}'.format(output_file))
            display('Output file :: {}'.format(output_file))
            display('Output file will be opened in the browser after analysis is completed !!')
            profile = ProfileReport(data, explorative=True)
            profile.to_file(output_file=output_file, silent=False)
        elif provider == 'sweetviz':
            logger.info('Output file :: {}'.format(output_file))
            display('Output file :: {}'.format(output_file))
            display('Output file will be opened in the browser after analysis is completed !!')
            sweetviz_report = sv.analyze(data)
            sweetviz_report.show_html(output_file)
        elif provider == 'dtale':
            display('Opening dtale UI on the browser...')
            d = dtale.show(data, subprocess=False, reaper_on=True)
            d.open_browser()
    else:
        display('Invalid provider, valid providers are :: {}'.format(valid_providers))
        logger.info('Invalid provider, valid providers are :: {}'.format(valid_providers))
