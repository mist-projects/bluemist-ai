
__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import logging
import os
from logging import config

from pandas_profiling import ProfileReport
import dtale
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()

BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")
EDA_ARTIFACTS_PATH = BLUEMIST_PATH + '/' + 'artifacts/eda'

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def perform_eda(data,
                provider='pandas-profiling',
                target_variable=None,
                sample_size=10000,
                data_randomizer=2):
    """
        Performs Exploratory Data Analysis (EDA)

        data: pandas dataframe
            Dataframe for exploratory data analysis
        provider : {'pandas-profiling', 'sweetviz', 'dtale', 'autoviz'}, default='pandas-profiling'
            Library provider for exploratory data analysis
        sample_size: str, default=10000
            Number of rows to return from dataframe. ``None`` to perform eda on the complete dataset which can be slower
            if dataset has large number of rows and columns
        data_randomizer: int, default=None
            Controls the data split. Provide a value to reproduce the same split.

        Examples
        ---------
        *EDA using Pandas Profiling*

        .. raw:: html
           :file: ../../code_samples/quickstarts/eda/eda_pandas-profiling.html

        *EDA using SweetVIZ*

        .. raw:: html
           :file: ../../code_samples/quickstarts/eda/eda_sweetviz.html

        *EDA using D-TALE*

        .. raw:: html
           :file: ../../code_samples/quickstarts/eda/eda_dtale.html

        *EDA using AutoViz*

        .. raw:: html
           :file: ../../code_samples/quickstarts/eda/eda_autoviz.html

    """

    if sample_size is not None and data.shape[0] >= sample_size:
        data = data.sample(n=sample_size, random_state=data_randomizer)

    output_provider = EDA_ARTIFACTS_PATH + '/' + provider
    output_file = output_provider + '.html'

    valid_providers = ['pandas-profiling', 'sweetviz', 'dtale', 'autoviz']
    if provider in valid_providers:
        logger.info('Peforming EDA using :: {}'.format(provider))
        if provider == 'pandas-profiling':
            logger.info('Output file :: {}'.format(output_file))
            print('Output file :: {}'.format(output_file))
            print('Output file will be opened in the browser after analysis is completed !!')
            profile = ProfileReport(data, explorative=True)
            profile.to_file(output_file=output_file, silent=False)
        elif provider == 'sweetviz':
            logger.info('Output file :: {}'.format(output_file))
            print('Output file :: {}'.format(output_file))
            print('Output file will be opened in the browser after analysis is completed !!')
            sweetviz_report = sv.analyze(data)
            sweetviz_report.show_html(output_file)
        elif provider == 'dtale':
            print('Opening dtale UI on the browser...')
            d = dtale.show(data, subprocess=False, reaper_on=True)
            d.open_browser()
        elif provider == 'autoviz':
            dftc = AV.AutoViz(filename='', sep=',', depVar=target_variable, dfte=data, header=0, verbose=2,
                              lowess=False, chart_format='html', max_rows_analyzed=sample_size,
                              save_plot_dir=output_provider)
            print('Output files stored under the path :: ' + output_provider)
    else:
        print('Invalid provider, valid providers are :: {}'.format(valid_providers))
        logger.info('Invalid provider, valid providers are :: {}'.format(valid_providers))
