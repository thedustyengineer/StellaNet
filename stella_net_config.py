"""@package docstring
File contains StellaNet configuration information
"""

import distutils
import os
import logging


class logging_config:
    def __init__(self):
        logger = logging.getLogger('stella_net')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('stella_net.log') # create a file handler
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler() # create a console handler
        ch.setLevel(logging.ERROR) 

        log_format = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s')
