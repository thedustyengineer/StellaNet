"""@package docstring
File contains StellaNet custom exception definitions
"""
import stella_net_config
import logging

class ParamTooLargeError(Exception):
    """
    Description: the specified parameter was too large
    """
    def __init__(self):
        using new stella_net_config.logging_config():
            logging.exception('the specified parameter was too large')


    pass

class ParamTooSmallError(Exception):
    """
    Description: the specified parameter was too small
    """
    pass


