## @package stella_net_exceptions
#
# File contains StellaNet custom exception definitions

import stella_net_config
import logging

# setup
logger = logging.getLogger('stella_net')

class ParamTooLargeError(Exception):
    ## Description: 
    #   the specified parameter was too large
    def __init__(self):
        logger.exception('the specified parameter was too large')
    pass

class ParamTooSmallError(Exception):
    ## Description: 
    #   the specified parameter was too small
    def __init__(self):
        logger.exception('the specified parameter was too small')
    pass

class WavelengthSpacingError(Exception):
    ## Description: 
    #   wavelength spacing is not homogenous
    def __init__(self):
        logger.exception('wavelength spacing is not homogenous')
    pass


