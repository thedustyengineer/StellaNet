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

class NoiseAlreadyAppliedError(Exception):
    ## Description: 
    #   noise has already been applied to the spectrum
    def __init__(self):
        logger.exception('noise has already been applied to the spectrum')
    pass

class RadVelAlreadyAppliedError(Exception):
    ## Description: 
    #   radial velocity shift has already been applied to the spectrum
    def __init__(self):
        logger.exception('radial velocity shift has already been applied to the spectrum')
    pass

class VsiniAlreadyAppliedError(Exception):
    ## Description: 
    #   vsini has already been applied to the spectrum
    def __init__(self):
        logger.exception('vsini has already been applied to the spectrum')
    pass


