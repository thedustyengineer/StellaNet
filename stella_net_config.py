## @package stella_net_config
# File contains StellaNet configuration information

import distutils
import os
import logging



logger = logging.getLogger('stella_net')
logger.setLevel(logging.DEBUG)

# create a file handler
fh = logging.FileHandler('stella_net.log') 
fh.setLevel(logging.DEBUG)

# create a console handler
ch = logging.StreamHandler() 
ch.setLevel(logging.ERROR) 

# define and set the logging formatter
log_format = logging.Formatter('%(name)s :: %(levelname)s - %(asctime)s - %(message)s') 
fh.setFormatter(log_format)
ch.setFormatter(log_format)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
