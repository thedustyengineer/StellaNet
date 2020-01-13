"""@package docstring
File contains Perturbation class which includes methods pertaining to perturbations of spectra
"""

#!/usr/bin/env python
import distutils
import os
import logging
import stella_net_config
import stella_net_exceptions

class Perturbation:
    """
    Methods pertaining to perturbations of spectra
    """

    def __init__(self, name):
        self.name = name

    def apply_vsini(self, file_path, vsini_value):
        """
        Args:
            name: file_path: string value
            description: The path to the .fits formatted spectra that the vsini calculations will be applied to.

            name: vsini_value: float value
            description: The vsini that should be applied in km/s.

        Returns: 
            void

        Raises:
            AssertionError, ParamTooLargeError, ParamTooSmallError

        Examples:
            self.apply_vsini("C:/Path/to/folder")
        """

        assert(os.path.isfile(file_path)) # assert we actually got a file

        # check for params in allowed bounds
        if (vsini_value < 0): raise stella_net_exceptions.ParamTooSmallError
        if (vsini_value > 500): raise stella_net_exceptions.ParamTooLargeError

        logging.info('Applying vsini perturbation with value {}'.format(vsini_value))
        
    
