## @package perturbations
# File contains Perturbation class which includes methods pertaining to perturbations of spectra


#!/usr/bin/env python
import distutils
import os
import logging
import stella_net_config
import stella_net_exceptions
import numpy as np
from scipy.signal import fftconvolve

# setup
logger = logging.getLogger('stella_net')

class Perturbations:

    ## Applies vsini broadening to the provided spectrum.Spectrum object
    #
    # @param spectrum: a StellaNet spectrum.Spectrum object. See spectrum.Spectrum documentation for more info.
    #
    # @param vsini_value: The vsini that should be applied in km/s as a float value
    #
    # @return void
    #
    # @exception AssertionError 
    # @exception stella_net_exceptions.WavelengthSpacingError
    #
    # @note
    #
    #   Adapted from iSpec by Sergi-Blanco Cuaresma https://www.blancocuaresma.com/s/iSpec
    #   which was adapted
    #   from lsf_rotate.pro:
    #   http://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro
    #
    #   which was adapted from rotin3.f in the SYNSPEC software of Hubeny & Lanz
    #   http://nova.astro.umd.edu/index.html    Also see Eq. 17.12 in
    #   "The Observation and Analysis of Stellar Photospheres" by D. Gray (1992)
    @staticmethod
    def apply_vsini(spectrum, vsini_value):
        
        logger.info('applying vsini perturbation with value {}'.format(vsini_value))

        if (vsini_value <= 0): # can't apply a 0 or negative vsini
            raise stella_net_exceptions.ParamTooSmallError

        # check homogeneity of wavelength values and if homogenous assign deltav value
        for wave_index in range (1, len(spectrum.wavelengths)):
            if (wave_index != len(spectrum.wavelengths)):
                current_wavelength = spectrum.wavelengths[wave_index]
                previous_wavelength = spectrum.wavelengths[wave_index - 1]
                next_wavelength = spectrum.wavelengths[wave_index + 1]
                if not ((next_wavelength - current_wavelength) == (current_wavelength - previous_wavelength)):
                    raise stella_net_exceptions.WavelengthSpacingError
            else:
                deltav = current_wavelength - previous_wavelength
        
        epsilon = 0.6 # LDC value is typically taken to be 0.6 for the photosphere
        e1 = 2.0*(1.0 - epsilon)
        e2 = np.pi*epsilon/2.0
        e3 = np.pi*(1.0 - epsilon/3.0)

        npts = np.ceil(2*vsini_value/deltav)
        if npts % 2 == 0:
            npts += 1
        nwid = np.floor(npts/2)
        x = np.arange(npts) - nwid
        x = x*deltav/vsini_value
        x1 = np.abs(1.0 - x**2)

        velgrid = x*vsini_value
        kernel_x, kernel_y = velgrid, (e1*np.sqrt(x1) + e2*x1)/e3 # wavelengths, fluxes
        kernel_y /= kernel_y.sum()
        convolved_flux = 1 - fftconvolve(1-spectrum.flux, kernel_y, mode='same')

        spectrum.flux = convolved_flux

        return spectrum


    
    
