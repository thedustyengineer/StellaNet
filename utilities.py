## @package utilities
# File contains Perturbation and FileOperations classes that are used to support model augmentation and training


#!/usr/bin/env python
import distutils
import os
import logging
import stella_net_config
import stella_net_exceptions
import numpy as np
import spectrum
from scipy.signal import fftconvolve
from astropy.io import fits # DOCUMENTATION HERE: http://docs.astropy.org/en/stable/io/fits/
import glob
import csv
from operator import itemgetter


# setup
logger = logging.getLogger('stella_net')

## Methods used to augment the model training data set and test sets
class Perturbations:

    ## Applies vsini broadening to the provided spectrum.Spectrum object
    # @param spectrum: a StellaNet spectrum.Spectrum object. 
    # @param vsini_value: The vsini that should be applied in km/s as a float value
    # @return the updated StellaNet spectrum.Spectrum object. 
    # @exception stella_net_exceptions.WavelengthSpacingError
    # @note
    #   Adapted from iSpec by Sergi-Blanco Cuaresma https://www.blancocuaresma.com/s/iSpec
    #   which was adapted
    #   from lsf_rotate.pro:
    #   http://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro
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

        spectrum.flux = convolved_flux # update the flux value
        spectrum.vsini_applied = True
        spectrum.vsini_value = vsini_value
        return spectrum


    ## Applies gaussian noise to the provided spectrum.Spectrum object
    # @param spectrum: a StellaNet spectrum.Spectrum object. See spectrum.Spectrum documentation for more info.
    # @param snr: The desired signal to noise ratio
    # @return the updated StellaNet spectrum.Spectrum object. 
    @staticmethod
    def apply_snr(spectrum, snr):
        spectrum.flux= spectrum.flux/max(spectrum.flux) + \
            np.random.normal(size=len(spectrum.flux),scale=1.00/float(snr))
        spectrum.noise_applied = True
        spectrum.noise_value = snr
        return spectrum

    ## Applies a radial velocity (red/blueshift) to the provided spectrum.Spectrum object
    # @param spectrum: a StellaNet spectrum.Spectrum object. See spectrum.Spectrum documentation for more info.
    # @param velocity: the desired radial velocity shift
    # @return the updated StellaNet spectrum.Spectrum object. 
    @staticmethod
    def apply_rad_vel_shift(spectrum, velocity):
        print()

## Methods used for file operations (ie reading spectrum fits files, writing files, converting files, etc)
class FileOperations:

    ## Reads a fits formatted spectrum file and outputs a spectrum.Spectrum object
    # @param file_path: the path to the file that is to be read
    # @param table_num: the table number of the .fits file containing the wavelength, flux, and error values
    # @param wave_header: the string header of the wavelength values (if wavelength obtained via CDELT1, CRVAL1, and NAXIS1 use '0')
    # @param flux_header: the string header of the flux values (use '0' if flux vals are all that is in data)
    # @param error_header: the string header of the error values (may be '0' to use zeroes)
    # @return a StellaNet spectrum.Spectrum object generated from the file that was read
    @staticmethod
    def read_spectrum(file_path, table_num, wave_header, flux_header, error_header):
        logger.info('Reading file at' + file_path)
        hdul = fits.open(file_path)          # open the current fits file
    
        if (flux_header != '0'):
            flux = np.array(hdul[table_num].data[flux_header])   # the flux
        else:
            flux = np.array(hdul[table_num].data)

        if (error_header != '0'):
            error = np.array(hdul[table_num].data[error_header])
        else:
            error = np.zeros_like(flux)

        if (wave_header != '0'):
            wave = np.array(hdul[table_num].data[wave_header])
        else:
            pix_size = hdul[table_num].header['CRPIX1']
            w_delta = hdul[table_num].header['CDELT1']
            start_value = hdul[table_num].header['CRVAL1']
            w_count = hdul[table_num].header['NAXIS1']

            wave = ((np.arange(w_count) + 1.0) - pix_size) * w_delta + start_value

        new_spectrum = spectrum.Spectrum(wave, flux, error)

        return new_spectrum

    @staticmethod
    def build_dataset_from_ispec_grid_folder(directory, label_index):
        x_train = []
        y_train = []
        for file in os.listdir(directory):
            this_spectrum = FileOperations.read_spectrum(directory + '/' + file,0,'0','0','0')
            x_train.append(this_spectrum.fluxes)
            filename = file.split('_')
            y_train.append(int(filename[label_index]))
        return x_train, y_train

#FileOperations.read_spectrum('/Users/dustin/iSpec/nn_grid_spectrum/grid/3500_2.50_-1.00_0.00_2.00_0.00_0.00_0.00.fits.gz', \
   # 0, '0', '0', '0')
       
