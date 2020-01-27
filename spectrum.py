## @package spectrum
# File contains StellaNet format spectrum class definition


import distutils
import os
import logging
import stella_net_config
import stella_net_exceptions
import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits # DOCUMENTATION HERE: http://docs.astropy.org/en/stable/io/fits/
import glob
import csv
from operator import itemgetter
import matplotlib.pyplot as plt

logger = logging.getLogger('stella_net')

## StellaNet Spectrum object definition
class Spectrum:
   
    ## Object representing a stellar spectrum (or synthetic spectrum) and all of its associated parameters
    #
    # @param self: the object instance
    # @param wavelengths: the spectrum wavelength values in Angstroms
    # @param fluxes: the normalized spectrum flux values
    # @param errors: the per-wavelength errors in the flux (normalized accordingly)
    # @param vsini_applied: (default: False) bool indicating if the spectrum has been rotationally broadened
    # @param vsini_value: (default: 0) int indicating the vsini in km/s of the spectrum
    # @param noise_applied: (default: False) bool indicating if the spectrum has been perturbed with an SNR value
    # @param noise_value: (default: 0) int indicating the SNR
    # @param radial_velocity_shift_applied: (default: False) bool indicating if the spectrum has been red or blue shifted
    # @param radial_velocity_shift: (default: 0) a float indicating the radial velocity shift applied to the spectrum in km/s
    # @param is_synthetic: (default: True) bool indicating if the spectrum is synthetic or not
    # @param snr: The desired signal to noise ratio
    def __init__(self, wavelengths, fluxes, errors, vsini_applied = False, vsini_value = 0, \
    noise_applied = False, noise_value = 0, radial_velocity_shift_applied = False, radial_velocity_shift = 0, is_synthetic = True, teff = None, mh = None, logg = None):
    
        self.wavelengths = wavelengths
        self.fluxes = fluxes
        self.errors = errors
        self.teff = teff
        self.logg = logg
        self.mh = mh
        self.noise_applied = False
        self.noise_value = noise_value
        self.radial_velocity_shift_applied = False
        self.rad_vel_value = radial_velocity_shift
        self.vsini_applied = False
        self.vsini_value = vsini_value
        

    def apply_vsini(self, vsini_value):
        
        logger.info('applying vsini perturbation with value {}'.format(vsini_value))

        if (vsini_value <= 0): # can't apply a 0 or negative vsini
            raise stella_net_exceptions.ParamTooSmallError

        if (self.vsini_applied):
            raise stella_net_exceptions.VsiniAlreadyAppliedError

        # check homogeneity of wavelength values and if homogenous assign deltav value
        for wave_index in range (1, len(self.wavelengths)):
            if (wave_index != len(self.wavelengths)-1):
                current_wavelength = self.wavelengths[wave_index]
                previous_wavelength = self.wavelengths[wave_index - 1]
                next_wavelength = self.wavelengths[wave_index + 1]
                if not ((next_wavelength - current_wavelength) == (current_wavelength - previous_wavelength)):
                    raise stella_net_exceptions.WavelengthSpacingError
            else:
                deltav = current_wavelength - previous_wavelength
        
        epsilon = 0.6

        dl = self.wavelengths[1]-self.wavelengths[0]
        l0 = (self.wavelengths[1]+self.wavelengths[0])*0.5

        dlL = l0*(vsini_value/2.99792458e5)

        # Nondimensional grid spacing
        dx = dl/dlL

        # Go out to the the grid point in which dl/dlL=1 falls
        n = np.ceil((2. - dx)/2./dx)*2. + 1.

        # The wavelength grid
        k = np.abs(np.arange(n)- np.floor(n/2))

        kernel_x = (np.arange(n)-np.floor(n/2))*dx

        # Useful constants
        dx2 = dx**2.
        c1 =2.*(1. -epsilon)/np.pi/dlL/(1. - epsilon/3.)
        c2 = 0.5*epsilon/dlL/(1. - epsilon/3.)

        # Compute bulk of kernel
        kernel_y = c2 - c2*dx2/12. - c2*dx2*k**2. + \
                c1/8. * (     np.sqrt(4. - dx2*(1. - 2.*k)**2.) - \
                         2.*k*np.sqrt(4. - dx2*(1. - 2.*k)**2.) + \
                              np.sqrt(4. - dx2*(1. + 2.*k)**2.) + \
                         2.*k*np.sqrt(4. - dx2*(1. + 2.*k)**2.) - \
                         4.*np.arcsin(dx*(k-0.5))/dx + 4.*np.arcsin(dx*(k+0.5))/dx)

        ## Central point
        kernel_y[int(np.floor(n/2.))] = c2 - (c2*dx2)/12. + \
                       c1*np.sqrt(4. - dx2)/4. + c1*np.arcsin(dx/2.)/dx

        # Edge points
        kernel_y[0] = 1./24./dx*(3.*c1*dx*np.sqrt(4. - dx2*(1. -2.*k[0])**2.)*(1. -2.*k[0]) + \
                          c2*(2. + dx - 2.*dx*k[0])**2.*(4. + dx*(2.*k[0]-1.)) + \
                          12.*c1*np.arccos(dx*(k[0]-.5)))
        kernel_y[0] *= (1. - (k[0]-0.5)*dx)/dx  # Edge point flux compensation
        kernel_y[int(n-1)] = kernel_y[0] # Mirror last point last

        # Integrals done as the average, compensate
        kernel_y *= dx

        # Normalize
        kernel_y /= kernel_y.sum()

        #-- convolve the flux with the kernel
        flux_conv = 1 - fftconvolve(1-self.fluxes, kernel_y, mode='same') # Fastest
        #import scipy
        #flux_conv = 1 - scipy.convolve(1-flux, kernel_y, mode='same') # Equivalent but slower
        
        self.fluxes = flux_conv # update the flux value
        self.vsini_applied = True
        self.vsini_value = vsini_value


    def apply_snr(self, snr):
        
        if self.noise_applied:
            raise stella_net_exceptions.NoiseAlreadyAppliedError

        self.fluxes= self.fluxes/max(self.fluxes) + \
            np.random.normal(size=len(self.fluxes),scale=1.00/float(snr))
        self.noise_applied = True
        self.noise_value = snr
        self.noise_applied = True
        self.noise_value = snr
        
            

    def apply_rad_vel_shift(self, velocity):

        if self.radial_velocity_shift_applied:
            raise stella_net_exceptions.RadVelAlreadyAppliedError

        c = 299792458.0 # Speed of light (m/s)
        # relativistic wavelength correction
        velocity = (velocity*1000.) # convert velocity from km/s to m/s
        self.wavelengths = self.wavelengths * np.sqrt((1.-velocity/c)/(1.+velocity/c))
        self.radial_velocity_shift_applied = True
        self.rad_vel_value = velocity/1000.

    def write_column_spectrum(self, directory, use_opt_params=False):
        
        # make the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        if use_opt_params:
            filename = self.teff + '_' + self.logg + '_' + self.mh + '_' + str(self.vsini_value) + '_' + str(self.noise_value) + '_' + str(self.rad_vel_value) + '.tsv'
        else:
            filename = self.teff + '_' + self.logg + '_' + self.mh + '.tsv'

        with open(directory + '/' + filename,"a") as csvfile:
                rows = zip(np.round(self.wavelengths,6), np.round(self.fluxes,3), np.round(self.errors,3))
                for row in rows:
                    csvfile.write("{0}\t{1}\t{2}\n".format(*row))

    def PlotSpectrum(self, existing_plot=None):
        plt.plot(self.wavelengths, self.fluxes)
        plt.title('StellaNet Spectrum')
        plt.ylabel('Flux')
        plt.xlabel('Wavelength')
        plt.show()
