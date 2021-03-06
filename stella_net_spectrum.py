## @package stella_net_spectrum
# File contains StellaNet format Spectrum class definition

# local imports
import stella_net_config
import stella_net_exceptions

# other imports
import distutils
import os
import logging
import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits # DOCUMENTATION HERE: http://docs.astropy.org/en/stable/io/fits/
import glob
import csv
import math
from operator import itemgetter
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import splrep,splev

# set up logging
logger = logging.getLogger('stella_net')

## StellaNet Spectrum object definition
class Spectrum:
   
    ## Object representing a stellar spectrum (or synthetic spectrum) and all of its associated parameters
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
    # @param is_synthetic: (default: None) bool indicating if the spectrum is synthetic or not
    # @param teff: (default: None) the effective temperature
    # @param mh: (default: None) the metallicity
    # @param logg: (default: None) the log g
    # @param continuum: (default: None) the continuum array
    # @param wcont: (default: None) the wavelengths used for the continuum
    # @param fcont: (default: None) the fluxes used for the continuum
    def __init__(self, wavelengths, fluxes, errors, vsini_applied = False, vsini_value = 0, \
                noise_applied = False, noise_value = 0, radial_velocity_shift_applied = False, \
                radial_velocity_shift = 0, is_synthetic = None, teff = None, mh = None, logg = None, \
                continuum=None, wcont=None, fcont=None):
    
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
        self.continuum = continuum
        self.wcont = wcont
        self.fcont = fcont
        
    ## Applies vsini broadening to the spectrum
    # @param vsini_value: The vsini that should be applied in km/s as a float value
    # @exception stella_net_exceptions.WavelengthSpacingError
    # @exception stella_net_exceptions.ParamTooSmallError
    # @exception stella_net_exceptions.VsiniAlreadyAppliedError
    # @note
    #   Adapted from iSpec by Sergi-Blanco Cuaresma https://www.blancocuaresma.com/s/iSpec
    #   which was adapted
    #   from lsf_rotate.pro:
    #   http://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro
    #   which was adapted from rotin3.f in the SYNSPEC software of Hubeny & Lanz
    #   http://nova.astro.umd.edu/index.html     Also see Eq. 17.12 in
    #   "The Observation and Analysis of Stellar Photospheres" by D. Gray (1992)
    def apply_vsini(self, vsini_value):
        
        logger.info('Applying vsini perturbation with value {}'.format(vsini_value))

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

    ## Applies gaussian noise to the spectrum
    # @param snr: The desired signal to noise ratio
    def apply_snr(self, snr):
        logger.info('Applying snr perturbation with value {}'.format(snr))
        if self.noise_applied:
            raise stella_net_exceptions.NoiseAlreadyAppliedError
        self.fluxes= self.fluxes/max(self.fluxes) + \
            np.random.normal(size=len(self.fluxes),scale=1.00/float(snr))
        self.noise_applied = True
        self.noise_value = snr
        self.noise_applied = True
        self.noise_value = snr
        
            
    ## Applies a radial velocity (red/blueshift) to the spectrum
    # @param velocity: the radial velocity in km/s
    # @exception stella_net_exceptions.RadVelAlreadyAppliedError
    # @note
    # Relativistic radial velocity correction is based on:
    # http://spiff.rit.edu/classes/phys314/lectures/doppler/doppler.html
    def apply_rad_vel_shift(self, velocity):
        logger.info('Applying rad vel shift perturbation with value {}'.format(velocity))
        if self.radial_velocity_shift_applied:
            raise stella_net_exceptions.RadVelAlreadyAppliedError

        c = 299792458.0 # Speed of light (m/s)
        # relativistic wavelength correction
        velocity = (velocity*1000.) # convert velocity from km/s to m/s
        self.wavelengths = self.wavelengths * np.sqrt((1.-velocity/c)/(1.+velocity/c))
        self.radial_velocity_shift_applied = True
        self.rad_vel_value = velocity/1000.

    ## Writes the spectrum to a tab separated value (.tsv) file
    # @param directory: the directory to write to
    # @param filename: the output filename, if none specified parameters will be used for the name (default: None) 
    # @param use_opt_params: if True will use vsini, snr, and rad vel in the filename in addition to teff, logg, and [M/H]
    def write_column_spectrum(self, directory, filename=None, use_opt_params=False):
        
        # make the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        if filename == None:
            if use_opt_params:
                filename = self.teff + '_' + self.logg + '_' + self.mh + '_' + str(self.vsini_value) + '_' + str(self.noise_value) + '_' + str(self.rad_vel_value) + '.tsv'
            else:
                filename = self.teff + '_' + self.logg + '_' + self.mh + '.tsv'

        if self.errors is not None:
            with open(directory + '/' + filename,"a") as csvfile:
                    rows = zip(np.round(self.wavelengths,6), np.round(self.fluxes,3), np.round(self.errors,3))
                    logger.info('Writing file ' + filename)
                    for row in rows:
                        csvfile.write("{0}\t{1}\t{2}\n".format(*row))
        else:
            with open(directory + '/' + filename,"a") as csvfile:
                    rows = zip(np.round(self.wavelengths,6), np.round(self.fluxes,3))
                    logger.info('Writing file ' + filename)
                    for row in rows:
                        csvfile.write("{0}\t{1}\n".format(*row))

    ## Plots the spectrum
    # @param plot_continuum: True if the continuum should also be plotted
    # @param plot_to_file: a path to save the plot to as a .png. If specified, the plot
    # will not be displayed, only saved to file (default: None)
    def plot_spectrum(self, plot_continuum=False, plot_to_file=None):
        if plot_continuum:
            plt.plot(self.wcont,self.fcont, 'bo')
            plt.plot(self.wavelengths, self.continuum)

        plt.plot(self.wavelengths, self.fluxes)
        plt.title('StellaNet Spectrum')
        plt.ylabel('Flux')
        plt.xlabel('Wavelength')
        if plot_to_file != None:
            plt.savefig(plot_to_file)
            plt.close()
        else:
            plt.show()

    ## Interpolates fluxes to the desired shape and, optionally, cuts the spectrum to specified wavelengths
    # @param shape: The number of points required by the model (can be calculated as [max(wave)-min(wave)]/wavelength spacing)
    # @param replace_nan: True to replace NaN values in the spectrum fluxes (default True)
    # @param wavelengths: If not None, the min and max wavelengths to cut the spectrum to (i.e. [400,525]) (default None)
    def cut_and_interpolate_fluxes_to_grid(self, shape, replace_nan=True, wavelengths=None):
        waves = np.asarray(self.wavelengths) # the current wavelength spacing
        fluxes = np.asarray(self.fluxes) # the current fluxes
        if replace_nan:
            for idx, flux in enumerate(fluxes):
                if (np.isnan(flux) or flux == 'nan'):
                    print('fixing nan')
                    fluxes[idx] = 1.0

        if (min(waves) > 1000): # check if in A or nm
            waves = waves/10 # convert to nm

        if wavelengths != None:
            # find left and right boundary indices
            left_value_index = (np.abs(waves - wavelengths[0])).argmin()
            right_value_index = (np.abs(waves - wavelengths[1])).argmin()

            waves = waves[left_value_index:right_value_index]
            fluxes = fluxes[left_value_index:right_value_index]

        min_wave = min(waves)
        max_wave = max(waves)

        grid_waves = np.linspace(min_wave, max_wave, shape) # the ideal wavelength spacing
        fluxterp = np.interp(grid_waves, waves, fluxes) # calculate the interpolated fluxes
       
        self.fluxes = fluxterp
        self.wavelengths = grid_waves

    ## Normalizes the spectrum by dividing by the maximum flux value
    def max_normalize(self):
        self.fluxes = self.fluxes/max(self.fluxes)

    ## Smooths the spectrum using a boxcar convolution
    # @param points: the number of points to smooth by
    def boxcar_smooth(self, points):
         self.fluxes = convolve(self.fluxes, Box1DKernel(points))

    ## Finds the index of the specified value in an array
    # @param array: the array to find the value for
    # @param value: the value to find
    # @return index: the index of the value in the array
    @staticmethod
    def find_index(array,value):
        element = min(range(len(array)), key=lambda x:abs(array[x]-value))
        return(element)

    ## Normalizes the spectrum using a cubic spline fit over local maxima in a sliding window
    # @param knot_window_spacing: the window spacing (i.e. if 5, place a knot at the maxima of each 5 nm window)
    # @param show_plot: show the plot after normalizing (default False)
    def normalize(self, knot_window_spacing, show_plot=False):
        waves = self.wavelengths
        fluxes_for_cont = self.fluxes
        
        # Initialize arrays that will hold the x and y values of the continuum
        wcont=[]
        fcont=[]

        h_regions = [(653,661), (481,491), (428,441), (406,415)]

        for idx, flux in enumerate(fluxes_for_cont):
            if not np.isnan(flux) and not flux == 'nan':
                last_non_nan = flux
            else:
                print('fixing nan in normalize')
                fluxes_for_cont[idx] = last_non_nan

        # box car smooth the flux by a lot to (mostly) eliminate the effects of noise
        fluxes_for_cont = convolve(fluxes_for_cont, Box1DKernel(50))

        # initialize the first anchor window
        anchor_window = [min(waves), min(waves) + knot_window_spacing]
        while (anchor_window[1] <= max(waves)):  # Find anchor position indices at the boundaries of the anchor window
            left_index = self.find_index(waves,anchor_window[0])
            right_index = self.find_index(waves,anchor_window[1])
            local_max = max(fluxes_for_cont[left_index:right_index])
            anchor_index = self.find_index(fluxes_for_cont[left_index:right_index], local_max)
            wcont_maybe = waves[anchor_index + left_index]
            if (wcont_maybe not in wcont) and not any(lower <= wcont_maybe <= upper for (lower, upper) in h_regions):
                fcont.append(local_max)
                wcont.append(wcont_maybe)
            else:
                if anchor_window[0] == min(waves): # handle the case where our local_max happened to be in the first balmer line region
                    local_max = local_max = max(fluxes_for_cont[left_index:right_index/2])
                    anchor_index = self.find_index(fluxes_for_cont[left_index:right_index], local_max)
                    wcont_maybe = waves[anchor_index + left_index]
                    fcont.append(local_max)
                    wcont.append(wcont_maybe)
            anchor_window  = [anchor_window[0] + knot_window_spacing, anchor_window[1] + knot_window_spacing]
        #if any(lower <= wcont_maybe <= upper for (lower, upper) in h_regions):
        fcont.append(fluxes_for_cont[len(waves)-50])
        wcont.append(waves[len(waves)-50])
        
        spline = splrep(wcont,fcont,k=3)
        spline_continuum = splev(self.wavelengths,spline)

        self.wcont = wcont
        self.fcont = fcont
        self.continuum = spline_continuum

        if show_plot:
            self.plot_spectrum(plot_continuum=True)

        self.fluxes = self.fluxes/spline_continuum

        return wcont, fcont, spline_continuum

        

       