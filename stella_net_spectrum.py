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
    noise_applied = False, noise_value = 0, radial_velocity_shift_applied = False, radial_velocity_shift = 0, is_synthetic = True, teff = None, mh = None, logg = None, \
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


    def plot_spectrum(self, plot_continuum=False):
        if plot_continuum:
            plt.plot(self.wcont,self.fcont, 'bo')
            plt.plot(self.wavelengths, self.continuum)

        plt.plot(self.wavelengths, self.fluxes)
        plt.title('StellaNet Spectrum')
        plt.ylabel('Flux')
        plt.xlabel('Wavelength')
        plt.show()

    def cut_and_interpolate_fluxes_to_grid(self, shape, replace_nan=False, wavelengths=None):
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

        #waves.argmax(waves<300)
        #print()
        
        grid_waves = np.linspace(min_wave, max_wave, shape) # the ideal wavelength spacing
        fluxterp = np.interp(grid_waves, waves, fluxes) # calculate the interpolated fluxes
        #vals_before = int((min_wave-300)/0.01) + 1 # the number of ones to pad with at the beginning of the array
        #vals_after = int((700-max_wave)/0.01) # the number of ones to pad with at the end of the array
        #dont_pad = False
        #if (vals_before < 0): vals_before = 0
        #if (vals_after < 0): vals_after = 0
        #if ((min_wave < 300) or (max_wave < 700)):
            #grid_waves = np.pad(grid_waves, (vals_before, vals_after), mode='constant', constant_values = (0,0))
            #fluxterp = np.pad(fluxterp, (vals_before, vals_after), mode='constant', constant_values = (0,0))
        self.fluxes = fluxterp
        self.wavelengths = grid_waves


    def max_normalize(self):
        self.fluxes = self.fluxes/max(self.fluxes)

    def boxcar_smooth(self, points):
         self.fluxes = convolve(self.fluxes, Box1DKernel(points))

    @staticmethod
    def find_index(array,value):
        element = min(range(len(array)), key=lambda x:abs(array[x]-value))
        return(element)

    # break spectrum up into segments based on knot spacing and place spline knots at those locations, then linterp the continuum from the resulting
    # cubic spline and divide by that continuum
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

        

       