## @package stella_net_spectrum
# File contains StellaNet format Spectrum class definition

# local imports
from . import stella_net_config
from . import stella_net_exceptions

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
                    for row in rows:
                        csvfile.write("{0}\t{1}\t{2}\n".format(*row))
        else:
            with open(directory + '/' + filename,"a") as csvfile:
                    rows = zip(np.round(self.wavelengths,6), np.round(self.fluxes,3))
                    for row in rows:
                        csvfile.write("{0}\t{1}\n".format(*row))


    def plot_spectrum(self, existing_plot=None):
        plt.plot(self.wavelengths, self.fluxes)
        plt.title('StellaNet Spectrum')
        plt.ylabel('Flux')
        plt.xlabel('Wavelength')
        plt.show()

    def cut_and_interplate_fluxes_to_grid(self, wave_count, shape=27000, replace_nan=False):
        waves = np.asarray(self.wavelengths) # the current wavelength spacing
        fluxes = np.asarray(self.fluxes) # the current fluxes
        if replace_nan:
            for idx, flux in enumerate(fluxes):
                if (np.isnan(flux) or flux == 'nan'):
                    print('fixing nan')
                    fluxes[idx] = 1.0

        if (min(waves) > 1000): # check if in A or nm
            waves = waves/10 # convert to nm

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


    @staticmethod
    def find_index(array,value):
        element = min(range(len(array)), key=lambda x:abs(array[x]-value))
        return(element)

    # break spectrum up into segments based on knot spacing and place spline knots at those locations, then linterp the continuum from the resulting
    # cubic spline and divide by that continuum
    def spline_normalize(self, knot_window_spacing):
        waves = self.wavelengths
        fluxes = self.fluxes
        

       
        # Initialize arrays that will hold the x and y values of the continuum
        wcont=[]
        fcont=[]


        # initialize the first anchor window
        anchor_window = [min(waves), min(waves) + knot_window_spacing]
        while (anchor_window[1] < max(waves)):  # Find anchor position indices at the boundaries of the anchor window
            left_index = self.find_index(waves,anchor_window[0])
            right_index = self.find_index(waves,anchor_window[1])
            local_max = max(fluxes[left_index:right_index])
            anchor_index = self.find_index(fluxes[left_index:right_index],local_max)
            if (waves[anchor_index + left_index] not in wcont):
                fcont.append(fluxes[anchor_index + left_index]) # must add left index because find index returns index in range left_index:right_index
                wcont.append(waves[anchor_index + left_index])
            anchor_window  = [anchor_window[0] + knot_window_spacing, anchor_window[1] + knot_window_spacing]

        # add a point to the beginning and end of wcont so that the full wavelength range is covered
        #np.insert(wcont, 0, min(waves))
        #np.insert(fcont, 0, max(fluxes[0:20])) # set the first flux cont value to be the max on the left side of fluxes 
        #np.append(wcont, max(waves))
        #np.append(fcont, max(fluxes[fluxes.size-20:fluxes.size-1]))
        # Perform a spline fit on the anchor points to smooth out the continuum
        # spl = UnivariateSpline(wcont,fcont,s=0.8)

        # interpolate the continuum to match the wavelengths of the observed spectrum
        #fluxcont = np.interp(waves, wcont, fcont)
        
        spl = splrep(wcont,fcont,k=3)
        continuum = splev(waves,spl)

        self.fluxes = self.fluxes/continuum
        for flux in self.fluxes:
            if (np.isnan(flux) or flux == 'nan'):
                print('fixing nan')
                flux = 1.0

       