## @package stella_net_utilities
# File contains Perturbation and FileOperations classes that are used to support model augmentation and training

#!/usr/bin/env python

#local imports
import stella_net_config
import stella_net_exceptions
import stella_net_spectrum

#other imports
import distutils
import os
import logging
import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits # DOCUMENTATION HERE: http://docs.astropy.org/en/stable/io/fits/
import glob
import csv
from operator import itemgetter
import matplotlib.pyplot as plt
import random
import copy


# setup the logger
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
    #   http://nova.astro.umd.edu/index.html     Also see Eq. 17.12 in
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
    # @param velocity: the radial velocity in km/s
    # @return the updated StellaNet spectrum.Spectrum object. 
    # @note
    # Relativistic radial velocity correction is based on:
    # http://spiff.rit.edu/classes/phys314/lectures/doppler/doppler.html
    @staticmethod
    def apply_rad_vel_shift(spectrum, velocity):
        c = 299792458.0 # Speed of light (m/s)
        # relativistic wavelength correction
        velocity = (velocity*1000.) # convert velocity from km/s to m/s
        spectrum.wavelengths = spectrum.wavelengths * np.sqrt((1.-velocity/c)/(1.+velocity/c))
        return spectrum

## Methods used for file operations (ie reading spectrum fits files, writing files, converting files, etc)
class FileOperations:

    ## Reads a fits formatted spectrum file and outputs a spectrum.Spectrum object
    # @param file_path: the path to the file that is to be read
    # @param table_num: the table number of the .fits file containing the wavelength, flux, and error values
    # @param wave_header: the string header of the wavelength values (if wavelength obtained via CDELT1, CRVAL1, and NAXIS1 use '0')
    # @param flux_header: the string header of the flux values (use '0' if flux vals are all that is in data)
    # @param error_header: the string header of the error values (may be '0' to use zeroes)
    # @param parse_params: true if the application should parse parameters from the filename (default False)
    # @param read_range: the range of wavelengths to include in nm and the number of points (i.e. [400, 600]), 
    # can be None if you want to read all. First value is inclusive, second value is exclusive.
    # If the exact points do not exist then their closest match will be read (default None).
    # @param is_feros: indicates if the passed fits spectrum file is a FEROS file (in which case the wave, flux, and error
    # arrays are all [[]] arrays and need to be converted to [])
    # @return a StellaNet spectrum.Spectrum object generated from the file that was read
    # @note
    #   FEROS and HARPS .fits files require the headers to be set to 'WAVE', 'FLUX', 'ERR'
    #   and for the caller to set is_feros=True
    @staticmethod
    def read_fits_spectrum(file_path, table_num, wave_header=None, flux_header=None, error_header=None, parse_params = False, \
        read_range=None, is_feros=False):
        logger.info('Reading file at' + file_path)

        # the iSpec naming convention label indices
        teff_label_index = 0
        logg_label_index = 1
        mh_label_index = 2

        hdul = fits.open(file_path)          # open the current fits file
    
        if (flux_header != None):
            flux = np.array(hdul[table_num].data[flux_header])   # the flux
        else:
            flux = np.array(hdul[table_num].data)

        if (error_header != None):
            error = np.array(hdul[table_num].data[error_header])
        else:
            error = np.zeros_like(flux)

        if (wave_header != None):
            wave = np.array(hdul[table_num].data[wave_header])
        else:
            pix_size = hdul[table_num].header['CRPIX1']
            w_delta = hdul[table_num].header['CDELT1']
            start_value = hdul[table_num].header['CRVAL1']
            w_count = hdul[table_num].header['NAXIS1']
            wave = ((np.arange(w_count) + 1.0) - pix_size) * w_delta + start_value

        # convert [[]] to []
        if is_feros:
            wave = wave[0]
            flux = flux[0]
            error = error[0]

        # if the spectrum is in A convert to nm - this assumes the spectrum is visible wavelength range
        # you'll need to disable this if you create your own IR range neural network
        # and use this library for perturbations etc
        if min(wave) > 1000:
                wave = wave/10.0

        if parse_params:
            filename = os.path.basename(file_path).split('_')
            if (read_range != None):
                # find left and right boundary indices
                left_value_index = (np.abs(wave - read_range[0])).argmin()
                right_value_index = (np.abs(wave - read_range[1])).argmin()

                wave = wave[left_value_index:right_value_index]
                flux = flux[left_value_index:right_value_index]
                
                new_spectrum = stella_net_spectrum.Spectrum(wave, flux, error, teff = filename[teff_label_index], logg = filename[logg_label_index], \
                mh = filename[mh_label_index])
            else:
                new_spectrum = stella_net_spectrum.Spectrum(wave,flux,error)
        else:
            if (read_range != None):
                # find left and right boundary indices
                left_value_index = (np.abs(wave - read_range[0])).argmin()
                right_value_index = (np.abs(wave - read_range[1])).argmin()

                wave = wave[left_value_index:right_value_index]
                flux = flux[left_value_index:right_value_index]
                new_spectrum = stella_net_spectrum.Spectrum(wave, flux, error)
            else:
                new_spectrum = stella_net_spectrum.Spectrum(wave,flux,error)

        return new_spectrum


    ## Reads a tsv formatted spectrum file and outputs a spectrum.Spectrum object
    # @param file_path: the path to the file that is to be read
    # @param read_range: the range of wavelengths to include in nm and the number of points (i.e. [400, 600]), 
    # can be None if you want to read all. First value is inclusive, second value is exclusive.
    # If the exact points do not exist then their closest match will be read (default None).
    # @param has_errors: should be true if the file has an errors column (default false)
    # @param parse_params: true if the application should parse parameters from the filename
    # params are expected in the format [teff_logg_mh_vsini_snr_radvel.tsv]
    # @return a StellaNet spectrum.Spectrum object generated from the file that was read
    @staticmethod
    def read_tsv_spectrum(file_path, read_range=None, has_errors=False, parse_params = False):
        logger.info('Reading file at' + file_path)

        # the StellaNet naming convention label indices
        teff_label_index = 0
        logg_label_index = 1
        mh_label_index = 2
        vsini_label_index = 3
        snr_label_index = 4
        rad_vel_label_index = 5

        spectrum_file = np.loadtxt(file_path)

        wave = spectrum_file[:,0]
        flux = spectrum_file[:,1]

        if (read_range != None):
            # find left and right boundary indices
            left_value_index = (np.abs(wave - read_range[0])).argmin()
            right_value_index = (np.abs(wave - read_range[1])).argmin()

            wave = wave[left_value_index:right_value_index]
            flux = flux[left_value_index:right_value_index]

        if has_errors:
            error = spectrum_file[:,2]
        else:
            error=None
        
        if parse_params:
            filename = os.path.basename(file_path).replace('.tsv','').replace('.fits','').split('_')
            new_spectrum = stella_net_spectrum.Spectrum(wave, flux, error, teff = filename[teff_label_index], logg = filename[logg_label_index], \
                mh = filename[mh_label_index], vsini_value = filename[vsini_label_index], noise_value = filename[snr_label_index], \
                radial_velocity_shift = filename[rad_vel_label_index])
        else:
            new_spectrum = stella_net_spectrum.Spectrum(wave, flux, error)

        return new_spectrum

    ## Cuts a directory of grid files to a specified wavelength range, useful for preparing a grid for training
    # copies the input grid to another directory with the specified range, non-destructive of input grid
    # @param input_directory: the path to the files that are to be cut to the specified wavelengths
    # @param output_directory: the directory the new files will be output to
    # @param start_wave: the minimum wavelength value
    # @param end_wave: the maximum wavelength value
    @staticmethod
    def cut_directory(input_directory, output_directory, start_wave, end_wave):
        for file in os.listdir(input_directory):
            if '.tsv' in file:
                logger.info('Processing file: ' + file)
                this_spectrum = FileOperations.read_tsv_spectrum(input_directory + '/' + file, read_range=[start_wave, end_wave], parse_params=True)
                this_spectrum.write_column_spectrum(output_directory, use_opt_params=True)


    ## Writes a tab separated value (tsv) format spectrum from the given spectrum object
    # @param spectrum: the spectrum.Spectrum object to write to disk in .tsv format
    # @param file_path: the destination file path
    @staticmethod
    def write_column_spectrum(spectrum, file_path):
        with open(file_path,"a") as csvfile:
                rows = zip(spectrum.wavelengths, spectrum.fluxes)
                for row in rows:
                    csvfile.write("{0}\t{1}\n".format(*row))


    ## Reads a directory containing .tsv or  iSpec .fits formatted spectra and builds a Keras training dataset
    # @param directory: the path containing the fits format spectra files
    # @param label_index: the index in the filename that indicates the label
    # (iSpec format filenames, i.e. teff_logg_MH_alpha_vmic_vmac_vsini_limbdarkening
    # @param save_npy_binary_file: if True saves the x_train (data) and y_train (labels) arrays as npy binary files
    # in the source directory
    # @return x_train, y_train tuple (both numpy arrays) where x_train is the flux values 
    # and y_train is the data labels
    @staticmethod
    def build_dataset_from_grid_folder(directory, label_index, save_npy_binary_file=False):
        x_train = []
        y_train = []
        file_num = 0 # initialize current file number
        file_count = len([name for name in os.listdir(directory) if (os.path.isfile(directory + '/' + name) and ('.fits' in name or '.tsv' in name))])

        for file in os.listdir(directory):
            current_percent = (file_num/file_count) * 100
            if '.fits' in file:
                logger.info(str(int(current_percent)) + '%' + ' -- ' + 'File ' + str(file_num) + ' of ' + str(file_count) + ':' + file)
                this_spectrum = FileOperations.read_fits_spectrum(directory + '/' + file,0)
                x_train.append(this_spectrum.fluxes)
                filename = file.split('_')
                y_train.append([float(filename[0]),float(filename[1]),float(filename[2])])
                file_num+=1
            if ('.tsv' in file) and not ('._' in file):
                logger.info(str(int(current_percent)) + '%' + ' -- ' + 'File ' + str(file_num) + ' of ' + str(file_count) + ':' + file)
                this_spectrum = FileOperations.read_tsv_spectrum(directory + '/' + file, parse_params=True)
                x_train.append(this_spectrum.fluxes)
                y_train.append([float(this_spectrum.teff), float(this_spectrum.logg), float(this_spectrum.mh)])
                file_num+=1

        # save the .npy binary files for fast loading later
        if save_npy_binary_file:
            np.save(directory + '/x_train.npy', np.asarray(x_train))
            np.save(directory + '/y_train.npy', np.asarray(y_train))
            
        return x_train, y_train

    ## Load numpy binaries created by build_dataset_from_grid_folder
    # @param x_train_path: the path to the x_train.npy file containg the spectral data
    # @param y_train_path: the path to the y_train.npy file containing the data labels
    # @return x_train, y_train tuple (both numpy arrays) where x_train is the flux values 
    # and y_train is the data labels
    @staticmethod
    def build_dataset_from_npy_binaries(x_train_path, y_train_path):
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        return x_train, y_train

    ## Apply perturbations to a training grid to augment the grid
    # @param input_directory: the path to the grid you want to augment with perturbations
    # @param output_directory: the path where the augmented grid will be saved
    # @param vsini: apply vsini perturbations (default True)
    # @param snr: apply snr perturbations (default True)
    # @param rad_vel: apply rad_vel perturbations (default False), not necessary for convolution networks
    # @param use_random_perturbations: applies random perturbations in the range specified in the method instead of fixed perturbation values (default False)
    # @param output_wavelengths: specify wavelengths to cut the input grid files to (i.e. [400,525]) (default None)
    # @param normalize: apply stella_net_spectrum.Spectrum.normalize(normalize_spacing)
    # @param normalize_spacing: the spacing to use for normalization if normalize=True (default 2)
    @staticmethod
    def apply_perturbations(input_directory, output_directory, vsini=True, snr=True, rad_vel=False, \
        use_random_perturbations=False, output_wavelengths=None, normalize=True, normalize_spacing=2):
        if use_random_perturbations:
            # ranges for random value generation
            vsini_value_range = range(0,300) # generates random vsini values in the specified range
            snr_value_range = range(50,200) # generates random snr values in the specified range
            #rad_vel_value_range = range(-20,20) # generates random rad_vel values in the specified range

            # counts for random value generation
            num_rand_vsini = 10 # get 10 random values
            num_rand_snr = 10
            #num_rand_rad_vel = 1 # don't need random rad_vel for convolutional networks

            # random value generation
            vsini_values = random.sample(vsini_value_range, num_rand_vsini) # you can manually specify like [5,25,50,100,200] if you prefer
            snr_values = random.sample(snr_value_range, num_rand_snr)
            #rad_vel_values = random.sample(rad_vel_value_range, num_rand_rad_vel)

        else:
            vsini_values = [10,20,50,75,100,150,200]
            snr_values = [50,100,150,200]
            #rad_vel_values = [-10, -5, -3, 0, 3, 5, 10]

        file_num = 0 # initialize current file number
        file_count = len([name for name in os.listdir(input_directory) if (os.path.isfile(input_directory + '/' + name) and ('.fits' in name or '.tsv' in name))])
        for file in os.listdir(input_directory):

            current_percent = (file_num/file_count) * 100

            isGridFile = False

            # load fits
            if '.fits' in file:
                logger.info(str(int(current_percent)) + '%' + ' -- ' + 'File ' + str(file_num) + ' of ' + str(file_count) + ':' + file)
                raw_spectrum = FileOperations.read_fits_spectrum(input_directory + '/' + file, 0, parse_params=True, read_range=output_wavelengths)
                if normalize:
                    raw_spectrum.normalize(normalize_spacing)
                isGridFile = True

            # load tsv
            if '.tsv' in file:
                logger.info(str(int(current_percent)) + '%' + ' -- ' + 'File ' + str(file_num) + ' of ' + str(file_count) + ':' + file)
                raw_spectrum = FileOperations.read_tsv_spectrum(input_directory + '/' + file, parse_params=True, read_range=output_wavelengths)
                if normalize:
                    raw_spectrum.normalize(normalize_spacing)
                isGridFile = True

            if isGridFile:
                for vsini_value in vsini_values:
                    #for rad_vel_value in rad_vel_values:
                        for snr_value in snr_values:
                            this_spectrum = copy.deepcopy(raw_spectrum) # deep copy so that we don't apply overlapping operations
                            if vsini:
                                this_spectrum.apply_vsini(vsini_value)
                            #if rad_vel:
                                #this_spectrum.apply_rad_vel_shift(rad_vel_value)
                            if snr:
                                this_spectrum.apply_snr(snr_value)
                            this_spectrum.write_column_spectrum(output_directory, use_opt_params=True)
            file_num+=1

#FileOperations.apply_perturbations('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/grid','/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_400-525nm_normalized', output_wavelengths=[400,525])