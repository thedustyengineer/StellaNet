## @package spectrum
# File contains StellaNet format spectrum class definition

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
    #
    # @param snr: The desired signal to noise ratio
    def __init__(self, wavelengths, fluxes, errors, vsini_applied = False, vsini_value = 0, \
    noise_applied = False, noise_value = 0, radial_velocity_shift_applied = False, radial_velocity_shift = 0, is_synthetic = True):
    
        self.wavelengths = wavelengths
        self.fluxes = fluxes
        self.errors = errors