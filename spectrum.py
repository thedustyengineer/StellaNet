## @package spectrum
# File contains StellaNet format spectrum class definition

## StellaNet Spectrum object definition
class Spectrum:
   
    vsini_applied = False
    vsini_value = 0

    noise_applied = False
    noise_value = 0

    radial_velocity_shift_applied = False
    radial_velocity_shift = 0

    def __init__(self, wavelengths, fluxes, errors, vsini_applied = False, vsini_value = 0, \
    noise_applied = False, noise_value = 0, radial_velocity_shift_applied = False, radial_velocity_shift = 0):
    
        self.wavelengths = wavelengths
        self.fluxes = fluxes
        self.errors = errors

    def get_vsini_applied(self):
        return self.vsini_applied

    def get_noise_applied(self):
        return self.noise_applied

    def get_radial_velocity_shift_applied(self):
        return self.radial_velocity_shift_applied