# StellaNet
An artificial convolutional neural network for the visible wavelength region of stellar spectra

For detailed information about this small, simple library, and an easy way to use the neural network, please visit <website here>

Quick Notes:

- If creating your own neural networks using StellaNet as a starting point, it is recommended to run the effective temperature, log g, and [M/H] branches of the neural network trainer separately, since they converge at different rates. The neural network included in the package works well for visible spectra with the grid I generated (detailed description on the StellaNet website, some brief details below).

- If your data set is not perturbed enough (i.e. the synthetic spectra you use do not cover a wide variety of SNR, vsini, metallicity, individual abundances, etc.) then your neural network will be very quickly overfit. The default models have been generated as follows:

- Effective temperatures: [3500K - 13000K] with step size 250K
- log g values: [2.0 - 5.0] with step size 0.5
- [M/H] values: [-3.00 - 0.5] with step size 0.25
- SNR values: 150, 300 (random gaussian noise)
- vsini values: 5, 50, 100, 200, 300
- Atomic abundances up to Fe were generated for each individual model as being a gaussian distribution around the [M/H] value with minima and maxima +- 1 from [M/H] (to prevent overfitting of [M/H])

This kind of distribution is recommended at a minimum to get a network that converges well for each parameter.

All synthetic spectra were generated with Kurucz/Castelli's model atmospheres (http://wwwuser.oats.inaf.it/castelli/grids.html) and the stellar spectral synthesis code used was SPECTRUM 2.76e by Richard Gray (http://www.appstate.edu/~grayro/spectrum/spectrum.html)

iSpec by Sergi Blanco-Cuaresma was modified to facilitate simple generation of the synthetic spectrum grid. I also referenced his code for the vsini perturbation function in StellaNet (which was itself derived from some older routines). I highly recommend you check out his work at https://www.blancocuaresma.com/s/iSpec

February 10, 2020
