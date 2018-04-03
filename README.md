# getSNmax
code to get a supernova (SN) max epoch and magnitude/flux with uncertainties via MC

The portio of a SN lightcurve near peak brightness has a simple parabolic shape. By selecting photometric measurements around peak we can find the date of maximum brightness and maximum flux/magnitude. This code allows the user to select the region of interest, generate realizations of the lightcurve based on the brightness measurements and uncertainties and fit a parabola to each realization. 

The epoch of maxumum and maximum brightness are the meadian of the respective distribution of values. 
Each realization is generated by drawing for every datapoint a value in a Gaussian distribution centered on the photometric measurement and with standard deviation equal to the measurement uncertainty. Notice that this is in princinciple, and for measurements that are free of systematics, correct for flux measurements. We also adopt a Gaussian distribution in magnitude space, although this is clearly not strictly correct.
The dependence of the result on the choice of region to fit is also explored by randomely dropping up to a few measurement at each edge of the region selected.

This method was first presented in [Bianco+2014](http://adsabs.harvard.edu/abs/2014ApJS..213...19B).
Compared to the work presented in Bianco+2014 there are a few changes:

1. The random draws are generated with the numpy random package. Earlier the python native random package was used, but te random draws were actually not drawn indepentently. This underestimates the uncertainty on the date of maximum, especially in the presence of large errorbars.
2. The maximum epoch and values are calculated as medians of the distributions, while they used to be calculated as means.
3. The uncertainties are reported as both standard deviation and percentiles (25th, 75th).


We enable the selection of the search region by stating how many datapoints have to be skipped from the beginning of the lightcurve and how many have to be used for the fit, either interactively or not, and by selecting the region graphically via matplotlib widget. 

The code does not require installation. It reads in the photometry as an ascii file, or using the SESNCfA library if it is set-up.

An example of the result of the code is below: the code prints the results to file, and plots the realizations, the fits, reporting the results on the plot.