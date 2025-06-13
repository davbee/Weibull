# This code demonstrates how to fit a Weibull distribution
#  to a set of failure data
# and plot the survival function of both the fitted distribution
#  and the original distribution.

import matplotlib.pyplot as plt
import numpy as np
from reliability.Distributions import (Exponential_Distribution,
                                       Lognormal_Distribution,
                                       Weibull_Distribution)
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P
from reliability.Other_functions import (crosshairs, histogram,
                                         make_right_censored_data)
from reliability.Probability_plotting import plot_points

# https://reliability.readthedocs.io/en/latest/Quickstart%20for%20reliability.html#a-quick-example
# creates the distribution object
dist = Weibull_Distribution(alpha=30, beta=2, gamma=0)
print(dist)

# draws 20 samples from the distribution. Seeded for repeatability
data = dist.random_samples(20, seed=42)

plt.subplot(121)

# fits a Weibull distribution to the data and generates the probability plot
fit = Fit_Weibull_2P(failures=data)

plt.subplot(122)

# uses the distribution object from Fit_Weibull_2P and
# plots the survival function
fit.distribution.SF(label="fitted distribution")

# plots the survival function of the original distribution
dist.SF(label="original distribution", linestyle="--")

# overlays the original data on the survival function
plot_points(failures=data, func="SF")

plt.legend()
plt.grid(True)  # enables the grid on the plot
plt.show()


# https://reliability.readthedocs.io/en/latest/Creating%20and%20plotting%20distributions.html#example-1
dist = Weibull_Distribution(alpha=50, beta=2)  # this created the distribution object
dist.PDF()  # this creates the plot of the PDF
plt.grid(True)  # enables the grid on the plot
plt.show()


# https://reliability.readthedocs.io/en/latest/Creating%20and%20plotting%20distributions.html#example-3
sf = dist.SF(20)

# we are converting the decimal answer (0.8521...) to a percentage
print("The value of the SF at 20 is", round(sf * 100, 2), "%")


# https://reliability.readthedocs.io/en/latest/Creating%20and%20plotting%20distributions.html#example-4
xvals = np.linspace(0, 1000, 1000)
infant_mortality = Weibull_Distribution(alpha=400, beta=0.7).HF(
    xvals=xvals, label="infant mortality [Weibull]"
)
random_failures = Exponential_Distribution(Lambda=0.001).HF(
    xvals=xvals, label="random failures [Exponential]"
)
wear_out = Lognormal_Distribution(mu=6.8, sigma=0.1).HF(
    xvals=xvals, label="wear out [Lognormal]"
)
combined = infant_mortality + random_failures + wear_out
plt.plot(xvals, combined, linestyle="--", label="Combined hazard rate")
plt.legend()
plt.grid(True)  # enables the grid on the plot
plt.title(
    'Example of how multiple failure modes at different stages of\nlife can create a "Bathtub curve" for the total Hazard function'
)
plt.xlim(0, 1000)
plt.ylim(bottom=0)
plt.show()


# https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html#fitting-a-specific-distribution-to-data
data = [
    58,
    75,
    36,
    52,
    63,
    65,
    22,
    17,
    28,
    64,
    23,
    40,
    73,
    45,
    52,
    36,
    52,
    60,
    13,
    55,
    82,
    55,
    34,
    57,
    23,
    42,
    66,
    35,
    34,
    25,
]  # made using Weibull Distribution(alpha=50,beta=3)
wb = Fit_Weibull_2P(failures=data)
plt.grid(True)  # enables the grid on the plot
plt.show()


# https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html#example-2
data = Weibull_Distribution(alpha=25, beta=4).random_samples(30)
weibull_fit = Fit_Weibull_2P(
    failures=data, show_probability_plot=False, print_results=False
)
weibull_fit.distribution.SF(label="Fitted Distribution", color="steelblue")
plot_points(failures=data, func="SF", label="failure data", color="red", alpha=0.7)
plt.legend()
plt.grid(True)  # enables the grid on the plot
plt.show()


# https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html#example-3
a = 30
b = 2
g = 20
threshold = 55
# generate a weibull distribution
dist = Weibull_Distribution(alpha=a, beta=b, gamma=g)
# create some data from the distribution
raw_data = dist.random_samples(500, seed=2)
# right censor some of the data
data = make_right_censored_data(
    raw_data, threshold=threshold
)
print("There are", len(data.right_censored), "right censored items.")
# fit the Weibull_3P distribution
wbf = Fit_Weibull_3P(
    failures=data.failures,
    right_censored=data.right_censored,
    show_probability_plot=False,
    print_results=False,
)
print(
    "Fit_Weibull_3P parameters:\nAlpha:",
    wbf.alpha,
    "\nBeta:",
    wbf.beta,
    "\nGamma",
    wbf.gamma,
)
# generates the histogram using optimal bin width and
# shades the censored part as white
histogram(
    raw_data, white_above=threshold
)  
dist.PDF(label="True Distribution")  # plots the true distribution's PDF
wbf.distribution.PDF(
    label="Fit_Weibull_3P", linestyle="--"
)  # plots to PDF of the fitted Weibull_3P
plt.title("Fitting comparison for failures and right censored data")
plt.legend()
plt.show()


# https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html#example-5
dist = Weibull_Distribution(alpha=500, beta=6)
data = dist.random_samples(50, seed=1)  # generate some data
# this will produce the large table of quantiles below the first table of results
Fit_Weibull_2P(failures=data, quantiles=True, CI=0.8, show_probability_plot=False)
print("----------------------------------------------------------")
# repeat the process but using specified quantiles.
output = Fit_Weibull_2P(failures=data, quantiles=[0.05, 0.5, 0.95], CI=0.8)
# these points have been manually annotated on the plot using crosshairs
crosshairs()
plt.show()

# the values from the quantiles dataframe can be extracted using pandas:
lower_estimates = output.quantiles["Lower Estimate"].values
print("Lower estimates:", lower_estimates)

# alternatively, the bounds can be extracted from the distribution object
lower, point, upper = output.distribution.CDF(CI_y=[0.05, 0.5, 0.95], CI=0.8)
print("Upper estimates:", upper)
