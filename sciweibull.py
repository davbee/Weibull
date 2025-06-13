# This script generates random samples from a Weibull distribution,
# plots the probability density function (PDF) of the Weibull distribution,
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min

# Define parameters
shape = 2.0  # Shape parameter; # beta
loc = 0  # Location parameter; gamma
scale = 1.0  # Scale parameter; alpha

# Create the distribution object
dist = weibull_min(shape, loc, scale)

# Generate random samples
num_samples = 1000
samples = dist.rvs(size=num_samples)
print(samples)
# Plot the probability density function (PDF)
x = np.linspace(0, 2.5, 100)
pdf = dist.pdf(x)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column
axs[0].plot(x, pdf, label="Weibull PDF", color="red")
axs[0].hist(
    samples,
    bins=30,
    density=True,
    alpha=0.6,
    color="green",
    label="Histogram of samples",
)
axs[0].set_xlabel("x")
axs[0].set_ylabel("Probability Density")
axs[0].set_title("Weibull Distribution")
axs[0].legend()
axs[0].grid(True)


# Create a second dataset for stats.weibull distribution
data = samples

# # Fit the data to a Weibull distribution
# params = stats.weibull_min.fit(data, floc=0)

# # Extract shape and scale
# shape, loc, scale = params

# Plot the histogram of the data and the fitted Weibull PDF
axs[1].plot(
    x,
    weibull_min.pdf(x, shape, loc, scale),
    color="blue",
    label="Weibull Distribution",
)
axs[1].hist(
    data,
    bins=30,
    density=True,
    alpha=0.6,
    color="orange",
    label="Histogram of samples",
)
axs[1].set_xlabel("x")
axs[1].set_ylabel("Probability Density")
axs[1].set_title("Weibull Distribution")
axs[1].legend()
axs[1].grid(True)

plt.show()
