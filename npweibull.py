# Generate samples from a Weibull distribution
import matplotlib.pyplot as plt
import numpy as np

# Parameters for the Weibull distribution
shape = 2.0  # Example shape parameter; beta
scale = 1.0  # Example scale parameter; alpha
samplenum = 1000

samples = np.random.weibull(shape, samplenum) * scale

plt.hist(
    samples,
    bins=30,
    density=True,
    alpha=0.6,
    color="blue",
    label="Histogram of samples",
)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Weibull Distribution")
plt.legend()
plt.grid(True)
plt.show()
