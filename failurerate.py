import math

import matplotlib.pyplot as plt
import numpy as np
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points


# Function to calculate the Mean Time Between Failures (MTBF) or theta
def calculate_mtbf(total_operational_time, number_of_failures):
    """
    Calculates the Mean Time Between Failures (MTBF).

    Parameters:
    total_operational_time (float): Total operational time in hours.
    number_of_failures (int): Number of failures that occurred.

    Returns:
    float: MTBF value in hours. Returns inf if no failures occurred.
    """
    if number_of_failures == 0:
        return float("inf")  # Infinite MTBF if no failures occurred
    return total_operational_time / number_of_failures


# Function to calculate the failure rate or lamda
def calculate_failure_rate(total_operational_time, number_of_failures):
    """
    Calculates the failure rate (λ).

    Parameters:
    total_operational_time (float): Total operational time in hours.
    number_of_failures (int): Number of failures that occurred.

    Returns:
    float: Failure rate. Returns 0.0 if no failures occurred.
    """
    if number_of_failures == 0:
        return 0.0  # No failures means zero failure rate
    return number_of_failures / total_operational_time


def calculate_exp_reliability(lamda, time):
    """
    Calculates reliability based on the Exponential distribution.

    Parameters:
    lamda (float): Failure rate (λ).
    time (float): Time period over which reliability is calculated.

    Returns:
    float: Reliability value.
    """
    return math.exp(-time * lamda)


def calculate_weibull_reliability(lamda, time, slope):
    """
    Calculates reliability based on the Weibull distribution.

    Parameters:
    lamda (float): Failure rate (λ).
    time (float): Time period over which reliability is calculated.
    slope (float): Slope parameter of the Weibull distribution.

    Returns:
    float: Reliability value.
    """
    return math.exp(-time * lamda) ** slope


def main():
    """
    The main function calculates and prints reliability metrics for a system
    based on failure data and operational time.

    Steps:
    Defines the number of units in the system (units).
    Specifies the total operational time (operational_time) and warranty
    period (warranty_period).
    Provides failure times for failing units (failing_times).
    Calculates the number of failing and passing units.
    Computes the total operational time for all units (total_op_time).
    Calculates the failure rate (failure_rate) and MTBF (mtbf).
    Computes reliability rates using Exponential and Weibull distributions
    (exp_rel_rate and wb_rel_rate).
    Prints all calculated metrics.

    Returns:
    failure_rate (float): Failure rate (λ).
    mtbf (float): Mean Time Between Failures.
    exp_rel_rate (float): Reliability rate based on Exponential distribution.
    wb_rel_rate (float): Reliability rate based on Weibull distribution.
    """

    # The number of units in the system
    units = 20

    # Total operational time in hours
    operational_time = 1000

    # Example failure times in hours
    failing_times = [550, 480, 680, 790, 860, 620]
    # failing_times = [860, 920, 800]  # Example failure times in hours
    # failing_times = [0]  # Example failure times in hours

    # Warranty period in hours
    warranty_period = 200

    # The number of units that have failed
    failing_units = len(failing_times)

    # The number of units that have not failed
    passing_units = units - failing_units

    # Total operational time for failing units
    total_op_time = passing_units * operational_time + sum(failing_times)

    # scale parameter for Weibull distribution (alpha)
    scale_param = 2  # Example value for Weibull distribution

    # Generate operational times for passing units (e.g., fixed operational time of 1000 hours)
    passing_times = np.full(passing_units, 1000)

    # Generate operational times for failing units (e.g., random values between 500 and 900 hours)
    failing_times = np.array(failing_times)

    # Combine passing and failing operational times into a single array
    operational_times = np.concatenate((passing_times, failing_times))

    print(f"Number of Failing Units: {failing_units}")
    print(f"Number of Passing Units: {passing_units}")
    print(f"Failure Times: {failing_times}")
    print(f"Total Operational Time for Failing Units: {sum(failing_times)} hours")
    print(f"Total Operational Time for Passing Units: {passing_units * operational_time} hours")
    print(f"Total Operational Time for All Units: {total_op_time} hours")
    print(f"Warranty Period: {warranty_period} hours")
    print(f"Operational times for all units (hours): {operational_times}\n")

    failure_rate = calculate_failure_rate(total_op_time, failing_units)
    print(f"Failure Rate (lamda): {failure_rate:.6f} failures per hour")

    mtbf = calculate_mtbf(total_op_time, failing_units)
    print(f"MTBF (theta): {mtbf:.2f} hours")

    exp_rel_rate = calculate_exp_reliability(failure_rate, warranty_period)
    print(f"Exponential-distributed Reliability Rate: {exp_rel_rate:.6f}")

    wb_rel_rate = calculate_weibull_reliability(
        failure_rate, warranty_period, scale_param
    )
    print(f"Weibull-distributed Reliability Rate: {wb_rel_rate:.6f}\n")

    plt.subplot(121)
    fit = Fit_Weibull_2P(failures=failing_times)
    plt.xlabel("Time (hours)")
    plt.ylabel("Probability of Failure")
    plt.title("Weibull Probability Plot")

    plt.subplot(122)
    fit.distribution.SF(
        label="fitted distribution"
    )  # uses the distribution object from Fit_Weibull_2P and plots the survival function

    plot_points(
        failures=failing_times, func="SF"
    )  # overlays the original data on the survival function
    plt.xlabel("Time (hours)")
    plt.ylabel("Survival Function (SF)")

    plt.legend()
    plt.grid(True)  # enables the grid on the plot
    plt.show()

    return failure_rate, mtbf, exp_rel_rate, wb_rel_rate


if __name__ == "__main__":
    main()
