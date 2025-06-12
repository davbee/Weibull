import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import (ExponentialFitter, GeneralizedGammaFitter,
                       KaplanMeierFitter, LogLogisticFitter, LogNormalFitter,
                       PiecewiseExponentialFitter, SplineFitter, WeibullFitter)
from lifelines.datasets import load_waltons

# from lifelines.utils import median_survival_times


df = load_waltons()  # returns a Pandas DataFrame

"""
T is an array of durations, E is a either boolean or binary array representing
whether the “death” was observed or not (alternatively an individual can be
censored).
"""
T = df["T"]
E = df["E"]

# kmf = KaplanMeierFitter()

# kmf.fit(T, event_observed=E)  # or, more succinctly, kmf.fit(T, E)
# kmf.survival_function_
# kmf.cumulative_density_
# kmf.plot_survival_function()
# kmf.plot_cumulative_density()

# kmf.fit(T, E, timeline=range(0, 100, 2))
# kmf.survival_function_  # index is now the same as range(0, 100, 2)
# kmf.confidence_interval_  # index is now the same as range(0, 100, 2)
# kmf.plot_survival_function()
# kmf.plot_cumulative_density()


# median_ = kmf.median_survival_time_
# median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
# print("Median Survival Time:", median_)
# print("Median Survival Time Confidence Interval:", median_confidence_interval_)

fig, axes = plt.subplots(3, 3, figsize=(13.5, 7.5))

kmf = KaplanMeierFitter().fit(T, E, label="KaplanMeierFitter")
wbf = WeibullFitter().fit(T, E, label="WeibullFitter")
exf = ExponentialFitter().fit(T, E, label="ExponentialFitter")
lnf = LogNormalFitter().fit(T, E, label="LogNormalFitter")
llf = LogLogisticFitter().fit(T, E, label="LogLogisticFitter")
pwf = PiecewiseExponentialFitter([40, 60]).fit(T, E, label="PiecewiseExponentialFitter")
ggf = GeneralizedGammaFitter().fit(T, E, label="GeneralizedGammaFitter")
sf = SplineFitter(np.percentile(T.loc[E.astype(bool)], [0, 50, 100])).fit(
    T, E, label="SplineFitter"
)

wbf.plot_survival_function(ax=axes[0][0])
exf.plot_survival_function(ax=axes[0][1])
lnf.plot_survival_function(ax=axes[0][2])
kmf.plot_survival_function(ax=axes[1][0])
llf.plot_survival_function(ax=axes[1][1])
pwf.plot_survival_function(ax=axes[1][2])
ggf.plot_survival_function(ax=axes[2][0])
sf.plot_survival_function(ax=axes[2][1])

plt.show()

wbf.print_summary()

print(wbf.lambda_)


# Create example data
durations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
events = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
df = pd.DataFrame({"duration": durations, "event": events})

wf = WeibullFitter()
wf.fit(df["duration"], df["event"])
# wf.plot_survival_function()
wf.plot()
plt.show()
wf.print_summary()

waltons = load_waltons()
wbf = WeibullFitter()
wbf.fit(waltons["T"], waltons["E"])
wbf.plot()
plt.show()
print(wbf.lambda_)
