import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points


app = Flask(__name__)


def generate_graphs(failing_times):
    plt.subplot(121)
    fit = Fit_Weibull_2P(failures=failing_times)
    plt.xlabel("Time (hours)")
    plt.ylabel("Probability of Failure")
    plt.title("Weibull Probability Plot")

    plt.subplot(122)
    fit.distribution.SF(label="Fitted Distribution")
    plot_points(failures=failing_times, func="SF")
    plt.xlabel("Time (hours)")
    plt.ylabel("Survival Function (SF)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    graph_data = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return graph_data


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get inputs from the form
        units = int(request.form["units"])
        operational_time = int(request.form["operational_time"])
        failing_times = list(map(int, request.form["failing_times"].split(",")))
        warranty_period = int(request.form["warranty_period"])

        # Calculate metrics
        failing_units = len(failing_times)
        passing_units = units - failing_units
        total_op_time = passing_units * operational_time + sum(failing_times)
        scale_param = 2  # Example value for Weibull distribution

        failure_rate = failing_units / total_op_time if failing_units > 0 else 0.0
        mtbf = total_op_time / failing_units if failing_units > 0 else float("inf")
        exp_rel_rate = round(np.exp(-warranty_period * failure_rate) * 100, 2)
        wb_rel_rate = round(np.exp(-warranty_period * failure_rate) ** scale_param * 100, 2)

        # Generate graphs
        graph_data = generate_graphs(failing_times)

        # Render results
        return render_template(
            "results.html",
            units=units,
            operational_time=operational_time,
            failing_times=failing_times,
            warranty_period=warranty_period,
            failing_units=failing_units,
            passing_units=passing_units,
            total_op_time=total_op_time,
            failure_rate=failure_rate,
            mtbf=mtbf,
            exp_rel_rate=exp_rel_rate,
            wb_rel_rate=wb_rel_rate,
            graph_data=graph_data,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
