import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import photolooper.rxn_ode_fitting as ode
import pandas as pd


def align_time(df):
    df["duration"] = (
        df["duration"] - df[df["status"] == "DEGASSING"]["duration"].values[0]
    )


def find_nearest(array, values):
    """
    Find the indices of the nearest values in the array to a list of target values.

    Parameters:
        array (numpy.ndarray): The input array.
        values (numpy.ndarray or scalar): The target values.

    Returns:
        list: A list of indices of the nearest values in the array to the target values.

    Note:
        If the input array has more than one dimension, the function flattens the first dimension.
        The function uses the `numpy.searchsorted` function to find the indices of the target values in the array.
        The function then checks if the index is not the last index in the array and if the difference between the target value and the previous value in the array is less than the difference between the target value and the current value in the array. If both conditions are true, the index of the previous value is returned, otherwise the index of the current value is returned.
    """

    if array.ndim != 1:
        array_1d = array[:, 0]
    else:
        array_1d = array

    values = np.atleast_1d(values)
    hits = []

    for i in range(len(values)):
        idx = np.searchsorted(array_1d, values[i], side="left")
        if idx > 0 and (
            idx == len(array_1d)
            or math.fabs(values[i] - array_1d[idx - 1])
            < math.fabs(values[i] - array_1d[idx])
        ):
            hits.append(idx - 1)
        else:
            hits.append(idx)

    return hits


def poly(a, x):
    y = a[0] * x**0

    for i in range(1, len(a)):
        y += a[i] * x**i

    return y


def residual_generic(p, x, y, function):
    y_fit = function(p, x)
    res = y - y_fit

    return res


def pre_signal_fitting(data, start, end, order_poly, plotting=False, filename=None):
    """Pre-signal fitting for baseline correction"""

    idx = find_nearest(data, np.array([start, end]))
    x = data[:, 0][idx[0] : idx[1]]
    y = data[:, 1][idx[0] : idx[1]]

    p_guess = np.ones(order_poly + 1)
    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, poly))
    p_solved = p.x

    y_baseline = poly(p_solved, data[:, 0])
    y_corrected = data[:, 1] - y_baseline

    if plotting:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".", markersize=2.0)
        ax.plot(data[:, 0], y_baseline, linewidth=1.0)
        ax.plot(data[:, 0], y_corrected)
        if filename is not None:
            fig.savefig(filename, dpi=400)

    data_corr = np.c_[data[:, 0], y_corrected]

    return data_corr


def first_order_combined(p, t):
    return p[0] - (p[0] * np.exp(-p[1] * t))


def first_order_shift(p, t):
    """Function with value zero for t < p[2], first order for t > p[2]"""

    idx = find_nearest(t, p[2])[0]

    t_base = t[: idx + 1]
    t_feature = t[idx + 1 :]

    y_base = t_base * 0
    y_feature = first_order_combined(p[:2], (t_feature - p[2]))

    return np.r_[y_base, y_feature]


def first_order_fitting_without_normalization(
    p_guess, data, plotting=False, filename=None
):
    """Fitting first_order_shift function to data, which has not been normalized"""

    x = data[:, 0]
    y = data[:, 1]

    p = least_squares(fun=residual_generic, x0=p_guess, args=(x, y, first_order_shift))
    y_fit = first_order_shift(p.x, x)

    if plotting:
        fig, ax = plt.subplots()
        ax.plot(x, y, ".")
        ax.plot(x, y_fit, linewidth=1)
        ax.set_xlabel("time / s")
        ax.set_ylabel("O2 / uM/L")
        if filename is not None:
            fig.savefig(filename, dpi=400)

    plt.close()

    return p.x


def preprocess_data(data_df, offset):
    """Pre-processing data by selecting relevant portion of data and then performing
    baseline correction based on pre-reaction phase.
    """

    align_time(data_df)

    # subset data to relevant statuses
    data_subset = data_df[data_df["status"].isin(["PREREACTION-BASELINE", "REACTION"])]
    data_subset = data_subset[
        data_subset["command"].isin(["LAMP-ON", "FIRESTING-START"])
    ]
    start = data_subset["duration"].values[0]
    end = data_subset[data_subset["status"] == "REACTION"]["duration"].values[0]
    time = data_subset["duration"].values
    o2_data = data_subset["uM_1"].values

    data_corrected = pre_signal_fitting(
        np.c_[time, o2_data], start, end, 2, plotting=False
    )

    rxn_subset = data_subset[data_subset["status"] == "REACTION"]

    rxn_start = find_nearest(time, rxn_subset["duration"].values[0] + offset)[0]
    rxn_end = find_nearest(time, rxn_subset["duration"].values[-1])[0]

    return data_subset, data_corrected, rxn_start, rxn_end


def plotting_fit_results(p, time_reaction, data_reaction, initial_state, matrix):
    """Plotting of fit results."""

    fig, ax = plt.subplots()

    random_value = np.random.rand()
    cmap = plt.get_cmap("plasma")
    random_color = cmap(random_value)

    y_fit = ode.ODE_matrix_fit_func(p, initial_state, time_reaction, matrix, idx=2)
    ax.plot(time_reaction, data_reaction, ".", color=random_color)
    ax.plot(time_reaction, y_fit, color=random_color)
    ax.set_xlabel("Time / s")
    ax.set_ylabel(r"Oxygen / $\mu$mol/L")

    return fig


def fit_data(
    data_df: pd.DataFrame,
    filename: str = None,
    offset: int = 0,
    reaction_string=["A > B, k1", "B > C, k2"],
    bounds=[[0, 1], [0.15, 0.15]],
    idx_for_rate_constant=0,
    idx_for_fitting=2,
    plotting: bool = False,
    return_full: bool = False,
):
    """
    Fitting data to arbitrary reaction model using 'rxn_ode_fitting' module.

    Reaction string is specified, in this case A >k1> B >k2> C, to model induction period due to O2 diffusion to sensor.
    Experimental data is fitted to concentration profile of C (idx_for_fitting = 2).
    k1 is optimized and k2 is fixed to 0.15 (bounds), k1 is returned (idx_for_rate_constant = 0).
    The bounds for k2 might have to be adjusted for other reactions, we will have to take a look at this with future data.
    return_full is for interfacing with other code for debugging.

    Args:
        data_df (pd.DataFrame):
            Pandas dataframe containing experimental data.
        filename (str):
            filename
        offset (int):
            Offset for fitting of experimental data (for example to skip induction period). However, using
            the offset seems to lower the quality of the fit.
        reaction_string (List[str]):
            Reaction string describing reaction sequence that is fitted to data.
        idx_for_rate_constant (int):
            idx to select rate constant from obtained array. Depending on reaction string, array has as many entries as
            there are k values. Default is that first rate constant is picked.
        idx_for_fitting (int):
            idx to select which part of model is fitted to experimental data. Default is that species "C" (idx = 2) is
            fitted to data.
        plotting (bool):
            Flag to control if plot is generated.
        return_full (bool):
            Flag to control is full output is returned (default = False).
    """

    data_subset, data_corrected, rxn_start, rxn_end = preprocess_data(
        data_df, offset=offset
    )

    time_reaction = (
        data_corrected[:, 0][rxn_start:rxn_end] - data_corrected[:, 0][rxn_start]
    )
    data_reaction = data_corrected[:, 1][rxn_start:rxn_end]

    idx = np.argmax(data_reaction)
    time_reaction = time_reaction[:idx]
    data_reaction = data_reaction[:idx]

    p, matrix, initial_state, _residual = ode.ODE_fitting(
        data_reaction,
        time_reaction,
        reaction_string,
        idx=idx_for_fitting,
        bounds_arr=bounds,
    )

    rate_constant = p[idx_for_rate_constant]

    if plotting is True:
        fig = plotting_fit_results(
            p, time_reaction, data_reaction, initial_state, matrix
        )
        if filename is not None:
            fig.savefig(filename, dpi=400)

    if return_full is True:
        return rate_constant, data_subset, data_corrected, rxn_start, rxn_end

    return rate_constant
