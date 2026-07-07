def compute_ylim(means, stds, upper, lower, target_fraction=0.28):
    """
    Compute y-axis limits so the RF band (upper - lower) occupies
    approximately target_fraction of the total plot height (between 1/4 and 1/3).
    The axis is always wide enough to show all data points with their error bars.
    """
    band_height = upper - lower
    data_min = (means - stds).min()
    data_max = (means + stds).max()
    data_span = data_max - data_min

    if band_height > 0:
        target_height = band_height / target_fraction
    else:
        target_height = max(data_span * 1.2, 0.01)

    height = max(target_height, data_span * 1.1)
    center = (data_min + data_max) / 2
    return center - height / 2, center + height / 2


def add_significance_bars(ax, model_order, y_start, best_model, sig_models):
    """
    Draw horizontal significance bars from `best_model` to each model in
    `sig_models`, each marked with an asterisk. The bars are stacked evenly
    within the headroom already reserved above `y_start`, up to the axis'
    current y-limit, so the caller is expected to have padded the y-limits
    beforehand to leave room for them.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on. Its x-axis is assumed to place categories at
        integer positions 0, 1, 2, ... in the order given by `model_order`.
    model_order : sequence of str
        Model names in the order they appear on the x-axis.
    y_start : float
        Data coordinate above which the first bar is drawn (e.g. the
        highest point reached by the error bars).
    best_model : str
        Name of the reference model (the one with the highest metric value).
    sig_models : list of str
        Models found to be significantly different from `best_model`;
        one bar is drawn to each.
    """
    if not sig_models:
        return

    x_positions = {model: i for i, model in enumerate(model_order)}
    best_x = x_positions[best_model]

    _, y1 = ax.get_ylim()
    bar_gap = (y1 - y_start) / (len(sig_models) + 1)

    for k, model in enumerate(sig_models):
        y = y_start + bar_gap * (k + 1)
        other_x = x_positions[model]
        x_lo, x_hi = sorted([best_x, other_x])
        ax.plot([x_lo, x_hi], [y, y], color="black", lw=1.3)
        ax.text((x_lo + x_hi) / 2, y, "*", ha="center", va="bottom", fontsize=17)
