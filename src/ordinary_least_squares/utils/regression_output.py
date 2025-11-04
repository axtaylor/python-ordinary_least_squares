def summary(*args):

    col_width, col_span, models = (
        15,
        20,
        list(args)
    )

    for i, model in enumerate(models):
        if model.theta is None:
            raise ValueError(f"Error: Model {i+1} is not fitted.")

    format_length = col_span + (len(models)*col_width)
    header = (
        f"\n{"="*format_length}\n"
        "OLS Regression Results\n"
        f"{"="*format_length}\n"
        f"{'Dependent:':<{col_span}}" + "".join(f"{m.target:>{col_width}}" for m in models) + "\n"
        f"{"-"*format_length}\n"
    )

    all_features = []
    for model in models:
        for feature in model.feature_names:
            if feature not in all_features:
                all_features.append(feature)

    rows = []
    for feature in all_features:
        coef_row = f"{feature:<{col_span}}"
        se_row = " " * col_span
        #t_row = " " * col_span

        for model in models:
            if feature in model.feature_names:
                feature_index = list(model.feature_names).index(feature)
                coef = model.theta[feature_index]
                se = model.std_error_coefficient[feature_index]
                p = model.p_value_coefficient[feature_index]
                #t = model.t_stat_coefficient[feature_index]
                stars = (
                    "***" if p < 0.01 else
                    "**" if p < 0.05 else
                    "*" if p < 0.1 else
                    ""
                )
                coef_fmt = (
                    f"{coef:.4f}{stars}"
                    if abs(coef) > 0.0001
                    else f"{coef:.2e}{stars}"
                )
                se_fmt = (
                    f"({se:.4f})"
                    if abs(se) > 0.0001
                    else f"({se:.2e})"
                )
                #t_fmt = f"{t:.4f}" if abs(t) > 0.0001 else f"({t:.2e})"
                coef_row += f"{coef_fmt:>{col_width}}"
                se_row += f"{se_fmt:>{col_width}}"
                #t_row += f"{t_fmt:>{col_width}}"
            else:
                coef_row += " " * col_width
                se_row += " " * col_width
                #t_row += " " * col_width

        rows.append(" ")
        rows.append(coef_row)
        rows.append(se_row)
        #rows.append(t_row)

    stats_lines = [
        ("R-squared", "r_squared"),
        ("Adjusted R-squared", "r_squared_adjusted"),
        ("F Statistic", "f_statistic"),
        ("Observations", lambda m: m.X.shape[0]),
        ("Log Likelihood", "log_likelihood"),
        ("AIC", "aic"),
        ("BIC", "bic")
    ]

    stats = f"\n{"-"*format_length}\n"

    for label, attr in stats_lines:
        stat_row = f"{label:<{col_span}}"
        for model in models:
            stat_row += f"{(attr(model) if callable(attr) else getattr(model, attr)):>{col_width}.3f}"
        stats += stat_row + "\n"

    return (
        header +
        "\n".join(rows) + "\n" +
        stats +
        f"{"="*format_length}\n"
        "*p<0.1; **p<0.05; ***p<0.01\n"
    )