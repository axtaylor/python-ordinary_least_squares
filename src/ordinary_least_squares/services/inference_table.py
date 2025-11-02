import numpy as np

class InferenceTable:    

    def inference_table(model):
        return [
        {
            "feature": feature,
            'coefficient': (np.round(coefficient,4) if abs(coefficient) > 0.0001 else np.format_float_scientific(coefficient, precision=2)),
            'se': (np.round(se,4) if abs(se) > 0.0001 else np.format_float_scientific(se, precision=2)),
            't_statistic': np.round(t, 4),
            'P>|t|': f'{p:.3f}',
            f'conf_interval__{model.alpha}': [
                (np.round(low,3) if abs(low) > 0.0001 else np.format_float_scientific(low, precision=2)),
                (np.round(high,3) if abs(high) > 0.0001 else np.format_float_scientific(high, precision=2)),
            ],
        }
        for feature, coefficient, se, t, p, low, high in
        zip(model.feature_names, model.theta, model.std_error_coefficient, model.t_stat_coefficient, model.p_value_coefficient, model.ci_low, model.ci_high)
    ]