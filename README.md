# python3-ordinary_least_squares

### About

Python packaged designed for advanced OLS inference.


### Example Output


```
==================================================
OLS Regression Results
==================================================
Dependent:                     educ      Education
--------------------------------------------------
 
const                     7.3256***      7.3256***
                           (0.3684)       (0.3684)
 
paeduc                    0.2144***      0.2144***
                           (0.0241)       (0.0241)
 
maeduc                    0.2569***      0.2569***
                           (0.0271)       (0.0271)
 
age                       0.0241***      0.0241***
                           (0.0043)       (0.0043)

--------------------------------------------------
R-squared                     0.276          0.276
Adjusted R-squared            0.274          0.274
F Statistic                 177.548        177.548
Observations               1402.000       1402.000
Log Likelihood            -3359.107      -3359.107
AIC                        6726.213       6726.213
BIC                        6747.196       6747.196
==================================================
*p<0.1; **p<0.05; ***p<0.01
```

### Build and Install the Package

Build as a package from the source directory:

```bash
git clone https://github.com/axtaylor/python-ordinary_least_squares.git

cd ./python-ordinary_least_squares

python -m build

pip install ./dist/ordinary_least_squares-0.0.1-py3-none-any.whl
```

### Importing the Package

```python
from ordinary_least_squares import *
```

```python
import ordinary_least_squares as ols
```

```python
from ordinary_least_squares import LinearRegressionOLS, summary
```




## Basic Usage


### Model Fitting

```python
LinearRegressionOLS().fit(
                        X=X,
                        y=y,
                        feature_names=None,
                        target_name=None,
                        alpha=0.05
                        )
```

`X`: `np.ndarray` or `pd.DataFrame` consisting of features and a column of ones for the intercept.

`y`: `np.ndarray` or `pd.Series` consisting of a target.

`feature_names`: List of strings consisting of names for the features. Use only when fitting models on `np.ndarray` objects. Do not include a name for the constant column. Enter names in order of columns in the array.

`target_name`: String consisting of a name for the target data. Use only when fitting models on `np.ndarray` objects. 

`alpha = 0.05`: Confidence interval for the model's initial predictions.






---
### Model Predicting

```python
model.predict(X, alpha=0.05, return_table=False)
```

`X`: `np.ndarray` consisting of values for the models features,
in order. Do not include the intercept.

`alpha = 0.05`: Confidence interval for the model's prediction.

`return_table = False`: Returns a prediction as `np.float64()` when false, else returns a dictionary containing the prediction, standard error, t-statistic, p-value, and confidence ranges.





---
### Advanced Usage Documentation

See the Jupyter notebook for advanced use cases.

- Hypothesis testing on predictions using numeric values or feature arrays.
- Generating tables predicting the target value over a range of X values.
- Generating a table of the first derivative for a set of discrete predictions.
- Stacking multiple regression outputs.
- Testing models for multicolinearity.
- Applying robust standard errors.

```
/tests/linear_regression_example.ipynb
```







---
### Citation

If you use this package in your research, please cite:
```bibtex
@software{python-ordinary_least_squares,
  author = {Lucas Taylor},
  title = {ordinary_least_squares: OLS Regression for Python},
  year = {2025},
  url = {https://github.com/axtaylor/ordinary_least_squares}
}
```

