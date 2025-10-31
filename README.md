# python3-ordinary_least_squares

### About

Python packaged designed for OLS inference.

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

### Building the Package

Build as a package from the source directory:

```bash
cd ./python-ordinary_least_squares

python -m build
```

Install with pip:

```bash
pip install ./dist/ordinary_least_squares-0.0.1-py3-none-any.whl
```

### Importing the Package

Import the package into your project

```python
from ordinary_least_squares import *
```

```python
import ordinary_least_squares as ols
```

```python
from ordinary_least_squares import LinearRegressionOLS, summary
```


### Model Fitting

`X`: `np.ndarray` or `pd.DataFrame` consisting of features and a column of ones for the intercept.

`y`: `np.ndarray` or `pd.Series` consisting of a target.

`feature_names`: List of strings consisting of names for the features. Use only when fitting models on `np.ndarray` objects. Do not include a name for the constant column. Enter names in order of columns in the array.

`target_name`: String consisting of a name for the target data. Use only when fitting models on `np.ndarray` objects. 

`alpha = 0.05`: Confidence interval for the model's initial predictions.

```python
model = LinearRegressionOLS().fit(
                                X=x,
                                y=y,
                                feature_names=None,
                                target_name=None,
                                alpha=0.05
                            )
```

### Example Usage

See:

```
/tests/linear_regression_example.ipynb
```

