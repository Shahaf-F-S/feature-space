# feature-space

> A module framework for constructing a network of features supporting each other, yet each feature is calculated only once.

## Installation

```
pip install feature-space
```

## examples

basic creation of features and datasets with multiple dependencies.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        key: np.random.random(100)
        for key in ('Open', 'High', 'Low', 'Close', 'Volume')
    }
)

print(df)
```

output
```
        Open      High       Low     Close    Volume
0   0.306962  0.090669  0.957007  0.382841  0.331181
1   0.668492  0.233647  0.601794  0.533531  0.761473
2   0.582980  0.765049  0.453987  0.989116  0.439396
3   0.053769  0.512395  0.763573  0.589263  0.886496
4   0.690432  0.372401  0.960555  0.202977  0.133927
..       ...       ...       ...       ...       ...
95  0.469604  0.591768  0.590435  0.138835  0.217345
96  0.304976  0.521499  0.006687  0.545035  0.974107
97  0.816594  0.639280  0.702651  0.942868  0.681855
98  0.387333  0.232820  0.563151  0.123126  0.051621
99  0.930279  0.657109  0.620474  0.794123  0.134324
```

creating the indicators with their relationships.
```python
from feature_space import Column, RSI, Change, Momentum

high = Column('High')
low = Column('Low')
close = Column('Close')

close_change = Change(close)
close_rsi_14 = RSI(close_change, 14)
close_momentum = Momentum(close_change, 35)
```

creating a dataset to contain and control the features.
using the dataset object is simple, but everything it does can be done 
with individual interactions with each feature.
```python
from feature_space import Dataset

change_indicators = Dataset(
    name='Change_Features',
    features=[close_change, close_rsi_14, close_momentum]
)

change_indicators.calculate(df)

df.dropna(inplace=True)

print(df)
```

output - all features are present in the dataframe, 
and each one was calculated only once, 
even though some features are required by more than one feature.
```
output
Open      High       Low     Close    Volume  Close_Change  Close_RSI_14  Close_Momentum_35
13  0.762020  0.053808  0.079920  0.061354  0.169120     -0.514332     45.932592          -0.321487
14  0.683689  0.948868  0.291903  0.461534  0.557272      0.400181     50.904070           0.078693
15  0.729113  0.352819  0.267228  0.923362  0.331447      0.461828     54.179768           0.540522
16  0.633024  0.931491  0.092854  0.910211  0.164508     -0.013152     49.065301           0.527370
17  0.321494  0.662967  0.253199  0.643929  0.810552     -0.266282     50.668724           0.261088
..       ...       ...       ...       ...       ...           ...           ...                ...
95  0.469604  0.591768  0.590435  0.138835  0.217345     -0.635993     43.876190          -0.447753
96  0.304976  0.521499  0.006687  0.545035  0.974107      0.406200     51.201515          -0.293790
97  0.816594  0.639280  0.702651  0.942868  0.681855      0.397833     57.625151           0.194520
98  0.387333  0.232820  0.563151  0.123126  0.051621     -0.819742     45.336996           0.061992
99  0.930279  0.657109  0.620474  0.794123  0.134324      0.670997     48.895303          -0.160722
```

to recalculate each feature one again, maby after a change in the original source data,
simply call the .clear() method on a Feature or a Dataset object.
This will not remove any data from the dataframe, just clear the cached referenses to the 
series in the features.

if you wish to override existing data, you can specify.

```python
change_indicators.clear()
change_indicators.calculate(df, override=True)
```

Save and load a whole dataset with its inter-dependency of features:
```python
change_indicators.save('dataset.pkl')
change_indicators = Dataset.load('dataset.pkl')
```