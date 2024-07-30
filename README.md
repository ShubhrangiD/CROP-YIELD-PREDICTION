# Crop Yield Prediction

This repository contains code for predicting crop yields using various machine learning models. The data used includes information about average rainfall, pesticide use, and average temperature for different countries and crops over the years. The goal is to build a predictive system that can estimate crop yields based on these features.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Save and Load Model](#save-and-load-model)
- [Contributing](#Contributing)

## Installation

To use this project, you need to have Python installed. You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Data

The data file `yield_df.csv` contains the following columns:
- Area
- Item
- Year
- hg/ha_yield
- average_rain_fall_mm_per_year
- pesticides_tonnes
- avg_temp

Load the data:

```python
import numpy as np
import pandas as pd

df = pd.read_csv('yield_df.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
```

## Preprocessing

Remove duplicates and handle missing or incorrect values:

```python
df.drop_duplicates(inplace=True)

def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return False

to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
df = df.drop(to_drop)
```

## Exploratory Data Analysis

### Checking the data:

```python
df.head()
df.shape
df.isnull().sum()
df.info()
```

### Basic Statistics:

```python
df.describe()
```

### Visualization:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,20))
sns.countplot(y=df['Area'])

plt.figure(figsize=(10,20))
sns.barplot(y=df['Area'].unique(), x=[df[df['Area']==state]['hg/ha_yield'].sum() for state in df['Area'].unique()])

sns.countplot(y=df['Item'])

plt.figure(figsize=(10,20))
sns.barplot(y=df['Item'].unique(), x=[df[df['Item']==crop]['hg/ha_yield'].sum() for crop in df['Item'].unique()])
```

## Model Training

Prepare the data for training:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

col = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area','Item','hg/ha_yield']
df = df[col]

x = df.drop('hg/ha_yield', axis=1)
y = df['hg/ha_yield']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

ohe = OneHotEncoder(drop='first')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehotencoder', ohe, [4, 5]),
        ('standarization', scaler, [0, 1, 2, 3])
    ],
    remainder='passthrough'
)

x_train_dummy = preprocessor.fit_transform(x_train)
x_test_dummy = preprocessor.transform(x_test)
```

Train multiple models and evaluate their performance:

```python
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    'lr': LinearRegression(),
    'lss': Lasso(),
    'rg': Ridge(),
    'knr': KNeighborsRegressor(),
    'dtr': DecisionTreeRegressor()
}

for name, model in models.items():
    model.fit(x_train_dummy, y_train)
    y_pred = model.predict(x_test_dummy)
    print(f"{name} MAE: {mean_absolute_error(y_test, y_pred)} R2: {r2_score(y_test, y_pred)}")
```

## Prediction

Use the Decision Tree Regressor model for prediction:

```python
dtr = DecisionTreeRegressor()
dtr.fit(x_train_dummy, y_train)

def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
    transformed_features = preprocessor.transform(features)
    predicted_value = dtr.predict(transformed_features).reshape(1, -1)
    return predicted_value[0]

result = prediction(2000, 59.0, 3024.11, 26.55, 'Saudi Arabia', 'Sorghum')
print(result)
```

## Save and Load Model

Save the trained model and preprocessor using pickle:

```python
import pickle

pickle.dump(dtr, open('dtr.pkl', 'wb'))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))
```

Load the model and preprocessor:

```python
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
```

##Contributing

Contributions are welcome! Please read the CONTRIBUTING file for guidelines on contributing to this project.

