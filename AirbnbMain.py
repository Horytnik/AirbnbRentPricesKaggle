import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import xgboost
import shap

data = pd.read_csv('listings_summary.csv')
print(data.columns)
print(data.city.unique())
