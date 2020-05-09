#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import xgboost
import shap

data = pd.read_csv('listings_summary.csv')
data.columns

data.city.unique()

# Cast price as float
data.price = data.price.apply(lambda x: x.replace("$", ""))
data.price = data.price.apply(lambda x: x.replace(",", ""))
data.price = data.price.astype("float")
data.price.describe()

# Get rid of outliers
print("99.5% properties have a price lower than {0: .2f}".format(np.percentile(data.price, 99.5)))
data = data[(data.price <= np.percentile(data.price, 99.5)) & (data.price > 0)]

# Our first linear model
model = smf.ols('price ~ host_is_superhost + bedrooms + number_of_reviews + review_scores_rating', data=data).fit()

# Inspect the results
print(model.summary())

data.replace({'host_is_superhost' : {'t' : True, 'f' : False}}, inplace=True)

# Our first linear model
model = smf.ols('price ~ host_is_superhost + bedrooms', data=data).fit()

# Inspect the results
print(model.summary())

# Checking p-values meaning
N = 1000
x = np.zeros(N)
for i in range(N):
    df = pd.DataFrame({'x' : np.random.normal(size=30), 'y' : np.random.normal(size=30)})
    model = smf.ols('y ~ x', data=df).fit()
    x[i] = model.pvalues.x
sns.set_style('white')
sns.kdeplot(x, bw=0.01)
print('How often we get a coefficient which is as, or more extreme as the one we got?')
print(np.percentile(x, 5))

plt.scatter(np.random.normal(size=30), np.random.normal(size=30))


# Checking R^2 meaning
df = pd.DataFrame({'x' : [1,2,3,4,5,6], 'y' : [1,4,9,16,25,36]})
model = smf.ols('y ~ x', data=df).fit()
print(model.rsquared)
plt.scatter(x=df.x, y=df.y)


def rsquared(y_true, y_theo):
    y_true = np.array(y_true)
    y_theo = np.array(y_theo)
    return sum((y_theo-y_true.mean())**2)/sum((y_true-y_true.mean())**2)



df = pd.DataFrame({'x' : [1,2,3,4,5,6], 'y' : [1,4,9,16,25,36]})
df['x2'] = df.x**2
model = smf.ols('y ~ x + x2', data=df).fit()
model.rsquared

sns.kdeplot(np.random.normal(size=10000), bw=0.5)


# Lasso 

variables = ['bathrooms', 'bedrooms', 'beds', 'guests_included', 'minimum_nights', 'number_of_reviews', 'review_scores_rating']

data_subset = data.loc[:, variables + ['price']]
data_subset.dropna(inplace=True)

X = data_subset.loc[:,variables]
X = preprocessing.scale(X)
y = np.array(data_subset.price)

alphas_lasso, coefs_lasso, _ = linear_model.lasso_path(X, y, 5e-5, fit_intercept=False)

# Display results
plt.figure(figsize = (12,8))

for i in range(X.shape[1]):
    plt.plot(alphas_lasso, coefs_lasso[i], label = variables[i])

plt.xscale('log')
plt.xlabel('Log($\\lambda$)')
plt.ylabel('coefficients')
plt.title('Lasso paths - Sklearn')
plt.legend()
plt.axis('tight')

# XGBoost

variables = ['bathrooms', 'bedrooms', 'beds', 'guests_included', 'minimum_nights', 'number_of_reviews', 'review_scores_rating']

data_subset = data.loc[:, variables + ['price']]
data_subset.dropna(inplace=True)
data_subset = data_subset.iloc[:1000,:]

X = data_subset.loc[:,variables]
#X = preprocessing.scale(X) #We don't have to scale it as trees don't care about distributions, means, and all that
y = np.array(data_subset.price)

model = xgboost.XGBRegressor(n_estimators=10)
model.fit(X, y)


predicted = model.predict(X)
mean_squared_error(y, predicted)
#What does it even mean? Todo: check different error functions and how they can be explained
#Did we do a proper training and prediction? NO - we need to have proper TRAIN and TEST sets! Todo at home ;)


#Playing with Shapley values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


X.iloc[0,:]


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

shap.summary_plot(shap_values, X, plot_type="bar")

shap.summary_plot(shap_values, features=X, feature_names=X.columns)


#Sources:
#https://www.kaggle.com/yaowenling/berlin-airbnb-data-exploration-public
#https://xavierbourretsicotte.github.io/lasso_implementation.html

