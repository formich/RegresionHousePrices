import lib as lib
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt

# Importing csv datasets into pandas' DataFrame
train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

# In order to get better accuracy on the final prediction, I cut out the price column
# from the trainset and merge the two datasets together this result in a larger dataset and
# hopefully better estimations
whole_set = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                       test.loc[:, 'MSSubClass':'SaleCondition']))

#  Remove all features that contains too many NaN
lib.remove_nan(whole_set, threshold=0.4)

# Since Kaggle evaluates submissions using the logarithm of the predicted value
# makes sense to train the model with the logarithm of the prices
train["SalePrice"] = np.log(train["SalePrice"])

# Extract numerical features
numeric_features = whole_set.dtypes[whole_set.dtypes != "object"].index

# remove outliers distinguishing from train set and test set
outliers_train, outliers_test = lib.detect_outliers(whole_set[numeric_features])
whole_set = lib.remove_outliers(whole_set, outliers_train)
train = lib.remove_outliers(train, outliers_train)

# Here we are computing the skewness of the samples, if it's bigger than a specific
# threshold I will transform the features by applying log(feature + 1)
skewness_arr = lib.compute_skewness(train[numeric_features])
skewed_features = lib.select_skewed_features(skewness_arr)
whole_set[skewed_features] = np.log1p(whole_set[skewed_features])

# Generate dummy variables from the remaining categorical variables
whole_set = pd.get_dummies(whole_set)

# Filling NaN values with the mean of the column
whole_set = whole_set.fillna(whole_set.mean())

# Creating matrices and prediction vector to generate model
X = whole_set[:train.shape[0]]
X_test = whole_set[train.shape[0]:]
y = train.SalePrice

# This block commented right down here generates the plot of the score function what's in the PDF
# now is commented because is not requested to delivery and slow down a bit the execution
#
# Generate arrays to plot R^2 with respect of alpha
# test_range = np.arange(0.0001, 0.003, 0.00001)
# alpha_arr, r2_arr, opt_alpha, r2max = lib.opt_alpha_cv(X, y, test_range=test_range)
# cv_lasso = pd.Series(r2_arr, index=alpha_arr)
# # plot the cross validation
# lib.pplot(cv_lasso, opt_alpha, r2max)

# Computing optimum alpha through a cross validation process
opt_alpha = lib.opt_alpha_cv(X, y)[2]
# Instance the model
lasso = Lasso(alpha=opt_alpha, fit_intercept=True)
# Fit data
lasso.fit(X, y)

# Finally compute the prediction!
pred = pd.DataFrame({"SalePrice": np.exp(lasso.predict(X_test))}, index=test.index)
pred.to_csv("data/pred.csv")
