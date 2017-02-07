import lib1764939 as lib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 6.0)


# Importing dataset into pandas' DataFrame

train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)
whole_set = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                       test.loc[:, 'MSSubClass':'SaleCondition']))

#  Remove all features that contains too many NaN

lib.remove_nan(whole_set, threshold=0.4)

# Since Kaggle evaluates submissions using the logarithm of the predicted value
# makes sense to train the model with the logarithm of the prices

train["SalePrice"] = np.log(train["SalePrice"])

# Some features in the dataset are converted to numerical version

# Extract numerical features and remove from the list features that
# are no apparent highly price-related

numeric_features = whole_set.dtypes[whole_set.dtypes != "object"].index

# Here we are computing the skewness of the samples, if it's bigger than a specific
# threshold I will transform the features by applying log(feature + 1),
# this will make the features more "Normal"

skewness_arr = lib.compute_skewness(train[numeric_features])
skewed_features = lib.select_skewed_features(skewness_arr)
whole_set[skewed_features] = np.log1p(whole_set[skewed_features])

# Generate DummyVariable from the remaining categorical variable
whole_set = pd.get_dummies(whole_set)

# # Remove outliers
# lib.remove_outliers(whole_set)

# Filling NaN values with the mean of the column
whole_set = whole_set.fillna(whole_set.mean())


# Creating matrices and prediction vector to generate model
X = whole_set[:train.shape[0]]
y = train.SalePrice
X_test = whole_set[train.shape[0]:]

# Compute optimum alpha for Ridge regression with cross validation
opt_alpha, scores = lib.cv(X, y, test_range=100)

ridge = Ridge(alpha=opt_alpha, fit_intercept=True)
ridge.fit(X, y)

pred = pd.DataFrame({"SalePrice": np.exp(ridge.predict(X_test))}, index=test.index)
pred.to_csv("pred.csv")
