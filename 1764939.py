import lib1764939 as lib
import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt


# =======================================================================
#    Importing dataset into pandas' DataFrame
# =======================================================================
train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)
whole_set = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))

# =======================================================================
#   Remove all features that contains too many NaN, if we don't have
#   enough information about some feature it could ruin the prediction
# =======================================================================
lib.remove_nan(whole_set, threshold=0.4)









