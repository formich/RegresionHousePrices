import pandas as pd
import numpy as np


# =======================================================================
#   Very simple function that take a pandas' dataframe and remove
#   any columns that exceed a certain ratio of NaN among its entries.
# =======================================================================
def remove_nan(dataframe, threshold=1/2):
    size = len(dataframe)
    dataframe.dropna(thresh=size - np.floor(size*threshold), axis=1, inplace=True)


